"""Route-specific candidate bags for multi-instance (MIL) grading (v1.5).

A *bag* for one (study, level, side) finding is K candidate crops + per-candidate features,
instead of the single best crop. Foraminal bags = the top-K parasagittal sagittal-T1 slices (by
localizer confidence), each cropped at that level's localizer point. Subarticular bags = the
top-K axial-T2 slices for the level (by the axial scorer's P(level|slice)), each cropped at the
fixed paramedian offset. Reads **no** ground-truth coordinates (auto route only); GT severity is
the label only. Crops are saved float16 (K,3,224,224); bags are gitignored.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .axial_level import SUBARTICULAR_COND, SUBARTICULAR_OFFSETS
from .crops import extract_25d
from .dicom_io import normalize_intensity, read_dicom
from .foraminal_localize import FORAMINAL, side_candidate_instances, slices_by_lr

CROP_SIZE = 224


class SliceLRU:
    """Decode + cache normalized full slices for one series (so K crops share decodes)."""

    def __init__(self, images_dir: Path, study: str, series: str, max_items: int = 96) -> None:
        self.dir = images_dir / study / series
        self.max = max_items
        self._c: OrderedDict[int, np.ndarray | None] = OrderedDict()

    def get(self, inst: int) -> np.ndarray | None:
        if inst in self._c:
            self._c.move_to_end(inst)
            return self._c[inst]
        p = self.dir / f"{inst}.dcm"
        arr: np.ndarray | None = None
        if p.exists():
            with contextlib.suppress(Exception):
                arr = normalize_intensity(read_dicom(p)).astype(np.float32)
        self._c[inst] = arr
        if len(self._c) > self.max:
            self._c.popitem(last=False)
        return arr


def _crop25d(lru: SliceLRU, center: int, x: float, y: float) -> np.ndarray | None:
    slices: dict[int, np.ndarray] = {}
    for i in (center - 1, center, center + 1):
        a = lru.get(i)
        if a is not None:
            slices[i] = a
    if center not in slices:
        return None
    arr, _ = extract_25d(slices, center, x, y, CROP_SIZE)
    return arr.astype(np.float16)


def foraminal_bags(
    images_dir: Path,
    study: str,
    series: str,
    model,
    slice_size: int,
    device,
    level_labels: dict[tuple[str, str], Any],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Top-k parasagittal-T1 bags per (level, side); ``level_labels``: (side, level)->label row."""
    from .foraminal_localize import _localize_slice

    lr = slices_by_lr(images_dir, study, series)
    if len(lr) < 3:
        return []
    out: list[dict[str, Any]] = []
    for cond in FORAMINAL:
        side = "left" if cond.startswith("left") else "right"
        cands = side_candidate_instances(lr, side)
        scored = []  # (mean_conf, inst, pts, conf)
        lru = SliceLRU(images_dir, study, series)
        for inst in cands:
            pts, conf = _localize_slice(images_dir, study, series, inst, model, slice_size, device)
            if pts is None:
                continue
            scored.append((float(np.mean(conf)), int(inst), pts, conf))
        if not scored:
            continue
        scored.sort(key=lambda t: -t[0])
        top = scored[:k]
        from ..constants import LEVELS

        for li, lv in enumerate(LEVELS):
            if (side, lv) not in level_labels:
                continue
            crops, confs = [], []
            for _mc, inst, pts, conf in top:
                x, y = float(pts[li, 0]), float(pts[li, 1])
                c = _crop25d(lru, inst, x, y)
                if c is None:
                    continue
                crops.append(c)
                confs.append(round(float(conf[li]), 4))
            if not crops:
                continue
            out.append(
                {
                    "condition": cond,
                    "side": side,
                    "level": lv,
                    "crops": np.stack(crops),  # (k', 3, H, W) float16
                    "cand_conf": confs,
                    "label_row": level_labels[(side, lv)],
                }
            )
    return out


def subarticular_bags(
    images_dir: Path,
    study: str,
    series: str,
    logps: np.ndarray,
    zsorted: list[int],
    level_labels: dict[tuple[str, str], Any],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Top-k axial-T2 bags per (level, side) from the scorer's per-slice level log-probs."""
    from ..constants import LEVELS

    lru = SliceLRU(images_dir, study, series)
    out: list[dict[str, Any]] = []
    for li, lv in enumerate(LEVELS):
        order = np.argsort(-logps[:, li])  # slices most likely to be this level
        top_slices = [int(zsorted[r]) for r in order[:k]]
        top_scores = [round(float(np.exp(logps[r, li])), 4) for r in order[:k]]
        for side, cond in SUBARTICULAR_COND.items():
            if (side, lv) not in level_labels:
                continue
            ox, oy = SUBARTICULAR_OFFSETS[side]
            crops, confs = [], []
            for inst, sc in zip(top_slices, top_scores, strict=False):
                a = lru.get(inst)
                if a is None:
                    continue
                h, w = a.shape
                c = _crop25d(lru, inst, ox * w, oy * h)
                if c is None:
                    continue
                crops.append(c)
                confs.append(sc)
            if not crops:
                continue
            out.append(
                {
                    "condition": cond,
                    "side": side,
                    "level": lv,
                    "crops": np.stack(crops),
                    "cand_conf": confs,
                    "label_row": level_labels[(side, lv)],
                }
            )
    return out
