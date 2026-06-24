"""Robust auto-inference training data: localizer-aware crop/slice jitter.

Phase 1 showed the oracle->auto severe-recall collapse is **in-plane (crop-centre)
driven**: a grader trained only on perfectly GT-centred crops is brittle to the
localizer's in-plane offset (median ~2.5 px but heavy-tailed, mean ~17 px in the real
auto path, worst at upper levels). This module exposes the grader to that
distribution at training time:

* :class:`LocalizerErrorProfile` — per-level in-plane residual + slice-offset
  distribution measured from the localizer (the jitter distribution to train against).
* :class:`JitterSampler` — draws ``(dx, dy, ds)`` offsets per level.
* :func:`build_canal_slice_cache` — one-time decode of the source slices needed to
  re-crop at a jittered centre (so jitter is genuine re-cropping, not translation of a
  pre-cut crop).
* :class:`RobustCanalCropDataset` — yields the standard severity item schema, cropping
  on the fly at the (jittered) centre; optionally yields a second jittered view for
  consistency regularization.

Determinism: a single seeded NumPy generator advances per ``__getitem__`` (use
``num_workers=0`` for exact reproducibility). Jitter is disabled (mode="none") ==> the
crop is identical to the oracle crop. No GT *severity* is ever used except as the
training target; no localizer coordinates are read at eval time.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..constants import (
    CONDITION_TO_INDEX,
    LEVEL_TO_INDEX,
    LEVELS,
    NUM_ANATOMY_PRIOR_CHANNELS,
)
from .crops import extract_25d


# --------------------------------------------------------------------------- #
# localizer error profile (the jitter distribution to train against)
# --------------------------------------------------------------------------- #
@dataclass
class LocalizerErrorProfile:
    """Per-level in-plane residual + slice-offset distribution of the localizer."""

    per_level_dxdy: dict[str, np.ndarray]  # level -> (N, 2) residual (auto - gt) px
    slice_offsets: np.ndarray  # (M,) mid_inst - gt_inst
    per_level_sigma: dict[str, tuple[float, float]]  # level -> (sigma_x, sigma_y)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"per_level": {}, "pooled": {}, "slice_offset": {}}
        pooled = (
            np.concatenate(list(self.per_level_dxdy.values()))
            if self.per_level_dxdy
            else np.zeros((0, 2))
        )
        for lv, arr in self.per_level_dxdy.items():
            d = np.linalg.norm(arr, axis=1) if len(arr) else np.zeros(0)
            out["per_level"][lv] = {
                "n": int(len(arr)),
                "sigma_x": float(self.per_level_sigma[lv][0]),
                "sigma_y": float(self.per_level_sigma[lv][1]),
                "median_px": float(np.median(d)) if len(d) else 0.0,
                "mean_px": float(np.mean(d)) if len(d) else 0.0,
                "p90_px": float(np.percentile(d, 90)) if len(d) else 0.0,
            }
        dp = np.linalg.norm(pooled, axis=1) if len(pooled) else np.zeros(0)
        out["pooled"] = {
            "n": int(len(pooled)),
            "median_px": float(np.median(dp)) if len(dp) else 0.0,
            "mean_px": float(np.mean(dp)) if len(dp) else 0.0,
            "p90_px": float(np.percentile(dp, 90)) if len(dp) else 0.0,
            "p99_px": float(np.percentile(dp, 99)) if len(dp) else 0.0,
        }
        so = self.slice_offsets
        out["slice_offset"] = {
            "n": int(len(so)),
            "mean": float(np.mean(so)) if len(so) else 0.0,
            "std": float(np.std(so)) if len(so) else 0.0,
            "frac_nonzero": float(np.mean(so != 0)) if len(so) else 0.0,
            "abs_p90": float(np.percentile(np.abs(so), 90)) if len(so) else 0.0,
        }
        return out


def build_localizer_error_profile(
    c1_gt_manifest: str | Path,
    c2_auto_manifest: str | Path,
    c3_mid_manifest: str | Path,
) -> LocalizerErrorProfile:
    """Measure the localizer residual distribution from the 2x2 cells.

    ``c1`` carries GT (x, y, instance); ``c2`` carries the localizer (auto) (x, y) at
    the GT slice; ``c3`` carries the geometric-mid instance. The residual is
    ``auto - gt`` in-plane and ``mid_inst - gt_inst`` for the slice offset.
    """
    c1 = pd.read_parquet(c1_gt_manifest)
    c2 = pd.read_parquet(c2_auto_manifest)
    c3 = pd.read_parquet(c3_mid_manifest)
    for f in (c1, c2, c3):
        f["study_id"] = f["study_id"].astype(str)
        f["series_id"] = f["series_id"].astype(str)
    keys = ["study_id", "series_id", "level"]
    m = c1.merge(c2, on=keys, suffixes=("_gt", "_auto"))
    per_level: dict[str, np.ndarray] = {}
    per_sigma: dict[str, tuple[float, float]] = {}
    for lv in LEVELS:
        g = m[m.level == lv]
        if len(g) == 0:
            per_level[lv] = np.zeros((0, 2), dtype=np.float64)
            per_sigma[lv] = (0.0, 0.0)
            continue
        dx = (g.x_auto - g.x_gt).to_numpy()
        dy = (g.y_auto - g.y_gt).to_numpy()
        arr = np.stack([dx, dy], axis=1)
        per_level[lv] = arr
        per_sigma[lv] = (float(np.std(dx)), float(np.std(dy)))
    ms = c1.merge(c3, on=keys, suffixes=("_gt", "_mid"))
    slice_off = (ms.instance_number_mid - ms.instance_number_gt).to_numpy().astype(int)
    return LocalizerErrorProfile(per_level, slice_off, per_sigma)


# --------------------------------------------------------------------------- #
# jitter sampler
# --------------------------------------------------------------------------- #
@dataclass
class CropJitterConfig:
    """Configuration for localizer-aware crop/slice jitter.

    mode:
      * ``none``        — no jitter (crop == oracle crop).
      * ``fixed``       — isotropic N(0, ``xy_sigma``) in-plane.
      * ``level_aware`` — per-level N(0, sigma_level) (sigmas from the profile),
                          plus a heavy-tail mixture (prob ``tail_prob`` -> N(0, ``tail_sigma``)).
      * ``empirical``   — sample an actual measured per-level residual ``(dx, dy)``.
    """

    mode: str = "none"
    xy_sigma: float = 2.5
    tail_prob: float = 0.15
    tail_sigma: float = 20.0
    slice_jitter: int = 0  # max |instance offset| sampled uniformly in [-k, k]
    max_offset: float = 60.0  # clamp |dx|,|dy| to avoid pathological crops
    seed: int = 1337

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class JitterSampler:
    """Draws ``(dx, dy, ds)`` offsets per level under a :class:`CropJitterConfig`."""

    def __init__(self, cfg: CropJitterConfig, profile: LocalizerErrorProfile | None = None):
        self.cfg = cfg
        self.profile = profile
        if cfg.mode in ("level_aware", "empirical") and profile is None:
            raise ValueError(f"jitter mode {cfg.mode!r} requires a LocalizerErrorProfile")

    def sample(self, level: str, rng: np.random.Generator) -> tuple[float, float, int]:
        c = self.cfg
        if c.mode == "none":
            return 0.0, 0.0, 0
        if c.mode == "fixed":
            dx, dy = rng.normal(0.0, c.xy_sigma, size=2)
        elif c.mode == "level_aware":
            sx, sy = self.profile.per_level_sigma.get(level, (c.xy_sigma, c.xy_sigma))
            sx = sx or c.xy_sigma
            sy = sy or c.xy_sigma
            if rng.random() < c.tail_prob:
                dx, dy = rng.normal(0.0, c.tail_sigma, size=2)
            else:
                dx, dy = rng.normal(0.0, sx), rng.normal(0.0, sy)
        elif c.mode == "empirical":
            pool = self.profile.per_level_dxdy.get(level)
            if pool is None or len(pool) == 0:
                dx, dy = rng.normal(0.0, c.xy_sigma, size=2)
            else:
                dx, dy = pool[rng.integers(0, len(pool))]
        else:
            raise ValueError(f"unknown jitter mode {c.mode!r}")
        dx = float(np.clip(dx, -c.max_offset, c.max_offset))
        dy = float(np.clip(dy, -c.max_offset, c.max_offset))
        ds = int(rng.integers(-c.slice_jitter, c.slice_jitter + 1)) if c.slice_jitter > 0 else 0
        return dx, dy, ds


# --------------------------------------------------------------------------- #
# source-slice cache (one-time decode so jitter is real re-cropping)
# --------------------------------------------------------------------------- #
def _slice_npy_rel(study: str, series: str, inst: int) -> str:
    return f"{study}_{series}_{inst}.npy"


def build_canal_slice_cache(
    rsna_root: str | Path,
    oracle_cache: str | Path,
    out_dir: str | Path,
    *,
    split: str = "train",
    window: int = 3,
) -> dict[str, Any]:
    """Decode + cache the source slices needed to re-crop canal nodes at a jittered
    centre. For each (study, series) caches instances within ``window`` of any GT
    canal instance (so xy-jitter + 2.5D neighbours + slice-jitter all have pixels).
    Resume-safe. Returns a summary; the per-node manifest is written alongside.
    """
    from ..utils.paths import ensure_dir
    from .crops import read_manifest
    from .dicom_io import normalize_intensity, read_dicom
    from .rsna_index import RsnaPaths

    rsna_root = Path(rsna_root)
    out_dir = Path(out_dir)
    ensure_dir(out_dir / "slices")
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)

    man = read_manifest(Path(oracle_cache) / "manifest.parquet")
    man = man[
        (man["condition"] == "spinal_canal_stenosis")
        & (man["split"] == split)
        & (man["severity_index"].isin([0, 1, 2]))
    ].copy()
    man["study_id"] = man["study_id"].astype(str)
    man["series_id"] = man["series_id"].astype(str)

    needed: dict[tuple[str, str], set[int]] = {}
    for r in man.itertuples():
        key = (str(r.study_id), str(r.series_id))
        needed.setdefault(key, set())
        for off in range(-window - 1, window + 2):  # +1 for 2.5D neighbour
            needed[key].add(int(r.instance_number) + off)

    decoded, missing = 0, 0
    for (study, series), insts in needed.items():
        for inst in insts:
            rel = _slice_npy_rel(study, series, inst)
            fp = out_dir / "slices" / rel
            if fp.exists():
                continue
            dp = images_dir / study / series / f"{inst}.dcm"
            if not dp.exists():
                missing += 1
                continue
            try:
                arr = normalize_intensity(read_dicom(dp)).astype(np.float32)
            except Exception:  # noqa: BLE001
                missing += 1
                continue
            np.save(fp, arr)
            decoded += 1

    man_out = out_dir / f"canal_{split}_nodes.parquet"
    man.to_parquet(man_out, index=False)
    summary = {
        "rsna_root": str(rsna_root),
        "oracle_cache": str(oracle_cache),
        "out_dir": str(out_dir),
        "split": split,
        "window": window,
        "n_nodes": int(len(man)),
        "n_series": len(needed),
        "decoded_slices": decoded,
        "missing_slices": missing,
        "node_manifest": str(man_out),
    }
    (out_dir / f"slice_cache_summary_{split}.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# robust dataset
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    study_id: str
    series_id: str
    instance_number: int
    level: str
    condition: str
    x: float
    y: float
    severity_index: int


class RobustCanalCropDataset(Dataset):
    """Canal severity dataset that re-crops 2.5D on the fly at a jittered centre.

    Yields the standard item schema (``image``, ``anatomy`` zeros, ``level_idx``,
    ``condition_idx``, ``target``, ``study_id``). With ``two_views=True`` also yields
    ``image2`` (an independently jittered view) for consistency regularization.
    """

    def __init__(
        self,
        node_manifest: pd.DataFrame,
        slice_cache_dir: str | Path,
        *,
        crop_size: int,
        jitter: JitterSampler,
        two_views: bool = False,
        seed: int = 1337,
    ) -> None:
        self.slice_dir = Path(slice_cache_dir) / "slices"
        self.crop_size = int(crop_size)
        self.jitter = jitter
        self.two_views = bool(two_views)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}
        self.nodes: list[_Node] = [
            _Node(
                study_id=str(r.study_id),
                series_id=str(r.series_id),
                instance_number=int(r.instance_number),
                level=str(r.level),
                condition=str(r.condition),
                x=float(r.x),
                y=float(r.y),
                severity_index=int(r.severity_index),
            )
            for r in node_manifest.itertuples()
        ]

    def __len__(self) -> int:
        return len(self.nodes)

    def _load_slice(self, study: str, series: str, inst: int) -> np.ndarray | None:
        rel = _slice_npy_rel(study, series, inst)
        if rel in self._cache:
            return self._cache[rel]
        fp = self.slice_dir / rel
        arr = np.load(fp).astype(np.float32) if fp.exists() else None
        self._cache[rel] = arr
        return arr

    def _crop(self, node: _Node, dx: float, dy: float, ds: int) -> np.ndarray:
        center = node.instance_number + ds
        slices: dict[int, np.ndarray] = {}
        for i in (center - 1, center, center + 1):
            a = self._load_slice(node.study_id, node.series_id, i)
            if a is not None:
                slices[i] = a
        if center not in slices:  # fall back to the un-jittered centre
            center = node.instance_number
            for i in (center - 1, center, center + 1):
                a = self._load_slice(node.study_id, node.series_id, i)
                if a is not None:
                    slices[i] = a
        arr, _ = extract_25d(slices, center, node.x + dx, node.y + dy, self.crop_size)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _item(self, node: _Node, image: np.ndarray) -> dict[str, Any]:
        return {
            "image": torch.from_numpy(image),
            "anatomy": torch.zeros(
                (NUM_ANATOMY_PRIOR_CHANNELS, self.crop_size, self.crop_size), dtype=torch.float32
            ),
            "level_idx": torch.tensor(LEVEL_TO_INDEX[node.level], dtype=torch.long),
            "condition_idx": torch.tensor(CONDITION_TO_INDEX[node.condition], dtype=torch.long),
            "target": torch.tensor(int(node.severity_index), dtype=torch.long),
            "study_id": node.study_id,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        node = self.nodes[index]
        dx, dy, ds = self.jitter.sample(node.level, self.rng)
        item = self._item(node, self._crop(node, dx, dy, ds))
        if self.two_views:
            dx2, dy2, ds2 = self.jitter.sample(node.level, self.rng)
            item["image2"] = torch.from_numpy(self._crop(node, dx2, dy2, ds2))
        return item
