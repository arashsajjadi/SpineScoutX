"""Morphology feature engine: structured geometry from anatomy-prior masks.

Input is the cached anatomy-prior tensor for one crop, ``[3, H, W]`` binary
channels in ``ANATOMY_PRIOR_CHANNELS`` order ``(disc, spinal_canal, vertebra)``.
For spinal-canal stenosis the canal calibre features are the clinically meaningful
ones: a stenotic canal is smaller / narrower, so ``canal_area``, ``min_canal_width``
and ``canal_disc_ratio`` are expected to fall with severity. Foraminal / subarticular
regions are approximate (see :func:`constants.evidence_region_for`).

Everything here is a deterministic function of the mask — no model, no randomness,
no GT coordinates. Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import ANATOMY_PRIOR_CHANNELS
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir

log = get_logger()

# channel order of the cached anatomy prior tensor
DISC, CANAL, VERT = (ANATOMY_PRIOR_CHANNELS.index(c) for c in ("disc", "spinal_canal", "vertebra"))

# feature columns produced per crop (stable order; used by the model feature head)
FEATURE_NAMES: tuple[str, ...] = (
    "disc_area",
    "canal_area",
    "vert_area",
    "canal_disc_ratio",
    "canal_vert_ratio",
    "min_canal_width",
    "mean_canal_width",
    "canal_width_cv",
    "canal_ap_extent",
    "canal_compactness",
    "canal_cx",
    "canal_cy",
    "canal_lr_asymmetry",
    "canal_present",
)
NUM_FEATURES = len(FEATURE_NAMES)


def _area_fraction(mask: np.ndarray) -> float:
    """Foreground fraction of a binary channel (∈ [0, 1])."""
    return float(mask.mean()) if mask.size else 0.0


def _row_widths(canal: np.ndarray) -> np.ndarray:
    """Per-row horizontal extent of the canal (in pixels) for rows that contain it."""
    widths = []
    for row in canal:
        xs = np.flatnonzero(row)
        if xs.size:
            widths.append(float(xs[-1] - xs[0] + 1))
    return np.asarray(widths, dtype=np.float32)


def morphology_features(anatomy_mask: np.ndarray) -> dict[str, float]:
    """Compute the structured morphology feature dict for one crop's anatomy mask.

    ``anatomy_mask`` is ``[3, H, W]`` (or ``[H, W]`` treated as canal-only). Values
    are thresholded at 0.5 so soft priors are handled. All features are normalised by
    crop size where it makes them scale-free, so they transfer across crop sizes.
    """
    arr = np.asarray(anatomy_mask, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([np.zeros_like(arr), arr, np.zeros_like(arr)], axis=0)
    binm = (arr >= 0.5).astype(np.float32)
    h, w = binm.shape[-2:]
    diag = float(np.hypot(h, w))

    disc, canal, vert = binm[DISC], binm[CANAL], binm[VERT]
    disc_a, canal_a, vert_a = _area_fraction(disc), _area_fraction(canal), _area_fraction(vert)

    feats: dict[str, float] = {
        "disc_area": disc_a,
        "canal_area": canal_a,
        "vert_area": vert_a,
        "canal_disc_ratio": canal_a / disc_a if disc_a > 1e-6 else 0.0,
        "canal_vert_ratio": canal_a / vert_a if vert_a > 1e-6 else 0.0,
        "min_canal_width": 0.0,
        "mean_canal_width": 0.0,
        "canal_width_cv": 0.0,
        "canal_ap_extent": 0.0,
        "canal_compactness": 0.0,
        "canal_cx": 0.0,
        "canal_cy": 0.0,
        "canal_lr_asymmetry": 0.0,
        "canal_present": float(canal_a > 0.0),
    }

    ys, xs = np.nonzero(canal)
    if xs.size == 0:
        return feats

    widths = _row_widths(canal)  # pixels, per occupied row
    if widths.size:
        feats["min_canal_width"] = float(widths.min()) / w
        feats["mean_canal_width"] = float(widths.mean()) / w
        feats["canal_width_cv"] = (
            float(widths.std() / widths.mean()) if widths.mean() > 1e-6 else 0.0
        )
    # anterior-posterior (vertical) extent of the canal column, scale-free
    feats["canal_ap_extent"] = float(ys.max() - ys.min() + 1) / h
    # compactness = area / bounding-box area  (1.0 = fills its bbox; low = irregular)
    bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    feats["canal_compactness"] = float(canal.sum() / bbox) if bbox > 0 else 0.0
    # centroid in [0,1] crop coordinates
    feats["canal_cx"] = float(xs.mean()) / w
    feats["canal_cy"] = float(ys.mean()) / h
    # left/right area asymmetry about the canal centroid (0 = symmetric)
    cx = int(round(xs.mean()))
    left = float(canal[:, :cx].sum())
    right = float(canal[:, cx:].sum())
    tot = left + right
    feats["canal_lr_asymmetry"] = abs(left - right) / tot if tot > 0 else 0.0
    # keep widths comparable across crop sizes via the diagonal too (unused col guard)
    _ = diag
    return feats


def feature_vector(anatomy_mask: np.ndarray) -> np.ndarray:
    """Return the morphology features as a fixed-order ``float32`` vector."""
    f = morphology_features(anatomy_mask)
    return np.asarray([f[name] for name in FEATURE_NAMES], dtype=np.float32)


def build_morphology_table(
    anatomy_cache: str | Path,
    crop_manifest: str | Path,
    out_cache: str | Path | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Compute morphology features for every crop with a cached anatomy prior.

    ``crop_manifest`` provides the (study, level, condition, severity) keys; the
    anatomy mask is read from ``anatomy_cache / crop_path``. Crops without a cached
    prior are skipped (logged), never fabricated. Optionally cached to parquet.
    """
    from ..data.crops import read_manifest

    anatomy_cache = Path(anatomy_cache)
    man = read_manifest(Path(crop_manifest))
    if limit is not None:
        man = man.head(int(limit))

    rows: list[dict[str, object]] = []
    missing = 0
    for r in man.itertuples():
        prior_path = anatomy_cache / str(r.crop_path)
        if not prior_path.exists():
            missing += 1
            continue
        feats = morphology_features(np.load(prior_path).astype(np.float32))
        row: dict[str, object] = {
            "crop_path": str(r.crop_path),
            "study_id": str(r.study_id),
            "level": str(r.level),
            "condition": str(r.condition),
            "severity": str(getattr(r, "severity", "")),
            "severity_index": int(getattr(r, "severity_index", -1)),
            "split": str(getattr(r, "split", "")),
            "coordinate_source": str(getattr(r, "coordinate_source", "oracle")),
        }
        row.update(feats)
        rows.append(row)

    if missing:
        log.warning("morphology: %d crops had no cached anatomy prior (skipped)", missing)
    frame = pd.DataFrame(rows)
    if out_cache is not None and len(frame):
        out = ensure_dir(out_cache)
        frame.to_parquet(out / "morphology.parquet", index=False)
    return frame
