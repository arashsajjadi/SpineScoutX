"""Disc-level localizer data + heatmap utilities (auto-crop, no GT at inference).

The localizer predicts the 5 lumbar disc-level keypoints (L1/L2…L5/S1) on a
mid-sagittal T2 slice. Training supervision comes from the RSNA spinal-canal
localizer coordinates; **at inference no GT coordinates are read** — the sagittal
T2 series is chosen from the series index and the mid slice is taken geometrically.

Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import LEVELS
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir

log = get_logger()

COORDS_CSV = "train_label_coordinates.csv"  # the file the auto path must NOT read


def gaussian_heatmaps(points: np.ndarray, size: int, sigma: float = 4.0) -> np.ndarray:
    """Return ``(K, size, size)`` Gaussian heatmaps peaked at each ``(x, y)`` point.

    A point with a NaN coordinate yields an all-zero channel (missing keypoint).
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    out = np.zeros((len(points), size, size), dtype=np.float32)
    for i, (x, y) in enumerate(points):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        out[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    return out


def extract_peaks(heatmaps: np.ndarray) -> np.ndarray:
    """Return ``(K, 2)`` ``(x, y)`` argmax peak per heatmap channel."""
    k = heatmaps.shape[0]
    pts = np.zeros((k, 2), dtype=np.float32)
    for i in range(k):
        flat = int(np.argmax(heatmaps[i]))
        y, x = np.unravel_index(flat, heatmaps[i].shape)
        pts[i] = (x, y)
    return pts


def peak_confidence(heatmaps: np.ndarray) -> np.ndarray:
    """Per-channel peak value (∈[0,1] after sigmoid) as a localization confidence."""
    return heatmaps.reshape(heatmaps.shape[0], -1).max(axis=1).astype(np.float32)


def pck(pred: np.ndarray, gt: np.ndarray, thresholds: tuple[int, ...]) -> dict[str, float]:
    """Percentage of Correct Keypoints at pixel thresholds (ignores NaN GT points)."""
    valid = np.isfinite(gt).all(axis=1)
    if valid.sum() == 0:
        return {f"pck@{t}": float("nan") for t in thresholds}
    d = np.linalg.norm(pred[valid] - gt[valid], axis=1)
    return {f"pck@{t}": float((d <= t).mean()) for t in thresholds}


def _mid_instance(series_dir: Path) -> int | None:
    """Geometric middle DICOM instance number of a series (no GT involved)."""
    insts = sorted(int(p.stem) for p in series_dir.glob("*.dcm") if p.stem.isdigit())
    return insts[len(insts) // 2] if insts else None


def prepare_localizer_data(
    rsna_root: str | Path,
    out_cache: str | Path,
    *,
    slice_size: int = 256,
    val_fraction: float = 0.2,
    seed: int = 1337,
    limit_studies: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Cache mid-sagittal-T2 slices + 5 disc-level keypoints for localizer training.

    Uses GT canal coordinates ONLY to build training targets (this is the supervised
    prep step). Returns a JSON-able summary. Resume-safe.
    """
    import cv2

    from .dicom_io import normalize_intensity, read_dicom
    from .rsna_index import RsnaPaths, build_series_index
    from .rsna_labels import load_coordinates
    from .splits import patient_level_split

    rsna_root = Path(rsna_root)
    out = Path(out_cache)
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)

    coords = load_coordinates(rsna_root)
    series = build_series_index(rsna_root)
    canal = coords[coords.condition == "spinal_canal_stenosis"].merge(
        series[["study_id", "series_id", "sequence_type"]], on=["study_id", "series_id"], how="left"
    )
    canal = canal[canal.sequence_type == "sagittal_t2"]

    studies = sorted(canal.study_id.unique())
    if limit_studies is not None:
        studies = studies[: int(limit_studies)]
    split_map = patient_level_split(studies, val_fraction, seed)

    summary: dict[str, object] = {
        "rsna_root": str(rsna_root),
        "out_cache": str(out),
        "slice_size": int(slice_size),
        "n_candidate_studies": len(studies),
        "split_counts": {s: int(sum(v == s for v in split_map.values())) for s in ("train", "val")},
    }
    if dry_run:
        summary["dry_run"] = True
        return summary

    ensure_dir(out / "slices")
    rows: list[dict[str, object]] = []
    skipped = 0
    for study in studies:
        g = canal[canal.study_id == study]
        series_id = str(g.series_id.iloc[0])
        inst = int(g.instance_number.median())
        dpath = images_dir / study / series_id / f"{inst}.dcm"
        if not dpath.exists():
            skipped += 1
            continue
        try:
            img = normalize_intensity(read_dicom(dpath))
        except Exception as exc:  # noqa: BLE001 - decode failures logged, never faked
            log.warning("localizer decode failed %s: %s", dpath, exc)
            skipped += 1
            continue
        h, w = img.shape
        sx, sy = slice_size / w, slice_size / h
        resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
        rel = f"slices/{study}.npy"
        np.save(out / rel, resized.astype(np.float32))
        # GT keypoints (scaled to slice_size); NaN where a level is missing.
        kpts = np.full((len(LEVELS), 2), np.nan, dtype=np.float32)
        for _, r in g.iterrows():
            li = LEVELS.index(r.level)
            kpts[li] = (float(r.x) * sx, float(r.y) * sy)
        row: dict[str, object] = {
            "study_id": study,
            "series_id": series_id,
            "instance_number": inst,
            "slice_path": rel,
            "orig_h": int(h),
            "orig_w": int(w),
            "split": split_map.get(study, "train"),
            "n_levels": int(np.isfinite(kpts).all(axis=1).sum()),
        }
        for li, lv in enumerate(LEVELS):
            row[f"kx_{lv}"], row[f"ky_{lv}"] = float(kpts[li, 0]), float(kpts[li, 1])
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "localizer_manifest.parquet", index=False)
    summary["n_cached"] = len(frame)
    summary["skipped"] = skipped
    summary["split"] = (
        {s: int((frame["split"] == s).sum()) for s in ("train", "val")} if len(frame) else {}
    )
    return summary


class LocalizerDataset:
    """Torch dataset of (mid-sagittal slice, 5 disc-level Gaussian heatmaps)."""

    def __init__(
        self, manifest_df: pd.DataFrame, cache_root: str | Path, slice_size: int, sigma: float = 4.0
    ) -> None:
        self.frame = manifest_df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.slice_size = int(slice_size)
        self.sigma = float(sigma)

    def __len__(self) -> int:
        return len(self.frame)

    def keypoints(self, row: pd.Series) -> np.ndarray:
        return np.array([[row[f"kx_{lv}"], row[f"ky_{lv}"]] for lv in LEVELS], dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        row = self.frame.iloc[index]
        img = np.load(self.cache_root / str(row["slice_path"])).astype(np.float32)
        kpts = self.keypoints(row)
        heat = gaussian_heatmaps(kpts, self.slice_size, self.sigma)
        return {
            "image": torch.from_numpy(img[None]),
            "heatmap": torch.from_numpy(heat),
            "keypoints": torch.from_numpy(kpts),
            "study_id": str(row["study_id"]),
        }
