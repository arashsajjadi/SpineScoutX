"""SPIDER dataset indexing, volume loading, and label remapping.

The SPIDER lumbar-spine MRI dataset provides anatomical segmentation masks
(vertebrae, intervertebral discs, spinal canal). This module discovers
image/mask pairs, loads volumes via optional medical-imaging readers, and
collapses SPIDER's raw label ids into the 4-class anatomy scheme defined in
:mod:`spinescoutx.constants`.

Research-only: SPIDER masks are anatomy, not pathology.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..constants import (
    ANATOMY_CLASS_TO_INDEX,
    FOREGROUND_ANATOMY_CLASSES,
)
from ..utils.logging import get_logger

log = get_logger()

# File extensions we treat as volumes (single-file NIfTI or MetaImage).
_VOLUME_SUFFIXES: tuple[str, ...] = (".nii", ".nii.gz", ".mha", ".mhd")
# Substrings (case-insensitive) marking a file as a segmentation mask.
_MASK_HINTS: tuple[str, ...] = ("mask", "seg", "label")


@dataclass
class AvailabilityReport:
    """Filesystem availability summary (shape shared with ``rsna_index``)."""

    root: str
    exists: bool
    present: dict[str, bool]
    missing: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a plain-dict view (JSON-serialisable)."""
        return {
            "root": self.root,
            "exists": self.exists,
            "present": dict(self.present),
            "missing": list(self.missing),
            "notes": list(self.notes),
        }


@dataclass
class SpiderPaths:
    """Resolved expected sub-paths under a SPIDER dataset root."""

    root: str
    images_dir: str
    masks_dir: str

    @classmethod
    def from_root(cls, spider_root: str | Path) -> SpiderPaths:
        """Resolve conventional ``images``/``masks`` sub-directories."""
        root = Path(spider_root)
        return cls(
            root=str(root),
            images_dir=str(root / "images"),
            masks_dir=str(root / "masks"),
        )


def check_spider_available(spider_root: str | Path) -> AvailabilityReport:
    """Pure filesystem availability check for a SPIDER root (never raises)."""
    paths = SpiderPaths.from_root(spider_root)
    root = Path(paths.root)
    candidates: dict[str, str] = {
        "images_dir": paths.images_dir,
        "masks_dir": paths.masks_dir,
    }
    present: dict[str, bool] = {}
    missing: list[str] = []
    for key, value in candidates.items():
        ok = Path(value).exists()
        present[key] = ok
        if not ok:
            missing.append(key)
    notes: list[str] = []
    if not root.exists():
        notes.append(f"SPIDER root does not exist: {root}")
    return AvailabilityReport(
        root=str(root),
        exists=root.exists(),
        present=present,
        missing=missing,
        notes=notes,
    )


def _is_volume_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _VOLUME_SUFFIXES)


def _stem(path: Path) -> str:
    """Return a filename stem with all volume suffixes stripped (handles .nii.gz)."""
    name = path.name.lower()
    for suffix in sorted(_VOLUME_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _looks_like_mask(path: Path) -> bool:
    name = path.name.lower()
    return any(hint in name for hint in _MASK_HINTS)


def _guess_modality(path: Path) -> str:
    """Best-effort modality tag from the filename (case-insensitive)."""
    name = path.name.lower()
    if "t2" in name:
        return "t2"
    if "t1" in name:
        return "t1"
    return "unknown"


def build_spider_index(spider_root: str | Path) -> pd.DataFrame:
    """Pair SPIDER image volumes with masks by filename stem.

    Columns: ``[subject_id, image_path, mask_path, modality]``. Files are paired
    by their suffix-stripped filename stem; mask files are identified either by a
    naming hint (``mask``/``seg``/``label``) or by living under the masks dir.
    """
    paths = SpiderPaths.from_root(spider_root)
    images_dir = Path(paths.images_dir)
    masks_dir = Path(paths.masks_dir)

    image_files: list[Path] = []
    mask_files: list[Path] = []

    if images_dir.exists():
        for p in sorted(images_dir.rglob("*")):
            if p.is_file() and _is_volume_file(p):
                (mask_files if _looks_like_mask(p) else image_files).append(p)
    if masks_dir.exists():
        for p in sorted(masks_dir.rglob("*")):
            if p.is_file() and _is_volume_file(p):
                mask_files.append(p)

    mask_by_stem: dict[str, Path] = {}
    for m in mask_files:
        mask_by_stem.setdefault(_stem(m), m)

    rows: list[dict[str, object]] = []
    for img in sorted(image_files):
        stem = _stem(img)
        mask = mask_by_stem.get(stem)
        rows.append(
            {
                "subject_id": stem,
                "image_path": str(img),
                "mask_path": "" if mask is None else str(mask),
                "modality": _guess_modality(img),
            }
        )

    return pd.DataFrame(rows, columns=["subject_id", "image_path", "mask_path", "modality"])


def load_volume(path: str | Path) -> np.ndarray:
    """Load a medical-imaging volume as a float32 array.

    Tries SimpleITK first, then nibabel. Both are optional dependencies; if
    neither is importable a clear :class:`ImportError` is raised. Returns a
    ``float32`` array shaped ``(H, W)`` or ``(D, H, W)``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Volume not found: {p}")

    sitk_error: Exception | None = None
    try:
        import SimpleITK as sitk  # noqa: N813  (lazy optional import)
    except ImportError as exc:
        sitk_error = exc
    else:
        image = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(image)
        return np.asarray(arr, dtype=np.float32)

    nib_error: Exception | None = None
    try:
        import nibabel as nib  # lazy optional import
    except ImportError as exc:
        nib_error = exc
    else:
        loaded = nib.load(str(p))
        arr = np.asarray(loaded.get_fdata(), dtype=np.float32)
        return arr

    raise ImportError(
        "Loading SPIDER volumes requires SimpleITK or nibabel, but neither is "
        "installed. Install one, e.g. `pip install spinescoutx[medio]` "
        f"(SimpleITK import failed: {sitk_error}; nibabel import failed: {nib_error})."
    )


# --- SPIDER raw label -> 4-class remap -----------------------------------------
# SPIDER label convention (APPROXIMATE; documented assumption):
#   * vertebrae are encoded with ids in the low integer range (1..99);
#   * the spinal canal is encoded as id 100;
#   * intervertebral discs are encoded with ids >= 200.
# We collapse these into our scheme {0 background, 1 vertebra, 2 disc,
# 3 spinal_canal}. This mapping is an APPROXIMATION of the SPIDER labelling and
# may need revision for specific dataset releases; it is intentionally defensive
# (unknown / negative ids fall through to background).
_VERTEBRA_IDX = ANATOMY_CLASS_TO_INDEX["vertebra"]
_DISC_IDX = ANATOMY_CLASS_TO_INDEX["disc"]
_CANAL_IDX = ANATOMY_CLASS_TO_INDEX["spinal_canal"]
_CANAL_RAW_ID = 100
_DISC_RAW_MIN = 200
_VERTEBRA_RAW_MIN = 1
_VERTEBRA_RAW_MAX = 99


def remap_spider_labels(mask: np.ndarray) -> np.ndarray:
    """Collapse raw SPIDER label ids into the 4-class anatomy scheme.

    Output classes: ``0`` background, ``1`` vertebra, ``2`` disc,
    ``3`` spinal_canal (see :data:`spinescoutx.constants.ANATOMY_CLASSES`).

    APPROXIMATE mapping (see module comment): canal == id 100; disc == id >= 200;
    vertebra == ids 1..99 excluding 100. Pure numpy and unit-testable on
    synthetic integer label arrays.
    """
    raw = np.asarray(mask)
    ids = np.rint(raw).astype(np.int64)
    out = np.zeros(ids.shape, dtype=np.int64)

    vertebra = (ids >= _VERTEBRA_RAW_MIN) & (ids <= _VERTEBRA_RAW_MAX) & (ids != _CANAL_RAW_ID)
    disc = ids >= _DISC_RAW_MIN
    canal = ids == _CANAL_RAW_ID

    out[vertebra] = _VERTEBRA_IDX
    out[disc] = _DISC_IDX
    out[canal] = _CANAL_IDX
    return out


# --- 2D slice caching (volume -> cached crop_size slices + seg_index) ------------
def _patient_id(subject_id: str) -> str:
    """Patient id = leading integer of the subject stem (groups t1/t2 together)."""
    m = re.match(r"(\d+)", str(subject_id))
    return m.group(1) if m else str(subject_id)


def _slice_axis(shape: tuple[int, ...]) -> int:
    """Through-plane (sagittal) axis = the smallest dimension for lumbar MRI."""
    return int(np.argmin(shape))


def _resolve_subject_split(
    index: pd.DataFrame,
    official_split_csv: str | Path | None,
    val_fraction: float,
    seed: int,
    patient_level_split,
) -> tuple[dict[str, str], str]:
    """Map each ``subject_id`` to "train"/"val" plus a label for the split source.

    Prefers SPIDER's official ``overview.csv`` ``subset`` column (``training`` ->
    train, ``validation``/``test`` -> val) so results are comparable to the public
    benchmark; falls back to a deterministic seeded patient-level split.
    """
    if official_split_csv is not None and Path(official_split_csv).exists():
        overview = pd.read_csv(official_split_csv)
        name_col = "new_file_name" if "new_file_name" in overview.columns else overview.columns[0]
        if "subset" in overview.columns:
            mapping = {
                str(r[name_col]): (
                    "train" if str(r["subset"]).lower().startswith("train") else "val"
                )
                for _, r in overview.iterrows()
            }
            subject_split = {str(sid): mapping.get(str(sid), "val") for sid in index["subject_id"]}
            return subject_split, "spider_official_overview_csv"

    split_map = patient_level_split(sorted(index["patient_id"].unique()), val_fraction, seed)
    subject_split = {
        str(sid): split_map.get(str(pid), "train")
        for sid, pid in zip(index["subject_id"], index["patient_id"], strict=False)
    }
    return subject_split, "seeded_patient_level"


def cache_spider_slices(
    spider_root: str | Path,
    out_cache: str | Path,
    *,
    crop_size: int = 256,
    modalities: tuple[str, ...] = ("t1", "t2"),
    min_foreground: float = 0.002,
    val_fraction: float = 0.2,
    seed: int = 1337,
    limit_subjects: int | None = None,
    official_split_csv: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Decode SPIDER volumes into cached 2D slices + a patient-split ``seg_index``.

    For each image/mask volume we pick the sagittal (through-plane) axis, keep
    slices whose remapped mask has at least ``min_foreground`` foreground fraction,
    robustly normalize the image, resize image (area) and mask (nearest) to
    ``crop_size``, and save ``.npy`` pairs. Returns a JSON-able summary; with
    ``dry_run`` no files are written.

    Split is **patient-level** (t1/t2 of a subject stay together). If
    ``official_split_csv`` (SPIDER ``overview.csv``) is given, its ``subset``
    column is honored (``training`` -> train, else -> val); otherwise a
    deterministic seeded patient split is used.
    """
    from ..utils.paths import ensure_dir
    from .dicom_io import normalize_intensity
    from .splits import patient_level_split

    out = Path(out_cache)
    index = build_spider_index(spider_root)
    if modalities:
        index = index[index["modality"].isin(modalities)].reset_index(drop=True)
    index = index[index["mask_path"].astype(str) != ""].reset_index(drop=True)
    if len(index) == 0:
        raise FileNotFoundError(
            f"No paired SPIDER image/mask volumes found under {spider_root}. "
            "Check that images/ and masks/ contain matching .mha files."
        )
    index["patient_id"] = index["subject_id"].map(_patient_id)

    patients = sorted(index["patient_id"].unique(), key=lambda s: (len(s), s))
    if limit_subjects is not None:
        patients = patients[: int(limit_subjects)]
        index = index[index["patient_id"].isin(patients)].reset_index(drop=True)

    subject_split, split_source = _resolve_subject_split(
        index, official_split_csv, val_fraction, seed, patient_level_split
    )

    summary: dict[str, object] = {
        "spider_root": str(spider_root),
        "out_cache": str(out),
        "crop_size": int(crop_size),
        "modalities": list(modalities),
        "split_source": split_source,
        "n_patients": len(patients),
        "n_volumes": int(len(index)),
        "volume_split": {
            s: int(sum(subject_split.get(str(sid), "train") == s for sid in index["subject_id"]))
            for s in ("train", "val")
        },
    }
    if dry_run:
        summary["dry_run"] = True
        return summary

    images_out = ensure_dir(out / "images")
    masks_out = ensure_dir(out / "masks")
    rows: list[dict[str, object]] = []
    class_px = np.zeros(len(ANATOMY_CLASS_TO_INDEX), dtype=np.int64)
    skipped_volumes = 0

    for _, row in index.iterrows():
        subject = str(row["subject_id"])
        image_vol = load_volume(row["image_path"])
        mask_vol = remap_spider_labels(load_volume(row["mask_path"]))
        if image_vol.shape != mask_vol.shape:
            log.warning(
                "shape mismatch for %s: img %s vs mask %s; skipping",
                subject,
                image_vol.shape,
                mask_vol.shape,
            )
            skipped_volumes += 1
            continue
        if image_vol.ndim == 2:
            image_vol = image_vol[None]
            mask_vol = mask_vol[None]
        axis = _slice_axis(image_vol.shape)
        n_slices = image_vol.shape[axis]
        split = subject_split.get(subject, "train")
        for i in range(n_slices):
            m2d = np.ascontiguousarray(np.take(mask_vol, i, axis=axis)).astype(np.int64)
            if m2d.size == 0 or (np.count_nonzero(m2d) / m2d.size) < min_foreground:
                continue
            img2d = normalize_intensity(np.take(image_vol, i, axis=axis).astype(np.float32))
            img_r = cv2.resize(img2d, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            m_r = cv2.resize(
                m2d.astype(np.uint8), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            rel_img = f"images/{subject}_{i:03d}.npy"
            rel_mask = f"masks/{subject}_{i:03d}.npy"
            np.save(images_out / f"{subject}_{i:03d}.npy", img_r.astype(np.float32))
            np.save(masks_out / f"{subject}_{i:03d}.npy", m_r)
            rows.append(
                {
                    "subject_id": subject,
                    "patient_id": str(row["patient_id"]),
                    "modality": str(row["modality"]),
                    "slice_idx": int(i),
                    "image_path": rel_img,
                    "mask_path": rel_mask,
                    "split": split,
                }
            )
            for c in range(len(class_px)):
                class_px[c] += int(np.count_nonzero(m_r == c))

    frame = pd.DataFrame(rows)
    index_path = _write_seg_index(frame, out)
    summary["n_slices_cached"] = int(len(frame))
    summary["skipped_volumes"] = int(skipped_volumes)
    summary["seg_index"] = str(index_path)
    summary["slice_split"] = (
        {s: int((frame["split"] == s).sum()) for s in ("train", "val")}
        if len(frame)
        else {"train": 0, "val": 0}
    )
    total_fg = int(class_px[1:].sum())
    summary["foreground_class_fraction"] = {
        name: (float(class_px[idx] / total_fg) if total_fg else 0.0)
        for name, idx in ANATOMY_CLASS_TO_INDEX.items()
        if name in FOREGROUND_ANATOMY_CLASSES
    }
    return summary


def _write_seg_index(frame: pd.DataFrame, out: Path) -> Path:
    """Write the SPIDER seg index as parquet (if pyarrow available) else CSV."""
    try:
        import pyarrow  # noqa: F401

        path = out / "seg_index.parquet"
        frame.to_parquet(path, index=False)
    except ImportError:
        path = out / "seg_index.csv"
        frame.to_csv(path, index=False)
    return path
