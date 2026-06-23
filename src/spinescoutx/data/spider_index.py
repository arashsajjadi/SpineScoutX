"""SPIDER dataset indexing, volume loading, and label remapping.

The SPIDER lumbar-spine MRI dataset provides anatomical segmentation masks
(vertebrae, intervertebral discs, spinal canal). This module discovers
image/mask pairs, loads volumes via optional medical-imaging readers, and
collapses SPIDER's raw label ids into the 4-class anatomy scheme defined in
:mod:`spinescoutx.constants`.

Research-only: SPIDER masks are anatomy, not pathology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    ANATOMY_CLASS_TO_INDEX,
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
