"""Re-crop an auto-localized node at a perturbed centre/slice for evidence
stability — genuine re-cropping from the source slices (not translation of a
pre-cut crop), matching the exact preprocessing used to build the auto cache.

Used by ``scripts/run_evidence_stability.py``. No ground-truth coordinates are
read; the centre ``(x, y)`` and ``instance_number`` come from the auto manifest.

Research-only. Not diagnostic.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .crops import extract_25d
from .dicom_io import normalize_intensity, read_dicom


class SliceDecoder:
    """LRU cache of normalised full DICOM slices, keyed by file path.

    Decoding dominates evidence-stability runtime; K perturbations of one node
    share the same small slice window, so caching makes re-cropping cheap.
    """

    def __init__(self, max_items: int = 512) -> None:
        self.max_items = int(max_items)
        self._cache: OrderedDict[str, np.ndarray | None] = OrderedDict()

    def get(self, path: Path) -> np.ndarray | None:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr: np.ndarray | None
        if not path.exists():
            arr = None
        else:
            try:
                arr = normalize_intensity(read_dicom(path)).astype(np.float32)
            except Exception:  # noqa: BLE001 — a corrupt slice -> treat as missing
                arr = None
        self._cache[key] = arr
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return arr


def reextract_25d(
    decoder: SliceDecoder,
    dicom_path: str | Path,
    instance_number: int,
    x: float,
    y: float,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    ds: int = 0,
    crop_size: int = 224,
) -> np.ndarray | None:
    """Return a ``(3, crop_size, crop_size)`` 2.5D crop at the perturbed centre.

    ``dicom_path`` is the centre instance's file; neighbours are siblings named
    ``{inst}.dcm`` in the same directory. Falls back to the un-jittered centre if
    the slice-shifted centre is missing; returns ``None`` if no centre slice exists.
    """
    dicom_path = Path(dicom_path)
    series_dir = dicom_path.parent
    center = int(instance_number) + int(ds)

    def load(i: int) -> np.ndarray | None:
        return decoder.get(series_dir / f"{i}.dcm")

    slices: dict[int, np.ndarray] = {}
    for i in (center - 1, center, center + 1):
        a = load(i)
        if a is not None:
            slices[i] = a
    if center not in slices:  # slice-jitter ran off the stack edge -> fall back
        center = int(instance_number)
        slices = {}
        for i in (center - 1, center, center + 1):
            a = load(i)
            if a is not None:
                slices[i] = a
    if center not in slices:
        return None
    arr, _ = extract_25d(slices, center, float(x) + float(dx), float(y) + float(dy), int(crop_size))
    return np.ascontiguousarray(arr, dtype=np.float32)
