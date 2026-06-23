"""DICOM reading and intensity-normalization helpers.

``pydicom`` is an optional dependency: it is imported lazily inside the functions
that need it so the rest of the package works without it. ``normalize_intensity``
and ``to_uint8`` are pure-numpy and have no DICOM dependency, so they are
unit-testable on synthetic arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_DICOM_EXTRA_HINT = "pip install spinescoutx[dicom]  (provides pydicom)"


class DicomDecodeError(Exception):
    """Raised when a DICOM file cannot be read or has no pixel data.

    Attributes
    ----------
    study_id, series_id, path:
        Best-effort identifiers for the offending file (may be ``None``).
    category:
        One of ``{"missing_file", "decode_error", "no_pixel_data"}``.
    message:
        Human-readable explanation.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str,
        path: str | None = None,
        study_id: str | None = None,
        series_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.path = path
        self.study_id = study_id
        self.series_id = series_id


def _import_pydicom() -> Any:
    """Lazy-import pydicom with a clear, actionable error message."""
    try:
        import pydicom  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without pydicom
        raise ImportError(
            f"pydicom is required to read DICOM files. Install it with: {_DICOM_EXTRA_HINT}"
        ) from exc
    return pydicom


def read_dicom(path: str | Path) -> np.ndarray:
    """Read a single DICOM slice into a 2D ``float32`` array.

    Applies ``RescaleSlope`` / ``RescaleIntercept`` when present. Raises
    :class:`DicomDecodeError` (never a bare pydicom error) on failure.
    """
    pydicom = _import_pydicom()
    p = Path(path)
    if not p.exists():
        raise DicomDecodeError(f"DICOM file not found: {p}", category="missing_file", path=str(p))
    try:
        ds = pydicom.dcmread(str(p))
    except Exception as exc:
        raise DicomDecodeError(
            f"Failed to decode DICOM {p}: {exc}", category="decode_error", path=str(p)
        ) from exc

    study_id = _maybe_str(getattr(ds, "StudyInstanceUID", None))
    series_id = _maybe_str(getattr(ds, "SeriesInstanceUID", None))

    if not hasattr(ds, "PixelData"):
        raise DicomDecodeError(
            f"DICOM {p} has no pixel data",
            category="no_pixel_data",
            path=str(p),
            study_id=study_id,
            series_id=series_id,
        )
    try:
        pixels = ds.pixel_array
    except Exception as exc:
        raise DicomDecodeError(
            f"Failed to access pixel array for {p}: {exc}",
            category="no_pixel_data",
            path=str(p),
            study_id=study_id,
            series_id=series_id,
        ) from exc

    arr = np.asarray(pixels, dtype=np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    return np.ascontiguousarray(arr, dtype=np.float32)


def dicom_metadata(path: str | Path) -> dict[str, Any]:
    """Return a metadata dict for a DICOM file; missing tags map to ``None``."""
    pydicom = _import_pydicom()
    p = Path(path)
    if not p.exists():
        raise DicomDecodeError(f"DICOM file not found: {p}", category="missing_file", path=str(p))
    try:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
    except Exception as exc:
        raise DicomDecodeError(
            f"Failed to decode DICOM metadata {p}: {exc}",
            category="decode_error",
            path=str(p),
        ) from exc

    pixel_spacing = getattr(ds, "PixelSpacing", None)
    if pixel_spacing is not None:
        pixel_spacing = [float(v) for v in pixel_spacing]

    return {
        "study_id": _maybe_str(getattr(ds, "StudyInstanceUID", None)),
        "series_id": _maybe_str(getattr(ds, "SeriesInstanceUID", None)),
        "instance_number": _maybe_int(getattr(ds, "InstanceNumber", None)),
        "rows": _maybe_int(getattr(ds, "Rows", None)),
        "cols": _maybe_int(getattr(ds, "Columns", None)),
        "series_description": _maybe_str(getattr(ds, "SeriesDescription", None)),
        "modality": _maybe_str(getattr(ds, "Modality", None)),
        "pixel_spacing": pixel_spacing,
    }


def normalize_intensity(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Robust percentile clip then scale to ``[0, 1]`` ``float32``.

    Pure numpy (no pydicom). Clips ``arr`` to its ``low``/``high`` percentiles and
    linearly rescales the result to ``[0, 1]``. A constant image (or one whose
    low/high percentiles coincide) maps to all zeros.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return a.astype(np.float32, copy=True)
    lo = float(np.percentile(a, low))
    hi = float(np.percentile(a, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    clipped = np.clip(a, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    return scaled.astype(np.float32)


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    """Scale a ``[0, 1]`` array to ``uint8`` ``[0, 255]`` (values are clipped)."""
    a = np.asarray(arr01, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    return np.round(a * 255.0).astype(np.uint8)


def _maybe_str(value: Any) -> str | None:
    """Return ``str(value)`` or ``None`` if value is ``None``."""
    return None if value is None else str(value)


def _maybe_int(value: Any) -> int | None:
    """Best-effort int conversion; ``None`` on missing or non-numeric values."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
