"""Localizer-centred crop extraction and the canonical ``CropRecord``.

This module is NUMPY-ONLY (no torch). It defines the :class:`CropRecord`
dataclass that the rest of the data pipeline imports, plus pure crop-geometry
helpers (testable on synthetic arrays) and manifest read/write utilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import SEVERITY_TO_INDEX
from ..utils.logging import get_logger

log = get_logger()


@dataclass
class CropRecord:
    """One localizer-centred crop and its labels/provenance.

    ``severity`` is "" (and ``severity_index`` is -1) when the crop is
    unlabelled. ``side`` is None for non-sided conditions. Paths are strings so
    the record round-trips cleanly through pandas/parquet/CSV.
    """

    study_id: str
    series_id: str
    instance_number: int
    condition: str
    level: str
    side: str | None
    severity: str
    severity_index: int
    x: float
    y: float
    crop_path: str
    dicom_path: str
    split: str
    sequence: str
    patient_id: str
    pad_note: str = ""
    # Provenance of the crop centre: "oracle" (GT localizer coordinate; research
    # upper bound) or "auto" (predicted by the disc-level localizer; real inference).
    coordinate_source: str = "oracle"


def crop_bounds(x: float, y: float, size: int, h: int, w: int) -> tuple[int, int, int, int, bool]:
    """Return ``(y0, y1, x0, x1, needs_pad)`` for a ``size``x``size`` box.

    The box is centred at ``(x, y)`` (x=column, y=row) and clamped to the image
    of height ``h`` and width ``w``. ``needs_pad`` is True when the ideal box
    extends beyond the image bounds (so callers must zero-pad).
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    half = size // 2
    cx = int(round(x))
    cy = int(round(y))

    ideal_x0 = cx - half
    ideal_y0 = cy - half
    ideal_x1 = ideal_x0 + size
    ideal_y1 = ideal_y0 + size

    x0 = max(0, ideal_x0)
    y0 = max(0, ideal_y0)
    x1 = min(w, ideal_x1)
    y1 = min(h, ideal_y1)

    needs_pad = ideal_x0 < 0 or ideal_y0 < 0 or ideal_x1 > w or ideal_y1 > h
    return y0, y1, x0, x1, needs_pad


def extract_crop(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """Return a ``size``x``size`` float32 crop centred at ``(x, y)``.

    Out-of-bounds regions are zero-padded so the output is always exactly
    ``size``x``size``. Operates on a single 2D image.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"extract_crop expects a 2D image, got shape {img.shape}")
    h, w = img.shape
    half = size // 2
    cx = int(round(x))
    cy = int(round(y))
    ideal_x0 = cx - half
    ideal_y0 = cy - half

    y0, y1, x0, x1, _ = crop_bounds(x, y, size, h, w)
    out = np.zeros((size, size), dtype=np.float32)
    if x1 > x0 and y1 > y0:
        dst_y0 = y0 - ideal_y0
        dst_x0 = x0 - ideal_x0
        out[dst_y0 : dst_y0 + (y1 - y0), dst_x0 : dst_x0 + (x1 - x0)] = img[y0:y1, x0:x1]
    return out


def extract_25d(
    slices: dict[int, np.ndarray],
    center: int,
    x: float,
    y: float,
    size: int,
) -> tuple[np.ndarray, str]:
    """Stack (prev, center, next) crops into a ``(3, size, size)`` array.

    For a missing neighbour slice index, the nearest available slice is
    duplicated (pad note "dup_nearest_slice"); if no valid slice exists at all
    the channel is zero-padded (pad note "zero_pad"). Returns the stacked
    float32 array and a pad note summarising the 2.5D decisions.
    """
    if center not in slices:
        raise KeyError(f"center slice index {center} missing from slices")

    available = sorted(slices.keys())

    def nearest_index(target: int) -> int | None:
        if not available:
            return None
        return min(available, key=lambda k: (abs(k - target), k))

    notes: list[str] = []
    channels: list[np.ndarray] = []
    for offset in (-1, 0, 1):
        target = center + offset
        if target in slices:
            channels.append(extract_crop(slices[target], x, y, size))
            continue
        if offset == 0:  # center is guaranteed present above; defensive only
            channels.append(extract_crop(slices[center], x, y, size))
            continue
        near = nearest_index(target)
        if near is None:
            channels.append(np.zeros((size, size), dtype=np.float32))
            notes.append("zero_pad")
        else:
            channels.append(extract_crop(slices[near], x, y, size))
            notes.append("dup_nearest_slice")

    stacked = np.stack(channels, axis=0).astype(np.float32)
    pad_note = ";".join(dict.fromkeys(notes))  # de-dup, preserve order
    return stacked, pad_note


def _record_to_row(rec: CropRecord) -> dict[str, object]:
    row = asdict(rec)
    # Normalise None side to "" so columns stay string-typed in pandas/parquet.
    row["side"] = "" if rec.side is None else rec.side
    return row


def records_to_frame(records: list[CropRecord]) -> pd.DataFrame:
    """Convert a list of :class:`CropRecord` into a DataFrame (stable columns)."""
    columns = [f.name for f in fields(CropRecord)]
    rows = [_record_to_row(r) for r in records]
    return pd.DataFrame(rows, columns=columns)


def _coerce_str(value: object, default: str) -> str:
    """Return a clean string, falling back to ``default`` for NaN/None/empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value)
    return default if text == "" or text.lower() in ("nan", "none") else text


def _coerce_side(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text == "" or text.lower() == "nan" or text.lower() == "none":
        return None
    return text


def frame_to_records(df: pd.DataFrame) -> list[CropRecord]:
    """Convert a manifest DataFrame back into :class:`CropRecord` objects."""
    records: list[CropRecord] = []
    for _, row in df.iterrows():
        severity = "" if pd.isna(row.get("severity")) else str(row.get("severity"))
        sev_idx = row.get("severity_index")
        severity_index = -1 if pd.isna(sev_idx) else int(sev_idx)
        if severity and severity in SEVERITY_TO_INDEX:
            severity_index = SEVERITY_TO_INDEX[severity]
        records.append(
            CropRecord(
                study_id=str(row.get("study_id", "")),
                series_id=str(row.get("series_id", "")),
                instance_number=int(row.get("instance_number", 0)),
                condition=str(row.get("condition", "")),
                level=str(row.get("level", "")),
                side=_coerce_side(row.get("side")),
                severity=severity,
                severity_index=severity_index,
                x=float(row.get("x", 0.0)),
                y=float(row.get("y", 0.0)),
                crop_path=str(row.get("crop_path", "")),
                dicom_path=str(row.get("dicom_path", "")),
                split=str(row.get("split", "")) if not pd.isna(row.get("split")) else "",
                sequence=str(row.get("sequence", "")),
                patient_id=str(row.get("patient_id", "")),
                pad_note=str(row.get("pad_note", "")) if not pd.isna(row.get("pad_note")) else "",
                coordinate_source=_coerce_str(row.get("coordinate_source", "oracle"), "oracle"),
            )
        )
    return records


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401  (availability probe only)
    except ImportError:
        return False
    return True


def write_manifest(records: list[CropRecord], path: str | Path) -> Path:
    """Write a crop manifest, choosing parquet or CSV by extension/availability.

    Parquet is used when the path extension is ``.parquet`` and ``pyarrow`` is
    importable; otherwise the data is written as CSV (logging a warning and
    switching the extension to ``.csv`` when a parquet target cannot be honoured).
    """
    out = Path(path)
    frame = records_to_frame(records)
    if out.suffix == ".parquet":
        if _pyarrow_available():
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(out, index=False)
            return out
        out = out.with_suffix(".csv")
        log.warning("pyarrow unavailable; writing manifest as CSV to %s", out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a crop manifest written by :func:`write_manifest` (parquet or CSV)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)
