"""RSNA dataset path resolution, availability checks, and the series index.

All filesystem checks are tolerant: :func:`check_rsna_available` never raises so
the CLI ``doctor`` command can report a clean status even on a partial download.
``classify_sequence`` and :func:`build_series_index` are pure-pandas helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TRAIN_CSV_NAME = "train.csv"
TRAIN_COORDS_CSV_NAME = "train_label_coordinates.csv"
SERIES_DESC_CSV_NAME = "train_series_descriptions.csv"
TRAIN_IMAGES_DIR_NAME = "train_images"


@dataclass
class RsnaPaths:
    """Resolved expected file/directory locations under an RSNA root."""

    rsna_root: str
    train_csv: str
    train_coords_csv: str
    series_desc_csv: str
    train_images_dir: str

    @classmethod
    def from_root(cls, rsna_root: str | Path) -> RsnaPaths:
        """Build expected RSNA paths relative to ``rsna_root``."""
        root = Path(rsna_root)
        return cls(
            rsna_root=str(root),
            train_csv=str(root / TRAIN_CSV_NAME),
            train_coords_csv=str(root / TRAIN_COORDS_CSV_NAME),
            series_desc_csv=str(root / SERIES_DESC_CSV_NAME),
            train_images_dir=str(root / TRAIN_IMAGES_DIR_NAME),
        )


@dataclass
class AvailabilityReport:
    """Structured result of a dataset availability check."""

    root: str
    exists: bool
    present: dict[str, bool] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a plain-dict view (JSON-serializable)."""
        return {
            "root": self.root,
            "exists": self.exists,
            "present": dict(self.present),
            "missing": list(self.missing),
            "notes": list(self.notes),
        }


def check_rsna_available(rsna_root: str | Path) -> AvailabilityReport:
    """Pure filesystem availability check for an RSNA download. Never raises."""
    root = Path(rsna_root)
    paths = RsnaPaths.from_root(root)
    expected = {
        "train_csv": Path(paths.train_csv),
        "train_coords_csv": Path(paths.train_coords_csv),
        "series_desc_csv": Path(paths.series_desc_csv),
        "train_images_dir": Path(paths.train_images_dir),
    }
    present: dict[str, bool] = {}
    missing: list[str] = []
    for key, p in expected.items():
        ok = p.exists()
        present[key] = ok
        if not ok:
            missing.append(key)

    notes: list[str] = []
    root_exists = root.exists()
    if not root_exists:
        notes.append(f"RSNA root does not exist: {root}")
    if missing:
        notes.append(f"Missing RSNA artifacts: {', '.join(missing)}")
    return AvailabilityReport(
        root=str(root),
        exists=root_exists,
        present=present,
        missing=missing,
        notes=notes,
    )


def classify_sequence(description: str) -> str:
    """Map a series description string to a coarse sequence type.

    Case-insensitive. Returns one of
    ``{"sagittal_t1", "sagittal_t2", "axial_t2", "unknown"}``.
    """
    if not description:
        return "unknown"
    text = str(description).strip().lower()
    is_sagittal = "sag" in text
    is_axial = "ax" in text
    is_t1 = "t1" in text
    is_t2 = "t2" in text or "stir" in text

    if is_sagittal and is_t1:
        return "sagittal_t1"
    if is_sagittal and is_t2:
        return "sagittal_t2"
    if is_axial and is_t2:
        return "axial_t2"
    return "unknown"


def build_series_index(rsna_root: str | Path) -> pd.DataFrame:
    """Read the series-description CSV into a typed index DataFrame.

    Returns columns ``[study_id, series_id, series_description, sequence_type]``.
    """
    paths = RsnaPaths.from_root(rsna_root)
    csv_path = Path(paths.series_desc_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"RSNA series-description CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = {c.strip().lower(): c for c in df.columns}
    study_col = cols.get("study_id")
    series_col = cols.get("series_id")
    desc_col = cols.get("series_description")
    if study_col is None or series_col is None or desc_col is None:
        raise ValueError(
            "Series-description CSV must contain study_id, series_id, "
            f"series_description columns; got {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "study_id": df[study_col].astype(str),
            "series_id": df[series_col].astype(str),
            "series_description": df[desc_col].astype(str),
        }
    )
    out["sequence_type"] = out["series_description"].map(classify_sequence)
    return out
