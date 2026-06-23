"""Pure parsing of RSNA labels into a long-form, canonical-vocabulary frame.

The RSNA ``train.csv`` is *wide*: each row is a study and most columns encode a
``<condition>_<level>`` finding (e.g. ``spinal_canal_stenosis_l1_l2``) whose value
is a severity grade. We re-shape this into a long form with canonical condition,
level, side, and severity values drawn from :mod:`spinescoutx.constants`.

The coordinates CSV carries human-readable condition strings (e.g. ``"Left Neural
Foraminal Narrowing"``) which we canonicalize and split into base + side.

These functions are intentionally dependency-light (pandas only) so they can be
unit-tested on small synthetic DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import (
    CONDITIONS,
    LEVELS,
    SEVERITY_TO_INDEX,
    split_condition,
)
from .rsna_index import RsnaPaths

# Conditions sorted longest-first so that, when matching a wide column name as a
# prefix, the most specific (e.g. left/right) key wins before the base key.
_CONDITIONS_BY_LEN: tuple[str, ...] = tuple(sorted(CONDITIONS, key=len, reverse=True))


def _normalize_token(raw: str) -> str:
    """Lower-case and collapse whitespace / separators to single underscores."""
    text = str(raw).strip().lower()
    for ch in (" ", "-", "/", "\\"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with tolerant, normalized column names.

    Shared helper used by both label and coordinate loaders so that minor
    differences in CSV header casing / spacing do not break parsing.
    """
    out = df.copy()
    out.columns = [_normalize_token(c) for c in out.columns]
    return out


def canonical_severity(raw: str) -> str:
    """Map a raw severity string to a canonical severity key.

    ``"Normal/Mild" -> "normal_mild"``, ``"Moderate" -> "moderate"``,
    ``"Severe" -> "severe"``. Tolerant of case, spacing and ``/`` separators.
    Raises :class:`ValueError` on an unknown value.
    """
    token = _normalize_token(raw)
    mapping = {
        "normal_mild": "normal_mild",
        "moderate": "moderate",
        "severe": "severe",
    }
    if token not in mapping:
        raise ValueError(f"Unknown severity grade: {raw!r}")
    return mapping[token]


def canonical_condition(raw: str) -> str:
    """Map a raw condition string to a canonical condition key.

    E.g. ``"Spinal Canal Stenosis" -> "spinal_canal_stenosis"`` and
    ``"Left Neural Foraminal Narrowing" -> "left_neural_foraminal_narrowing"``.
    Raises :class:`ValueError` if the result is not a known condition.
    """
    token = _normalize_token(raw)
    if token not in CONDITIONS:
        raise ValueError(f"Unknown condition: {raw!r}")
    return token


def canonical_level(raw: str) -> str:
    """Map a raw level string to a canonical level key.

    E.g. ``"L1/L2" -> "l1_l2"``, ``"L5/S1" -> "l5_s1"``. Raises
    :class:`ValueError` on an unknown level.
    """
    token = _normalize_token(raw)
    if token not in LEVELS:
        raise ValueError(f"Unknown level: {raw!r}")
    return token


def _split_condition_level_column(column: str) -> tuple[str, str] | None:
    """Split a wide-format column into (condition, level), or ``None``.

    Matches a known condition as the *prefix* and a known level as the *suffix*.
    """
    token = _normalize_token(column)
    for condition in _CONDITIONS_BY_LEN:
        prefix = f"{condition}_"
        if not token.startswith(prefix):
            continue
        suffix = token[len(prefix) :]
        if suffix in LEVELS:
            return condition, suffix
    return None


def parse_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape a wide RSNA label frame into long, canonical form.

    Returns columns ``[study_id, condition, level, side, severity,
    severity_index]``. Cells whose severity is missing/NaN are skipped. Cells
    whose severity is non-null but unrecognized raise :class:`ValueError`.
    """
    norm = normalize_columns(df)
    if "study_id" not in norm.columns:
        raise ValueError(f"Label frame must contain a study_id column; got {list(df.columns)}")

    finding_columns: list[tuple[str, str, str]] = []
    for col in norm.columns:
        if col == "study_id":
            continue
        parsed = _split_condition_level_column(col)
        if parsed is not None:
            condition, level = parsed
            finding_columns.append((col, condition, level))

    rows: list[dict[str, object]] = []
    for _, record in norm.iterrows():
        study_id = str(record["study_id"])
        for col, condition, level in finding_columns:
            value = record[col]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if pd.isna(value):
                continue
            severity = canonical_severity(str(value))
            _, side = split_condition(condition)
            rows.append(
                {
                    "study_id": study_id,
                    "condition": condition,
                    "level": level,
                    "side": side,
                    "severity": severity,
                    "severity_index": SEVERITY_TO_INDEX[severity],
                }
            )

    columns = ["study_id", "condition", "level", "side", "severity", "severity_index"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def load_labels(rsna_root: str | Path) -> pd.DataFrame:
    """Read ``train.csv`` and return the long-form canonical label frame."""
    paths = RsnaPaths.from_root(rsna_root)
    csv_path = Path(paths.train_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"RSNA train.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return parse_label_columns(df)


def load_coordinates(rsna_root: str | Path) -> pd.DataFrame:
    """Read ``train_label_coordinates.csv`` into canonical long form.

    Returns columns ``[study_id, series_id, instance_number, condition, level,
    side, x, y]``. Condition strings are canonicalized and the side is derived
    via :func:`spinescoutx.constants.split_condition`.
    """
    paths = RsnaPaths.from_root(rsna_root)
    csv_path = Path(paths.train_coords_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"RSNA coordinates CSV not found: {csv_path}")

    norm = normalize_columns(pd.read_csv(csv_path))
    required = ["study_id", "series_id", "instance_number", "condition", "level", "x", "y"]
    missing = [c for c in required if c not in norm.columns]
    if missing:
        raise ValueError(
            f"Coordinates CSV missing required columns {missing}; got {list(norm.columns)}"
        )

    conditions = norm["condition"].map(canonical_condition)
    sides = conditions.map(lambda c: split_condition(c)[1])
    out = pd.DataFrame(
        {
            "study_id": norm["study_id"].astype(str),
            "series_id": norm["series_id"].astype(str),
            "instance_number": norm["instance_number"].astype(int),
            "condition": conditions,
            "level": norm["level"].map(canonical_level),
            "side": sides,
            "x": norm["x"].astype(float),
            "y": norm["y"].astype(float),
        }
    )
    return out
