"""Deterministic patient-level train/val/test splits and leakage checks.

All randomness flows through :func:`numpy.random.default_rng` seeded by the
caller, so splits are reproducible. The core guarantee is that a given id maps
to exactly one split, and :func:`check_no_leakage` enforces this on a frame.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def patient_level_split(
    ids: Sequence[str],
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.0,
) -> dict[str, str]:
    """Assign each unique id to "train" / "val" / "test".

    Deterministic for a fixed ``seed``. ``val_fraction`` and ``test_fraction``
    are fractions of the unique-id count; the remainder is "train". Each id maps
    to exactly one split.
    """
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Fractions must be non-negative")
    if val_fraction + test_fraction > 1.0:
        raise ValueError("val_fraction + test_fraction must be <= 1.0")

    unique = sorted({str(i) for i in ids})
    n = len(unique)
    result: dict[str, str] = {}
    if n == 0:
        return result

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    shuffled = [unique[i] for i in order]

    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    # Keep at least one training id when any split is requested and possible.
    n_val = min(n_val, n)
    n_test = min(n_test, n - n_val)

    val_ids = shuffled[:n_val]
    test_ids = shuffled[n_val : n_val + n_test]
    train_ids = shuffled[n_val + n_test :]

    for i in train_ids:
        result[i] = "train"
    for i in val_ids:
        result[i] = "val"
    for i in test_ids:
        result[i] = "test"
    return result


def assign_splits(
    df: pd.DataFrame,
    split_map: dict[str, str],
    id_col: str = "patient_id",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a "split" column from ``split_map``."""
    if id_col not in df.columns:
        raise KeyError(f"id column {id_col!r} not in DataFrame")
    out = df.copy()
    out["split"] = out[id_col].astype(str).map(split_map).fillna("")
    return out


def check_no_leakage(
    df: pd.DataFrame,
    id_col: str = "patient_id",
    split_col: str = "split",
) -> None:
    """Raise ``ValueError`` if any id appears in more than one split."""
    for col in (id_col, split_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in DataFrame")
    grouped = df.groupby(df[id_col].astype(str))[split_col].apply(
        lambda s: {str(v) for v in s if str(v) != ""}
    )
    offenders = sorted(idx for idx, splits in grouped.items() if len(splits) > 1)
    if offenders:
        raise ValueError(f"Patient-level leakage: ids in multiple splits: {offenders}")


def split_stats(df: pd.DataFrame, split_col: str = "split") -> dict[str, int]:
    """Return a mapping of split name -> row count."""
    if split_col not in df.columns:
        raise KeyError(f"column {split_col!r} not in DataFrame")
    counts = df[split_col].astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def save_split(
    split_map: dict[str, str],
    path: str | Path,
    seed: int,
    timestamp: str,
    stats: dict | None = None,
) -> Path:
    """Persist a split mapping as JSON.

    ``timestamp`` is supplied by the caller (never read from a clock here) so the
    function stays pure/reproducible. Counts are derived from the mapping when
    ``stats`` is not provided.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if stats is None:
        counts: dict[str, int] = {}
        for split in split_map.values():
            counts[split] = counts.get(split, 0) + 1
    else:
        counts = {str(k): int(v) for k, v in stats.items()}
    payload = {
        "seed": int(seed),
        "timestamp": str(timestamp),
        "counts": counts,
        "mapping": dict(split_map),
    }
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
    return p
