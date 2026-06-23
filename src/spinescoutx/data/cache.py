"""Content-addressed caching helpers for arrays, configs, and manifests.

Hashes are deterministic: :func:`stable_hash` is the sha256 of canonical
JSON (sorted keys). These hashes key on-disk caches so that stale artefacts are
detected when the upstream config or manifest changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stable_hash(obj: object) -> str:
    """Return the sha256 hex digest of canonical JSON (sorted keys) of ``obj``.

    The object must be JSON-serialisable. Keys are sorted and separators are
    fixed so the digest is stable across runs and Python processes.
    """
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(cfg_dict: dict) -> str:
    """Return a stable hash of a config dict."""
    return stable_hash(cfg_dict)


def manifest_hash(df: pd.DataFrame) -> str:
    """Return a stable hash of a manifest DataFrame.

    The frame is reduced to a sorted list of per-row tuples (with sorted column
    order) so the digest is invariant to row/column ordering.
    """
    columns = sorted(df.columns.tolist())
    ordered = df[columns]
    rows = [tuple(str(v) for v in record) for record in ordered.itertuples(index=False)]
    rows.sort()
    return stable_hash({"columns": columns, "rows": rows})


def save_array(path: str | Path, arr: np.ndarray) -> Path:
    """Save a numpy array to ``path`` (``.npy``), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(arr))
    if p.suffix != ".npy":
        return p.with_suffix(".npy")
    return p


def load_array(path: str | Path) -> np.ndarray:
    """Load a numpy array previously saved with :func:`save_array`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cached array not found: {p}")
    return np.load(p)


def is_cached(path: str | Path) -> bool:
    """Return True if ``path`` exists as a file."""
    return Path(path).is_file()


def write_cache_meta(meta_path: str | Path, hash_value: str) -> Path:
    """Write a tiny JSON sidecar storing ``hash_value`` under ``"hash"``."""
    p = Path(meta_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump({"hash": hash_value}, fh, sort_keys=True)
    return p


def cache_is_stale(meta_path: str | Path, current_hash: str) -> bool:
    """Return True if the cache sidecar is missing, unreadable, or differs.

    A stale cache must be rebuilt. Missing sidecar, malformed JSON, or a stored
    hash that differs from ``current_hash`` all count as stale.
    """
    p = Path(meta_path)
    if not p.exists():
        return True
    try:
        with p.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return True
    return meta.get("hash") != current_hash
