"""Locked patient-level train / dev / test protocol (splits_v1).

The historical work used a single train/val split (seed 1337). For credible v1-track
claims we add an explicit **three-way patient-level** split with a *locked test* that is
used ONLY for final evaluation (never for model selection or tuning). Historical val
results are preserved separately and never presented as final v1 claims.

`build_splits_v1` is a thin, deterministic wrapper over the audited
:func:`~spinescoutx.data.splits.patient_level_split` (which guarantees each id maps to
exactly one split); it only renames ``val``→``dev`` so the vocabulary is unambiguous.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .splits import patient_level_split

SPLITS = ("train", "dev", "test")


def build_splits_v1(
    study_ids: Sequence[str],
    *,
    seed: int = 20260623,
    dev_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, str]:
    """Deterministic patient-level {study_id -> 'train'|'dev'|'test'}.

    ``dev`` is for model selection / tuning; ``test`` is the locked test (final eval
    only). Disjoint by construction (one id -> one split).
    """
    raw = patient_level_split(study_ids, val_fraction=dev_frac, seed=seed, test_fraction=test_frac)
    return {sid: ("dev" if split == "val" else split) for sid, split in raw.items()}


def save_splits_v1(
    split_map: dict[str, str], path: str | Path, *, seed: int, timestamp: str
) -> Path:
    """Persist the split mapping + counts as JSON (timestamp passed in, never a clock)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(1 for v in split_map.values() if v == s) for s in SPLITS}
    payload = {
        "schema": "splits_v1",
        "seed": int(seed),
        "timestamp": str(timestamp),
        "unit": "study_id (patient-level)",
        "counts": counts,
        "mapping": dict(split_map),
    }
    p.write_text(json.dumps(payload, sort_keys=True, indent=2))
    return p


def load_splits_v1(path: str | Path) -> dict[str, str]:
    """Load a saved splits_v1 mapping (study_id -> split)."""
    payload = json.loads(Path(path).read_text())
    return {str(k): str(v) for k, v in payload["mapping"].items()}


def assert_disjoint(split_map: dict[str, str]) -> None:
    """Raise if any study is assigned to more than one split (defensive; map is 1:1)."""
    sets = {s: {k for k, v in split_map.items() if v == s} for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and sets[a] & sets[b]:
                raise ValueError(
                    f"locked-test leakage: studies in both {a} and {b}: {sets[a] & sets[b]}"
                )


def stratified_counts(manifest: pd.DataFrame, split_map: dict[str, str]) -> dict[str, dict]:
    """Per-split crop / severe counts, broken down by condition. Manifest must have
    ``study_id``, ``condition``, ``severity``."""
    m = manifest.copy()
    m["study_id"] = m["study_id"].astype(str)
    m["_split"] = m["study_id"].map(split_map)
    out: dict[str, dict] = {}
    for split in SPLITS:
        g = m[m["_split"] == split]
        per_cond = {}
        for cond, gc in g.groupby("condition"):
            per_cond[str(cond)] = {
                "n": int(len(gc)),
                "n_severe": int((gc["severity"] == "severe").sum()),
                "n_studies": int(gc["study_id"].nunique()),
            }
        out[split] = {
            "n_crops": int(len(g)),
            "n_studies": int(g["study_id"].nunique()),
            "n_severe": int((g["severity"] == "severe").sum()),
            "by_condition": per_cond,
        }
    return out
