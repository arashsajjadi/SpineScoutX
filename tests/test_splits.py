"""Tests for deterministic patient-level splits and leakage checks."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from spinescoutx.data.splits import (
    assign_splits,
    check_no_leakage,
    patient_level_split,
    save_split,
    split_stats,
)


def _ids(n: int) -> list[str]:
    return [f"p{i:03d}" for i in range(n)]


def test_patient_level_split_deterministic() -> None:
    ids = _ids(50)
    a = patient_level_split(ids, val_fraction=0.2, seed=123)
    b = patient_level_split(ids, val_fraction=0.2, seed=123)
    assert a == b


def test_patient_level_split_seed_changes_assignment() -> None:
    ids = _ids(50)
    a = patient_level_split(ids, val_fraction=0.2, seed=1)
    b = patient_level_split(ids, val_fraction=0.2, seed=2)
    assert a != b


def test_each_id_exactly_one_split() -> None:
    ids = _ids(40)
    mapping = patient_level_split(ids, val_fraction=0.25, seed=7, test_fraction=0.25)
    assert set(mapping) == set(ids)
    assert all(v in {"train", "val", "test"} for v in mapping.values())


def test_val_fraction_roughly_honored() -> None:
    ids = _ids(100)
    mapping = patient_level_split(ids, val_fraction=0.2, seed=11)
    n_val = sum(1 for v in mapping.values() if v == "val")
    assert n_val == 20


def test_check_no_leakage_passes_on_clean_frame() -> None:
    df = pd.DataFrame(
        {"patient_id": ["a", "a", "b", "b"], "split": ["train", "train", "val", "val"]}
    )
    check_no_leakage(df)  # must not raise


def test_check_no_leakage_raises_on_leak() -> None:
    df = pd.DataFrame({"patient_id": ["a", "a", "b"], "split": ["train", "val", "val"]})
    with pytest.raises(ValueError, match="leakage"):
        check_no_leakage(df)


def test_assign_splits_adds_column() -> None:
    df = pd.DataFrame({"patient_id": ["a", "b", "c"]})
    mapping = {"a": "train", "b": "val", "c": "train"}
    out = assign_splits(df, mapping)
    assert list(out["split"]) == ["train", "val", "train"]


def test_split_stats_counts() -> None:
    df = pd.DataFrame({"split": ["train", "train", "val"]})
    stats = split_stats(df)
    assert stats["train"] == 2
    assert stats["val"] == 1


def test_save_split_writes_seed_and_timestamp(tmp_path) -> None:
    mapping = {"a": "train", "b": "val"}
    out = save_split(mapping, tmp_path / "split.json", seed=99, timestamp="2026-06-23T00:00:00Z")
    payload = json.loads(out.read_text())
    assert payload["seed"] == 99
    assert payload["timestamp"] == "2026-06-23T00:00:00Z"
    assert payload["mapping"] == mapping
    assert "counts" in payload
