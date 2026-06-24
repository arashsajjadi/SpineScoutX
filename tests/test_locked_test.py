"""Tests for the locked patient-level train/dev/test protocol (splits_v1)."""

from __future__ import annotations

import pandas as pd

from spinescoutx.data.locked_test import (
    SPLITS,
    assert_disjoint,
    build_splits_v1,
    stratified_counts,
)
from spinescoutx.data.splits import check_no_leakage


def test_splits_are_three_way_disjoint_and_deterministic():
    ids = [f"s{i}" for i in range(200)]
    a = build_splits_v1(ids, seed=7, dev_frac=0.15, test_frac=0.15)
    b = build_splits_v1(ids, seed=7, dev_frac=0.15, test_frac=0.15)
    assert a == b  # deterministic
    assert set(a.values()) <= set(SPLITS)
    assert_disjoint(a)  # no id in two splits
    # every id assigned exactly once
    assert len(a) == 200


def test_split_fractions_are_approximately_right():
    ids = [f"s{i}" for i in range(1000)]
    m = build_splits_v1(ids, seed=1, dev_frac=0.15, test_frac=0.15)
    counts = {s: sum(1 for v in m.values() if v == s) for s in SPLITS}
    assert abs(counts["dev"] - 150) <= 5
    assert abs(counts["test"] - 150) <= 5
    assert counts["train"] + counts["dev"] + counts["test"] == 1000


def test_no_patient_leakage_when_assigned_to_a_frame():
    ids = [f"s{i}" for i in range(60)]
    m = build_splits_v1(ids, seed=3)
    df = pd.DataFrame({"patient_id": ids, "split": [m[i] for i in ids]})
    check_no_leakage(df)  # must not raise


def test_stratified_counts_partition_the_manifest():
    rows = []
    for i in range(40):
        rows.append(
            {
                "study_id": f"s{i}",
                "condition": "spinal_canal_stenosis",
                "severity": "severe" if i % 4 == 0 else "normal_mild",
            }
        )
    man = pd.DataFrame(rows)
    m = build_splits_v1([f"s{i}" for i in range(40)], seed=5)
    c = stratified_counts(man, m)
    total = sum(c[s]["n_crops"] for s in SPLITS)
    assert total == 40  # splits partition every crop
    total_sev = sum(c[s]["n_severe"] for s in SPLITS)
    assert total_sev == 10
