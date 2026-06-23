"""Tests for the severe-first operating-frontier math (pure functions)."""

from __future__ import annotations

import numpy as np

from spinescoutx.evaluation.severe_frontier import (
    build_frontier,
    recall_at_far_budget,
    severe_auroc_pr,
    sweep_severe_threshold,
    threshold_for_recall,
)

# severity indices: 0 normal/mild, 1 moderate, 2 severe
Y = np.array([0, 0, 0, 1, 1, 2, 2, 2])  # 3 severe, 5 non-severe
P = np.array([0.01, 0.05, 0.30, 0.20, 0.60, 0.40, 0.85, 0.95])  # P(severe)


def test_sweep_monotone_recall() -> None:
    sweep = sweep_severe_threshold(Y, P)
    recalls = [r["severe_recall"] for r in sweep]
    # recall is non-increasing as threshold rises
    assert all(a >= b - 1e-9 for a, b in zip(recalls, recalls[1:], strict=False))
    assert sweep[0]["severe_recall"] == 1.0  # threshold 0 alarms everything
    assert sweep[-1]["false_alarm_rate"] == 0.0  # threshold 1 alarms ~nothing


def test_recall_at_far_budget_respects_budget() -> None:
    sweep = sweep_severe_threshold(Y, P)
    res = recall_at_far_budget(sweep, far_budget=0.0)
    # with zero false alarms allowed, only severe>all-nonsevere count
    assert res["false_alarm_rate"] <= 0.0
    assert 0.0 <= res["severe_recall"] <= 1.0


def test_threshold_for_recall_reaches_target() -> None:
    sweep = sweep_severe_threshold(Y, P)
    res = threshold_for_recall(sweep, target_recall=0.66)
    assert res["reached"] is True
    assert res["severe_recall"] >= 0.66


def test_threshold_for_unreachable_recall() -> None:
    # a model that never scores one severe high cannot reach recall 1.0 cheaply,
    # but recall 1.0 is always reachable at threshold 0 -> use >1 to be unreachable
    sweep = sweep_severe_threshold(Y, P)
    res = threshold_for_recall(sweep, target_recall=1.01)
    assert res["reached"] is False


def test_severe_auroc_pr() -> None:
    m = severe_auroc_pr(Y, P)
    assert 0.5 <= m["severe_auroc"] <= 1.0
    assert 0.0 <= m["severe_ap"] <= 1.0


def test_build_frontier_aligns_shared_nodes() -> None:
    a = {"s1|l1_l2": (2, 0.9), "s1|l2_l3": (0, 0.1), "s2|l1_l2": (1, 0.4)}
    b = {"s1|l1_l2": (2, 0.7), "s1|l2_l3": (0, 0.2)}  # missing s2|l1_l2
    out = build_frontier({"A": a, "B": b})
    assert out["n_shared_nodes"] == 2
    assert set(out["models"]) == {"A", "B"}
    for name in ("A", "B"):
        assert "severe_auroc" in out["models"][name]
        assert "recall_at_far" in out["models"][name]
