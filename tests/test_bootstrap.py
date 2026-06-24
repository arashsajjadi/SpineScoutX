"""Tests for cluster-bootstrap CIs and paired tests (statistical discipline)."""

from __future__ import annotations

import numpy as np

from spinescoutx.evaluation import bootstrap as bs


def _toy():
    # 6 studies x 2 nodes; severe = class 2. Probs favour the true class.
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    groups = np.array([f"s{i // 2}" for i in range(12)])
    # near-perfect probs
    p = np.full((12, 3), 0.05)
    p[np.arange(12), y] = 0.90
    p = p / p.sum(axis=1, keepdims=True)
    return y, p, groups


def test_bootstrap_ci_is_deterministic_and_brackets_point():
    y, p, g = _toy()
    a = bs.bootstrap_ci(y, p, g, bs.m_severe_recall, n_boot=500, seed=1337)
    b = bs.bootstrap_ci(y, p, g, bs.m_severe_recall, n_boot=500, seed=1337)
    assert a == b  # fully deterministic under the seed
    assert a["ci_lo"] <= a["point"] <= a["ci_hi"]
    assert a["n"] == 12
    assert a["n_severe"] == 4


def test_severe_recall_perfect_predictions():
    y, p, g = _toy()
    assert bs.m_severe_recall(y, p) == 1.0  # argmax matches every label


def test_paired_delta_identical_inputs_is_zero_and_not_decisive():
    y, p, g = _toy()
    d = bs.paired_bootstrap_delta(y, p, p, g, bs.m_weighted_logloss, n_boot=300, seed=7)
    assert abs(d["delta"]) < 1e-9
    assert d["ci_lo"] <= 0.0 <= d["ci_hi"]
    assert d["decisive"] is False


def test_paired_delta_detects_clear_difference():
    y, p, g = _toy()
    bad = np.full_like(p, 1.0 / 3.0)  # uninformative -> worse log loss
    d = bs.paired_bootstrap_delta(y, bad, p, g, bs.m_weighted_logloss, n_boot=400, seed=3)
    assert d["delta"] > 0  # bad has higher (worse) log loss
    assert d["decisive"] is True  # CI excludes 0


def test_mcnemar_severe_counts_discordant_pairs():
    y = np.array([2, 2, 2, 2, 0])
    pred_a = np.array([2, 2, 0, 2, 0])  # catches 3 of 4 severe
    pred_b = np.array([2, 0, 0, 0, 0])  # catches 1 of 4 severe
    out = bs.mcnemar_severe(y, pred_a, pred_b)
    # A catches & B misses on nodes 1,3 (idx) -> b=2 ; B catches & A misses -> 0
    assert out["b_a_catches_b_misses"] == 2
    assert out["c_a_misses_b_catches"] == 0
    assert out["n_discordant"] == 2


def test_ci_table_returns_all_standard_metrics():
    y, p, g = _toy()
    tab = bs.ci_table(y, p, g, n_boot=200, seed=1337)
    for k in ("weighted_logloss", "severe_recall", "severe_auroc", "recall_at_far10", "ece"):
        assert k in tab
        assert "ci_lo" in tab[k] and "ci_hi" in tab[k]
