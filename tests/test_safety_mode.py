"""Tests for Safety Mode decision layer + the cost-sensitive training loss."""

from __future__ import annotations

import numpy as np
import torch

from spinescoutx.evaluation import safety_mode as sm
from spinescoutx.training.losses import ExpectedCostLoss, severe_aware_cost_matrix


def _preds(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    # confident-ish correct probs with some noise
    p = rng.random((n, 3)) * 0.2
    p[np.arange(n), y] += 0.7
    p = p / p.sum(1, keepdims=True)
    return y, p


def test_operating_points_have_expected_keys():
    y, p = _preds()
    op = sm.operating_points(y, p, target_recalls=(0.9,))
    assert "balanced_argmax" in op and "safety" in op
    assert 0.0 <= op["balanced_argmax"]["severe_recall"] <= 1.0


def test_abstention_increases_effective_recall_monotonically_at_full_abstain():
    y, p = _preds()
    curve = sm.abstention_curve(y, p, n_steps=11)
    # at tau=0 nobody is abstained; at tau=1 everybody is -> effective recall 1.0
    assert curve[0]["abstain_rate"] == 0.0
    assert curve[-1]["abstain_rate"] == 1.0
    assert curve[-1]["effective_severe_recall_with_review"] == 1.0
    # effective recall with review is non-decreasing in tau
    effs = [r["effective_severe_recall_with_review"] for r in curve]
    assert all(b >= a - 1e-9 for a, b in zip(effs, effs[1:], strict=False))


def test_aux_conf_can_only_lower_confidence():
    y, p = _preds()
    aux = np.zeros(len(y))  # force min-confidence to 0 -> everything abstains above tau>0
    curve = sm.abstention_curve(y, p, aux_conf=aux, n_steps=5)
    assert curve[1]["abstain_rate"] == 1.0  # any tau>0 abstains all


def test_cost_weighted_penalises_severe_miss_more():
    y = np.array([2, 2])
    miss = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])  # predict normal for true severe
    catch = np.array([[0.05, 0.05, 0.9], [0.05, 0.05, 0.9]])
    assert (
        sm.cost_weighted_score(y, miss)["total_cost"]
        > sm.cost_weighted_score(y, catch)["total_cost"]
    )


def test_cost_matrix_severe_fn_is_largest():
    c = severe_aware_cost_matrix()
    assert c[2, 0] == c.max()  # true severe -> predicted normal/mild is the costliest
    assert c[0, 0] == 0.0


def test_expected_cost_loss_lower_when_confident_correct():
    loss = ExpectedCostLoss(severe_aware_cost_matrix())
    targets = torch.tensor([2, 2, 1])
    good = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 5.0, 0.0]])
    bad = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert float(loss(good, targets)) < float(loss(bad, targets))
