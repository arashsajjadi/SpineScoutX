"""Tests for severity classification metrics (hand-computed where possible)."""

from __future__ import annotations

import math

import numpy as np

from spinescoutx.constants import SEVERE_INDEX
from spinescoutx.evaluation.calibration import expected_calibration_error
from spinescoutx.evaluation.classification_metrics import (
    balanced_accuracy,
    severe_false_negative_rate,
    severe_recall,
    weighted_log_loss,
)


def test_weighted_log_loss_hand_computed() -> None:
    # y_true = [normal_mild (idx 0, weight 1), moderate (idx 1, weight 2)].
    # probs: true-class probs are 0.8 and 0.5.
    # loss = (1 * -ln(0.8) + 2 * -ln(0.5)) / (1 + 2).
    y_true = np.array([0, 1], dtype=np.int64)
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]], dtype=np.float64)

    expected = (1.0 * -math.log(0.8) + 2.0 * -math.log(0.5)) / (1.0 + 2.0)
    got = weighted_log_loss(y_true, probs)
    assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(got, 0.5364793041447, abs_tol=1e-9)


def test_weighted_log_loss_perfect_predictions_near_zero() -> None:
    y_true = np.array([0, 1, 2, 0, 2], dtype=np.int64)
    probs = np.full((y_true.size, 3), 1e-12, dtype=np.float64)
    probs[np.arange(y_true.size), y_true] = 1.0 - 2e-12
    loss = weighted_log_loss(y_true, probs)
    assert loss >= 0.0
    assert loss < 1e-6


def test_weighted_log_loss_empty() -> None:
    assert weighted_log_loss(np.array([], dtype=np.int64), np.zeros((0, 3))) == 0.0


def test_severe_recall_and_fnr() -> None:
    # 3 severe samples; 2 of them predicted severe -> recall 2/3, fnr 1/3.
    y_true = np.array([SEVERE_INDEX, SEVERE_INDEX, SEVERE_INDEX, 0, 1], dtype=np.int64)
    y_pred = np.array([SEVERE_INDEX, SEVERE_INDEX, 0, 0, 1], dtype=np.int64)
    assert math.isclose(severe_recall(y_true, y_pred), 2.0 / 3.0, abs_tol=1e-12)
    assert math.isclose(severe_false_negative_rate(y_true, y_pred), 1.0 / 3.0, abs_tol=1e-12)


def test_severe_recall_no_severe_present() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0], dtype=np.int64)
    assert severe_recall(y_true, y_pred) == 0.0
    assert severe_false_negative_rate(y_true, y_pred) == 0.0


def test_balanced_accuracy_perfect_and_chance() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    assert math.isclose(balanced_accuracy(y_true, y_true), 1.0, abs_tol=1e-12)


def test_ece_perfectly_calibrated_vs_overconfident() -> None:
    # Perfectly calibrated: confidence == accuracy within each effective bin.
    y_true = np.array([0, 1, 2, 0], dtype=np.int64)
    perfect = np.zeros((4, 3), dtype=np.float64)
    perfect[np.arange(4), y_true] = 1.0
    # Fill the remaining mass uniformly so confidence is exactly 1.0 and correct.
    ece_perfect = expected_calibration_error(y_true, perfect, n_bins=10)
    assert math.isclose(ece_perfect, 0.0, abs_tol=1e-12)

    # Over-confident wrong predictions: confidence ~1 but accuracy 0 -> ECE ~1.
    overconf = np.array([[0.99, 0.005, 0.005]] * 4, dtype=np.float64)
    wrong_true = np.array([1, 1, 2, 2], dtype=np.int64)
    ece_over = expected_calibration_error(wrong_true, overconf, n_bins=10)
    assert ece_over > ece_perfect
    assert ece_over > 0.9
