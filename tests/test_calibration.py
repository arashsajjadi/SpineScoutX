"""Tests for calibration metrics, temperature scaling, and uncertainty flags."""

from __future__ import annotations

import numpy as np

from spinescoutx.evaluation.calibration import (
    TemperatureScaler,
    calibration_report,
    confidence_to_uncertainty_flag,
    expected_calibration_error,
    reliability_curve,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    clipped = np.clip(probs[np.arange(labels.size), labels], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def test_ece_in_unit_range(overconfident_logits) -> None:
    logits, labels = overconfident_logits
    probs = _softmax(logits)
    ece = expected_calibration_error(labels, probs, n_bins=15)
    assert 0.0 <= ece <= 1.0


def test_reliability_curve_shape() -> None:
    labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    probs = np.array(
        [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [0.3, 0.5, 0.2]]
    )
    curve = reliability_curve(labels, probs, n_bins=10)
    assert len(curve["bin_confidence"]) == 10
    assert len(curve["bin_accuracy"]) == 10
    assert len(curve["bin_count"]) == 10
    assert len(curve["bin_edges"]) == 11


def test_temperature_scaling_reduces_or_keeps_nll(overconfident_logits) -> None:
    logits, labels = overconfident_logits
    before = _nll(_softmax(logits), labels)

    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.temperature > 0.0
    after = _nll(scaler.transform(logits), labels)

    # Temperature scaling may not strictly decrease NLL but must not blow it up.
    assert after <= before + 1e-6


def test_calibration_report_with_logits(overconfident_logits) -> None:
    logits, labels = overconfident_logits
    probs = _softmax(logits)
    report = calibration_report(labels, probs, logits=logits, n_bins=10)
    assert 0.0 <= report["ece"] <= 1.0
    assert "temperature" in report
    assert "ece_after" in report
    assert report["temperature"] > 0.0


def test_uncertainty_flag_thresholds() -> None:
    assert confidence_to_uncertainty_flag(0.95) == "high_confidence"
    assert confidence_to_uncertainty_flag(0.85) == "high_confidence"
    assert confidence_to_uncertainty_flag(0.70) == "moderate_confidence"
    assert confidence_to_uncertainty_flag(0.60) == "moderate_confidence"
    assert confidence_to_uncertainty_flag(0.50) == "review_required"
