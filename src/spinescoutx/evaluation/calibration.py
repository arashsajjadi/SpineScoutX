"""Confidence-calibration metrics and temperature scaling for SpineScoutX.

Provides top-label Expected Calibration Error (ECE), a reliability curve, a
scalar temperature scaler fit by LBFGS on the negative log-likelihood, and a
helper that maps a confidence value to a research uncertainty flag.

Research-only: confidence and uncertainty here describe model behaviour, not
clinical certainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import UNCERTAINTY_FLAGS


def _as_probs(probs: np.ndarray) -> np.ndarray:
    """Validate and return a 2D float array of class probabilities (N, C)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {p.shape}")
    return p


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-label Expected Calibration Error.

    For each sample the top predicted class and its confidence are taken. Samples
    are bucketed into ``n_bins`` equal-width confidence bins over [0, 1]; the ECE
    is the count-weighted mean absolute gap between bin accuracy and bin
    confidence.
    """
    p = _as_probs(probs)
    labels = np.asarray(y_true).astype(np.int64).ravel()
    if labels.shape[0] != p.shape[0]:
        raise ValueError(f"y_true/probs length mismatch: {labels.shape[0]} != {p.shape[0]}")
    if p.shape[0] == 0:
        return 0.0

    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = labels.shape[0]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = float(correct[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """Per-bin reliability statistics for a top-label reliability diagram.

    Returns a dict with ``bin_confidence``, ``bin_accuracy``, ``bin_count`` (one
    entry per bin; NaN for empty bins where a mean is undefined) and the
    ``bin_edges`` (length ``n_bins + 1``). Values are plain python lists.
    """
    p = _as_probs(probs)
    labels = np.asarray(y_true).astype(np.int64).ravel()
    if labels.shape[0] != p.shape[0]:
        raise ValueError(f"y_true/probs length mismatch: {labels.shape[0]} != {p.shape[0]}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidence: list[float] = []
    bin_accuracy: list[float] = []
    bin_count: list[float] = []

    if p.shape[0] == 0:
        for _ in range(n_bins):
            bin_confidence.append(float("nan"))
            bin_accuracy.append(float("nan"))
            bin_count.append(0.0)
        return {
            "bin_confidence": bin_confidence,
            "bin_accuracy": bin_accuracy,
            "bin_count": bin_count,
            "bin_edges": [float(e) for e in edges],
        }

    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        count = int(in_bin.sum())
        bin_count.append(float(count))
        if count == 0:
            bin_confidence.append(float("nan"))
            bin_accuracy.append(float("nan"))
        else:
            bin_confidence.append(float(confidences[in_bin].mean()))
            bin_accuracy.append(float(correct[in_bin].mean()))

    return {
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_count": bin_count,
        "bin_edges": [float(e) for e in edges],
    }


class TemperatureScaler:
    """Single-parameter temperature scaling for multiclass logits.

    Fits a positive scalar temperature ``T`` that minimizes the negative
    log-likelihood of the validation logits via LBFGS (torch). ``transform``
    divides logits by ``T`` and returns softmax probabilities.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature: float = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        """Optimize a scalar ``T > 0`` on NLL using LBFGS."""
        import torch

        logits_arr = np.asarray(logits, dtype=np.float64)
        if logits_arr.ndim != 2:
            raise ValueError(f"logits must be 2D (N, C), got shape {logits_arr.shape}")
        labels_arr = np.asarray(labels).astype(np.int64).ravel()
        if labels_arr.shape[0] != logits_arr.shape[0]:
            raise ValueError(
                f"logits/labels length mismatch: {logits_arr.shape[0]} != {labels_arr.shape[0]}"
            )
        if logits_arr.shape[0] == 0:
            self.temperature = 1.0
            return self

        logits_t = torch.tensor(logits_arr, dtype=torch.float64)
        labels_t = torch.tensor(labels_arr, dtype=torch.long)

        # Optimize log_t for an unconstrained, strictly-positive temperature.
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
        nll = torch.nn.functional.cross_entropy

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temp = torch.exp(log_t)
            loss = nll(logits_t / temp, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature = float(torch.exp(log_t).item())
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            self.temperature = 1.0
        # Guard against degenerate fits on tiny / ill-conditioned validation sets
        # (e.g. a handful of near-uniform logits can drive T to absurd values).
        self.temperature = float(np.clip(self.temperature, 1e-2, 100.0))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature and return softmax probabilities (N, C)."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        if logits_arr.ndim != 2:
            raise ValueError(f"logits must be 2D (N, C), got shape {logits_arr.shape}")
        scaled = logits_arr / float(self.temperature)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)


def confidence_to_uncertainty_flag(
    confidence: float,
    high: float = 0.85,
    moderate: float = 0.60,
) -> str:
    """Map a confidence value to a research uncertainty flag.

    ``>= high`` -> "high_confidence"; ``>= moderate`` -> "moderate_confidence";
    otherwise "review_required". Flag strings come from
    :data:`spinescoutx.constants.UNCERTAINTY_FLAGS`.
    """
    high_flag, moderate_flag, review_flag = UNCERTAINTY_FLAGS
    c = float(confidence)
    if c >= high:
        return high_flag
    if c >= moderate:
        return moderate_flag
    return review_flag


def calibration_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray | None = None,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Aggregate calibration diagnostics into a flat dict.

    Always reports ``ece`` and the ``reliability_curve``. When ``logits`` are
    provided a :class:`TemperatureScaler` is fit and ``temperature`` plus
    ``ece_after`` (ECE of the temperature-scaled probabilities) are added.
    """
    report: dict[str, Any] = {
        "ece": expected_calibration_error(y_true, probs, n_bins=n_bins),
        "reliability_curve": reliability_curve(y_true, probs, n_bins=n_bins),
    }
    if logits is not None:
        scaler = TemperatureScaler().fit(logits, y_true)
        scaled_probs = scaler.transform(logits)
        report["temperature"] = scaler.temperature
        report["ece_after"] = expected_calibration_error(y_true, scaled_probs, n_bins=n_bins)
    return report
