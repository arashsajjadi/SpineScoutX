"""Cluster (study-level) bootstrap confidence intervals + paired tests.

Statistical discipline for SpineScoutX: severe cases are rare (canal val has ~87
severe nodes), so point estimates of severe recall / weighted log loss are noisy.
Every headline comparison should carry a 95% CI and an explicit ``n`` / ``n_severe``.

Resampling is **clustered by ``study_id``** (the unit of independence): a study and
all its level-nodes are resampled together, which is the honest unit since nodes
within a study are correlated. Determinism is guaranteed by an explicit integer
seed (no clocks, no global RNG).

Research-only. Not diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..constants import SEVERE_INDEX, SEVERITY_SAMPLE_WEIGHTS
from .calibration import expected_calibration_error
from .classification_metrics import (
    balanced_accuracy,
    severe_auroc,
    severe_recall,
    weighted_log_loss,
)
from .severe_frontier import recall_at_far_budget, severe_auroc_pr, sweep_severe_threshold

# A metric function maps (y_true (N,), probs (N, 3)) -> float.
MetricFn = Callable[[np.ndarray, np.ndarray], float]


# --------------------------------------------------------------------------- #
# standard metric closures over (y, probs3)
# --------------------------------------------------------------------------- #
def _argmax(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1)


def m_weighted_logloss(y: np.ndarray, p: np.ndarray) -> float:
    return weighted_log_loss(y, p, SEVERITY_SAMPLE_WEIGHTS)


def m_severe_recall(y: np.ndarray, p: np.ndarray) -> float:
    return severe_recall(y, _argmax(p))


def m_severe_fnr(y: np.ndarray, p: np.ndarray) -> float:
    if int((y == SEVERE_INDEX).sum()) == 0:
        return float("nan")
    return 1.0 - severe_recall(y, _argmax(p))


def m_balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return balanced_accuracy(y, _argmax(p))


def m_severe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return severe_auroc(y, p)


def m_severe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return severe_auroc_pr(y, p[:, SEVERE_INDEX])["severe_ap"]


def m_ece(y: np.ndarray, p: np.ndarray) -> float:
    return expected_calibration_error(y, p)


def make_recall_at_far(far_budget: float) -> MetricFn:
    """Severe recall achievable at false-alarm-rate <= ``far_budget`` (sweeps P(severe))."""

    def _fn(y: np.ndarray, p: np.ndarray) -> float:
        sweep = sweep_severe_threshold(y, p[:, SEVERE_INDEX])
        return recall_at_far_budget(sweep, far_budget)["severe_recall"]

    return _fn


STANDARD_METRICS: dict[str, MetricFn] = {
    "weighted_logloss": m_weighted_logloss,
    "severe_recall": m_severe_recall,
    "severe_fnr": m_severe_fnr,
    "balanced_accuracy": m_balanced_accuracy,
    "severe_auroc": m_severe_auroc,
    "severe_ap": m_severe_ap,
    "ece": m_ece,
    "recall_at_far05": make_recall_at_far(0.05),
    "recall_at_far10": make_recall_at_far(0.10),
    "recall_at_far20": make_recall_at_far(0.20),
}


# --------------------------------------------------------------------------- #
# clustered resampling
# --------------------------------------------------------------------------- #
def _group_members(groups: Sequence[Any]) -> tuple[list[Any], dict[Any, np.ndarray]]:
    """Return (unique_groups, {group -> member row indices})."""
    g = np.asarray(groups)
    uniq = list(dict.fromkeys(g.tolist()))  # stable order
    members = {u: np.flatnonzero(g == u) for u in uniq}
    return uniq, members


def _resample_indices(
    uniq: list[Any], members: dict[Any, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Draw a cluster-bootstrap row-index sample (resample whole groups w/ replacement)."""
    picks = rng.integers(0, len(uniq), size=len(uniq))
    chosen = [members[uniq[i]] for i in picks]
    return np.concatenate(chosen) if chosen else np.array([], dtype=int)


def bootstrap_ci(
    y: np.ndarray,
    probs: np.ndarray,
    groups: Sequence[Any],
    metric_fn: MetricFn,
    *,
    n_boot: int = 2000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Cluster-bootstrap point estimate + (1-alpha) percentile CI for one metric.

    ``y`` is ``(N,)`` int labels; ``probs`` is ``(N, 3)``; ``groups`` is ``(N,)``
    study ids. Returns ``{point, ci_lo, ci_hi, n, n_severe, n_boot}``.
    """
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs).astype(float)
    uniq, members = _group_members(groups)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y, probs))
    samples: list[float] = []
    for _ in range(n_boot):
        idx = _resample_indices(uniq, members, rng)
        val = metric_fn(y[idx], probs[idx])
        if np.isfinite(val):
            samples.append(float(val))
    if samples:
        lo = float(np.percentile(samples, 100 * alpha / 2))
        hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    else:
        lo = hi = float("nan")
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": int(y.size),
        "n_severe": int((y == SEVERE_INDEX).sum()),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_delta(
    y: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    groups: Sequence[Any],
    metric_fn: MetricFn,
    *,
    n_boot: int = 2000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired cluster-bootstrap CI for ``metric(A) - metric(B)`` on the SAME nodes.

    ``probs_a`` and ``probs_b`` are aligned row-for-row with ``y`` / ``groups``
    (same node order). The same resample is applied to both, so the CI reflects the
    *paired* difference (the correct test when the two systems are scored on
    identical study/level nodes). ``decisive`` is True when the delta CI excludes 0.
    """
    y = np.asarray(y).astype(int)
    pa = np.asarray(probs_a).astype(float)
    pb = np.asarray(probs_b).astype(float)
    uniq, members = _group_members(groups)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y, pa) - metric_fn(y, pb))
    samples: list[float] = []
    for _ in range(n_boot):
        idx = _resample_indices(uniq, members, rng)
        va = metric_fn(y[idx], pa[idx])
        vb = metric_fn(y[idx], pb[idx])
        if np.isfinite(va) and np.isfinite(vb):
            samples.append(float(va - vb))
    if samples:
        lo = float(np.percentile(samples, 100 * alpha / 2))
        hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        decisive = bool(lo > 0 or hi < 0)
    else:
        lo = hi = float("nan")
        decisive = False
    return {"delta": point, "ci_lo": lo, "ci_hi": hi, "decisive": decisive, "n_boot": int(n_boot)}


def mcnemar_severe(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on severe-class hits for two systems on the same nodes.

    Restricts to true-severe nodes; counts discordant pairs ``b`` (A correct, B
    wrong) and ``c`` (A wrong, B correct) and returns the two-sided exact binomial
    p-value. ``pred_a`` / ``pred_b`` are hard argmax predictions aligned with ``y``.
    """
    from scipy.stats import binomtest

    y = np.asarray(y).astype(int)
    a = np.asarray(pred_a).astype(int)
    b_pred = np.asarray(pred_b).astype(int)
    sev = y == SEVERE_INDEX
    a_hit = (a == SEVERE_INDEX) & sev
    b_hit = (b_pred == SEVERE_INDEX) & sev
    b = int((a_hit & ~b_hit).sum())  # A catches, B misses
    c = int((~a_hit & b_hit).sum())  # A misses, B catches
    n = b + c
    if n == 0:
        p = 1.0
    else:
        p = float(binomtest(b, n, 0.5).pvalue)
    return {"b_a_catches_b_misses": b, "c_a_misses_b_catches": c, "n_discordant": n, "p_value": p}


def ci_table(
    y: np.ndarray,
    probs: np.ndarray,
    groups: Sequence[Any],
    *,
    metrics: dict[str, MetricFn] | None = None,
    n_boot: int = 2000,
    seed: int = 1337,
) -> dict[str, dict[str, float]]:
    """Bootstrap CIs for a standard battery of metrics (or a supplied subset)."""
    metrics = metrics or STANDARD_METRICS
    return {
        name: bootstrap_ci(y, probs, groups, fn, n_boot=n_boot, seed=seed)
        for name, fn in metrics.items()
    }
