"""Severe-first Safety Mode: operating points + abstention, on the AUTO distribution.

This is the *decision layer* on top of a trained grader (it does not retrain). It
turns calibrated per-node probabilities into safety-oriented behaviour and reports the
honest trade-offs on the real (auto-localized) inference distribution:

* **operating points** — ``balanced`` (argmax), ``safety`` (lower the severe
  threshold to reach a target severe recall, report the false-alarm cost),
  ``calibrated`` (argmax with the calibrated probabilities).
* **abstention / review policy** — flag low-confidence nodes for human review; a
  severe finding is "safe" if the model calls it severe **or** it is sent to review.
  We report, per review burden, the *effective severe recall with review* and the
  fraction of the model's severe false-negatives that abstention captures.
* **cost-weighted score** — an explicit cost matrix with severe false-negatives far
  more expensive than false alarms.

Honesty: if hitting a target severe recall needs a large false-alarm budget or a high
abstention rate, that is reported, not hidden. Research-only. Not diagnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import SEVERE_INDEX, SEVERITY_SAMPLE_WEIGHTS
from .severe_frontier import recall_at_far_budget, sweep_severe_threshold, threshold_for_recall


def _argmax(p: np.ndarray) -> np.ndarray:
    return np.argmax(p, axis=1)


def operating_points(
    y: np.ndarray, probs: np.ndarray, *, target_recalls: tuple[float, ...] = (0.90, 0.95)
) -> dict[str, Any]:
    """Balanced (argmax) vs safety (severe-threshold for target recall) operating points."""
    y = np.asarray(y).astype(int)
    p = np.asarray(probs).astype(float)
    p_sev = p[:, SEVERE_INDEX]
    is_sev = y == SEVERE_INDEX
    n_sev = int(is_sev.sum())
    n_neg = int((~is_sev).sum())

    pred = _argmax(p)
    bal = {
        "severe_recall": float(((pred == SEVERE_INDEX) & is_sev).sum() / max(n_sev, 1)),
        "false_alarm_rate": float(((pred == SEVERE_INDEX) & ~is_sev).sum() / max(n_neg, 1)),
        "threshold": "argmax",
    }
    sweep = sweep_severe_threshold(y, p_sev)
    safety = {f"recall>={r}": threshold_for_recall(sweep, r) for r in target_recalls}
    return {"n": int(y.size), "n_severe": n_sev, "balanced_argmax": bal, "safety": safety}


def abstention_curve(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    aux_conf: np.ndarray | None = None,
    n_steps: int = 41,
) -> list[dict[str, float]]:
    """Confidence-based review policy. A node is abstained (sent to review) when its
    top-class confidence (optionally min'd with an auxiliary localizer confidence)
    falls below a threshold. Reports, per threshold: review burden, and the
    *effective severe recall with review* (severe is safe if predicted severe OR
    abstained), plus the share of the model's severe FNs captured by review.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(probs).astype(float)
    pred = _argmax(p)
    conf = p.max(axis=1)
    if aux_conf is not None:
        conf = np.minimum(conf, np.asarray(aux_conf).astype(float))
    is_sev = y == SEVERE_INDEX
    n_sev = int(is_sev.sum())
    model_sev_fn = is_sev & (pred != SEVERE_INDEX)  # severe missed by the model
    n_fn = int(model_sev_fn.sum())

    rows: list[dict[str, float]] = []
    for tau in np.linspace(0.0, 1.0, n_steps):
        abstain = conf < tau
        retained = ~abstain
        # severe is "safe" if model called it severe (and retained) OR it was abstained
        safe_sev = is_sev & ((pred == SEVERE_INDEX) & retained | abstain)
        eff_recall = float(safe_sev.sum() / max(n_sev, 1))
        fn_captured = int((model_sev_fn & abstain).sum())
        rows.append(
            {
                "tau": float(tau),
                "abstain_rate": float(abstain.mean()),
                "effective_severe_recall_with_review": eff_recall,
                "model_severe_recall_on_retained": float(
                    ((pred == SEVERE_INDEX) & is_sev & retained).sum()
                    / max(int((is_sev & retained).sum()), 1)
                ),
                "severe_fn_capture_frac": float(fn_captured / max(n_fn, 1)),
                "n_abstained": int(abstain.sum()),
            }
        )
    return rows


def review_burden_for_target(
    curve: list[dict[str, float]], target_eff_recall: float
) -> dict[str, float]:
    """Lowest-abstention operating point whose effective-severe-recall-with-review
    reaches ``target_eff_recall``."""
    feasible = [r for r in curve if r["effective_severe_recall_with_review"] >= target_eff_recall]
    if not feasible:
        return {"reached": False, "target": target_eff_recall}
    best = min(feasible, key=lambda r: r["abstain_rate"])
    return {"reached": True, "target": target_eff_recall, **best}


def cost_weighted_score(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    cost_fn_severe: float = 10.0,
    cost_fn_moderate: float = 2.0,
    cost_fp: float = 1.0,
) -> dict[str, float]:
    """Total/mean misclassification cost under an explicit asymmetric cost matrix.

    Severe false-negatives (true severe predicted normal/mild) are the most costly;
    over-calling (predicting a higher severity than truth) costs ``cost_fp`` per step.
    Returns total and per-node cost (lower is better).
    """
    y = np.asarray(y).astype(int)
    pred = _argmax(np.asarray(probs).astype(float))
    cost = 0.0
    for t, pr in zip(y, pred, strict=False):
        if t == SEVERE_INDEX and pr == 0:
            cost += cost_fn_severe
        elif t == SEVERE_INDEX and pr == 1 or t == 1 and pr == 0:
            cost += cost_fn_moderate
        elif pr > t:  # over-call (false alarm direction)
            cost += cost_fp * (pr - t)
    n = max(int(y.size), 1)
    return {"total_cost": float(cost), "mean_cost": float(cost / n)}


def safety_report(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    aux_conf: np.ndarray | None = None,
    target_recalls: tuple[float, ...] = (0.90, 0.95),
) -> dict[str, Any]:
    """Full safety-mode report on one model's auto-distribution predictions."""
    curve = abstention_curve(y, probs, aux_conf=aux_conf)
    far = {
        f"far<={b}": recall_at_far_budget(
            sweep_severe_threshold(y, np.asarray(probs)[:, SEVERE_INDEX]), b
        )
        for b in (0.05, 0.10, 0.20)
    }
    return {
        "operating_points": operating_points(y, probs, target_recalls=target_recalls),
        "recall_at_far": far,
        "abstention_curve": curve,
        "review_burden": {
            f"eff_recall>={r}": review_burden_for_target(curve, r) for r in target_recalls
        },
        "cost_weighted": cost_weighted_score(y, probs),
        "rsna_weights": list(SEVERITY_SAMPLE_WEIGHTS),
    }
