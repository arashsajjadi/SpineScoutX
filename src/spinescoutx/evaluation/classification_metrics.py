"""Classification metrics for severity grading (research-only, deterministic).

All functions take ``y_true: np.ndarray`` shaped ``(N,)`` of integer severity
labels and either hard predictions ``y_pred (N,)`` or probabilities
``probs (N, NUM_SEVERITY_CLASSES)``. Inputs may be numpy arrays or torch tensors;
they are converted to numpy internally. No randomness, no clocks, no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ..constants import (
    NUM_SEVERITY_CLASSES,
    SEVERE_INDEX,
    SEVERITY_SAMPLE_WEIGHTS,
)


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert a numpy array or torch tensor to a contiguous numpy array."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def _as_int_labels(y_true: Any) -> np.ndarray:
    """Coerce labels to a 1-D int64 numpy array."""
    return _to_numpy(y_true).astype(np.int64).reshape(-1)


def weighted_log_loss(
    y_true: Any,
    probs: Any,
    class_weights: Sequence[float] = SEVERITY_SAMPLE_WEIGHTS,
    eps: float = 1e-15,
) -> float:
    """RSNA-style sample-weighted log loss over severity classes.

    For each sample ``i`` with true label ``y_i`` and predicted probability
    vector ``p_i`` (clipped to ``[eps, 1]``), the per-sample weight is
    ``w_i = class_weights[y_i]``. The loss is::

        loss = sum_i ( w_i * -log( clip(p_i[y_i], eps, 1) ) ) / sum_i w_i

    i.e. a weighted mean of the negative log-probability of the true class,
    where the weight depends only on the true class. Perfectly confident,
    correct probabilities give a loss of ~0.

    Parameters
    ----------
    y_true:
        Integer labels, shape ``(N,)``.
    probs:
        Probabilities, shape ``(N, C)``.
    class_weights:
        Per-class weight indexed by the true label.
    eps:
        Lower clip bound for probabilities.

    Returns
    -------
    float
        The weighted log loss. ``0.0`` if there are no samples.
    """
    y = _as_int_labels(y_true)
    p = _to_numpy(probs).astype(np.float64)
    if y.size == 0:
        return 0.0
    p = np.clip(p, eps, 1.0)
    weights = np.asarray(class_weights, dtype=np.float64)
    w_i = weights[y]
    true_probs = p[np.arange(y.size), y]
    neg_log = -np.log(true_probs)
    denom = float(np.sum(w_i))
    if denom == 0.0:
        return 0.0
    return float(np.sum(w_i * neg_log) / denom)


def macro_f1(y_true: Any, y_pred: Any) -> float:
    """Macro-averaged F1 over all severity classes."""
    y = _as_int_labels(y_true)
    yp = _as_int_labels(y_pred)
    if y.size == 0:
        return 0.0
    labels = list(range(NUM_SEVERITY_CLASSES))
    return float(f1_score(y, yp, labels=labels, average="macro", zero_division=0))


def per_group_f1(y_true: Any, y_pred: Any, groups: Any) -> dict[str, float]:
    """Macro-F1 computed within each unique group value.

    ``groups`` is an array aligned with ``y_true``; the returned dict maps each
    group (as ``str``) to the macro-F1 of its samples.
    """
    y = _as_int_labels(y_true)
    yp = _as_int_labels(y_pred)
    g = _to_numpy(groups).reshape(-1)
    out: dict[str, float] = {}
    for value in np.unique(g):
        mask = g == value
        out[str(value)] = macro_f1(y[mask], yp[mask])
    return out


def severe_recall(y_true: Any, y_pred: Any, severe_index: int = SEVERE_INDEX) -> float:
    """Recall (sensitivity) for the severe class.

    Returns ``0.0`` if no severe samples are present.
    """
    y = _as_int_labels(y_true)
    yp = _as_int_labels(y_pred)
    positives = y == severe_index
    n_pos = int(np.sum(positives))
    if n_pos == 0:
        return 0.0
    true_pos = int(np.sum(positives & (yp == severe_index)))
    return float(true_pos / n_pos)


def severe_false_negative_rate(y_true: Any, y_pred: Any) -> float:
    """False-negative rate for the severe class (= 1 - severe_recall).

    Returns ``0.0`` if no severe samples are present.
    """
    y = _as_int_labels(y_true)
    if int(np.sum(y == SEVERE_INDEX)) == 0:
        return 0.0
    return float(1.0 - severe_recall(y_true, y_pred))


def severe_auroc(y_true: Any, probs: Any) -> float:
    """One-vs-rest AUROC for the severe class (NaN-safe).

    Returns ``float('nan')`` when only one class is present among the binarized
    labels (AUROC is undefined).
    """
    y = _as_int_labels(y_true)
    p = _to_numpy(probs).astype(np.float64)
    if y.size == 0:
        return float("nan")
    binary = (y == SEVERE_INDEX).astype(np.int64)
    if np.unique(binary).size < 2:
        return float("nan")
    scores = p[:, SEVERE_INDEX]
    return float(roc_auc_score(binary, scores))


def balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    """Balanced accuracy (mean per-class recall)."""
    y = _as_int_labels(y_true)
    yp = _as_int_labels(y_pred)
    if y.size == 0:
        return 0.0
    return float(balanced_accuracy_score(y, yp))


def confusion(y_true: Any, y_pred: Any, num_classes: int = NUM_SEVERITY_CLASSES) -> np.ndarray:
    """Confusion matrix of shape ``(num_classes, num_classes)`` (rows = true)."""
    y = _as_int_labels(y_true)
    yp = _as_int_labels(y_pred)
    labels = list(range(num_classes))
    return confusion_matrix(y, yp, labels=labels).astype(np.int64)


def classification_report_dict(
    y_true: Any,
    probs: Any,
    *,
    conditions: Any | None = None,
    levels: Any | None = None,
) -> dict[str, Any]:
    """Aggregate severity metrics into a single flat dictionary.

    Hard predictions are derived as ``argmax(probs, axis=1)``. When
    ``conditions`` / ``levels`` arrays (aligned with ``y_true``) are supplied,
    per-group macro-F1 dicts are included; otherwise those entries are empty.

    Keys (exact):
        weighted_logloss, macro_f1, balanced_accuracy, severe_recall,
        severe_fnr, severe_auroc, confusion, per_condition_f1, per_level_f1.
    """
    y = _as_int_labels(y_true)
    p = _to_numpy(probs).astype(np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    y_pred = np.argmax(p, axis=1) if p.size else np.zeros_like(y)

    per_condition_f1: dict[str, float] = {}
    if conditions is not None:
        per_condition_f1 = per_group_f1(y, y_pred, conditions)
    per_level_f1: dict[str, float] = {}
    if levels is not None:
        per_level_f1 = per_group_f1(y, y_pred, levels)

    return {
        "weighted_logloss": weighted_log_loss(y, p),
        "macro_f1": macro_f1(y, y_pred),
        "balanced_accuracy": balanced_accuracy(y, y_pred),
        "severe_recall": severe_recall(y, y_pred),
        "severe_fnr": severe_false_negative_rate(y, y_pred),
        "severe_auroc": severe_auroc(y, p),
        "confusion": confusion(y, y_pred).tolist(),
        "per_condition_f1": per_condition_f1,
        "per_level_f1": per_level_f1,
    }
