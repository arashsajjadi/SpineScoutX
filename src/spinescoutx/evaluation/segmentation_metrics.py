"""Segmentation metrics over the 4-class anatomy scheme (research-only).

Predictions and targets are integer label maps with values in
``range(NUM_ANATOMY_CLASSES)``. Inputs may be numpy arrays or torch tensors;
they are converted to numpy internally. Per-class Dice / IoU are keyed by
``ANATOMY_CLASSES``. ``mean_dice`` averages over the foreground classes only.
No randomness, no clocks, no network.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import (
    ANATOMY_CLASS_TO_INDEX,
    ANATOMY_CLASSES,
    FOREGROUND_ANATOMY_CLASSES,
    NUM_ANATOMY_CLASSES,
)

_CANAL_INDEX = ANATOMY_CLASS_TO_INDEX["spinal_canal"]


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert a numpy array or torch tensor to a contiguous numpy array."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def _as_int_labels(arr: Any) -> np.ndarray:
    """Coerce a label map to an int64 numpy array."""
    return _to_numpy(arr).astype(np.int64)


def dice_per_class(
    pred: Any,
    target: Any,
    num_classes: int = NUM_ANATOMY_CLASSES,
    eps: float = 1e-6,
    ignore_background: bool = True,
) -> dict[str, float]:
    """Per-class Dice coefficient keyed by ``ANATOMY_CLASSES``.

    Dice for class ``c`` is ``2|P∩G| / (|P| + |G|)`` with an ``eps`` smoothing.
    When ``ignore_background`` is True the background class is omitted from the
    returned dict.
    """
    p = _as_int_labels(pred).reshape(-1)
    g = _as_int_labels(target).reshape(-1)
    out: dict[str, float] = {}
    for cls_idx in range(num_classes):
        name = ANATOMY_CLASSES[cls_idx]
        if ignore_background and name == "background":
            continue
        pred_c = p == cls_idx
        gt_c = g == cls_idx
        inter = float(np.sum(pred_c & gt_c))
        denom = float(np.sum(pred_c) + np.sum(gt_c))
        out[name] = float((2.0 * inter + eps) / (denom + eps))
    return out


def iou_per_class(
    pred: Any,
    target: Any,
    num_classes: int = NUM_ANATOMY_CLASSES,
    eps: float = 1e-6,
    ignore_background: bool = True,
) -> dict[str, float]:
    """Per-class IoU (Jaccard index) keyed by ``ANATOMY_CLASSES``."""
    p = _as_int_labels(pred).reshape(-1)
    g = _as_int_labels(target).reshape(-1)
    out: dict[str, float] = {}
    for cls_idx in range(num_classes):
        name = ANATOMY_CLASSES[cls_idx]
        if ignore_background and name == "background":
            continue
        pred_c = p == cls_idx
        gt_c = g == cls_idx
        inter = float(np.sum(pred_c & gt_c))
        union = float(np.sum(pred_c | gt_c))
        out[name] = float((inter + eps) / (union + eps))
    return out


def mean_dice(per_class: dict[str, float]) -> float:
    """Mean Dice over the foreground anatomy classes present in ``per_class``."""
    values = [per_class[name] for name in FOREGROUND_ANATOMY_CLASSES if name in per_class]
    if not values:
        return 0.0
    return float(np.mean(values))


def canal_dice(pred: Any, target: Any, eps: float = 1e-6) -> float:
    """Dice coefficient for the spinal-canal class only."""
    p = _as_int_labels(pred).reshape(-1)
    g = _as_int_labels(target).reshape(-1)
    pred_c = p == _CANAL_INDEX
    gt_c = g == _CANAL_INDEX
    inter = float(np.sum(pred_c & gt_c))
    denom = float(np.sum(pred_c) + np.sum(gt_c))
    return float((2.0 * inter + eps) / (denom + eps))


class SegMetricAccumulator:
    """Streaming accumulator for per-class Dice / IoU across batches.

    Sums per-class intersection / prediction / target cardinalities over many
    ``update`` calls, then derives Dice and IoU once in ``compute``. This keeps
    memory constant and yields dataset-level (micro) coefficients.
    """

    def __init__(
        self,
        num_classes: int = NUM_ANATOMY_CLASSES,
        eps: float = 1e-6,
        ignore_background: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background
        self._intersection = np.zeros(num_classes, dtype=np.float64)
        self._pred_sum = np.zeros(num_classes, dtype=np.float64)
        self._target_sum = np.zeros(num_classes, dtype=np.float64)

    def update(self, pred_batch: Any, target_batch: Any) -> None:
        """Accumulate intersection / cardinality sums from one batch."""
        p = _as_int_labels(pred_batch).reshape(-1)
        g = _as_int_labels(target_batch).reshape(-1)
        for cls_idx in range(self.num_classes):
            pred_c = p == cls_idx
            gt_c = g == cls_idx
            self._intersection[cls_idx] += float(np.sum(pred_c & gt_c))
            self._pred_sum[cls_idx] += float(np.sum(pred_c))
            self._target_sum[cls_idx] += float(np.sum(gt_c))

    def compute(self) -> dict[str, Any]:
        """Return accumulated Dice / IoU dicts plus mean_dice and canal_dice."""
        dice: dict[str, float] = {}
        iou: dict[str, float] = {}
        for cls_idx in range(self.num_classes):
            name = ANATOMY_CLASSES[cls_idx]
            if self.ignore_background and name == "background":
                continue
            inter = self._intersection[cls_idx]
            denom = self._pred_sum[cls_idx] + self._target_sum[cls_idx]
            union = self._pred_sum[cls_idx] + self._target_sum[cls_idx] - inter
            dice[name] = float((2.0 * inter + self.eps) / (denom + self.eps))
            iou[name] = float((inter + self.eps) / (union + self.eps))

        canal_inter = self._intersection[_CANAL_INDEX]
        canal_denom = self._pred_sum[_CANAL_INDEX] + self._target_sum[_CANAL_INDEX]
        canal = float((2.0 * canal_inter + self.eps) / (canal_denom + self.eps))

        return {
            "dice": dice,
            "iou": iou,
            "mean_dice": mean_dice(dice),
            "canal_dice": canal,
        }
