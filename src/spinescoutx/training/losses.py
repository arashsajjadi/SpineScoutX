"""Classification and segmentation loss functions for SpineScoutX.

Pure ``torch``; no training loop, no I/O, no global mutable state. The builders
read the relevant fields off :class:`~spinescoutx.config.TrainConfig` and return
callables / modules that the training loops consume.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..config import TrainConfig
from ..constants import NUM_SEVERITY_CLASSES


def severity_class_weights(counts: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.

    A class with zero observed samples contributes no inverse-frequency mass
    (its weight is set to 0 before normalization) so empty classes never produce
    infinities. The returned tensor is float32 of length ``len(counts)``.
    """
    counts_t = torch.as_tensor(list(counts), dtype=torch.float64)
    inv = torch.where(counts_t > 0, 1.0 / counts_t, torch.zeros_like(counts_t))
    total = inv.sum()
    if float(total) <= 0.0:
        return torch.ones(len(counts), dtype=torch.float32)
    nonzero = int((inv > 0).sum().item())
    weights = inv / total * nonzero
    return weights.to(dtype=torch.float32)


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean cross-entropy with optional per-class weighting.

    ``logits`` is ``(B, C)``; ``targets`` is ``(B,)`` long. ``class_weights`` (if
    given) is length ``C`` and is moved to the logits' device/dtype.
    """
    weight = None
    if class_weights is not None:
        weight = class_weights.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, targets, weight=weight)


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weighting.

    ``forward(logits (B, C), targets (B,) long) -> scalar``.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if weight is None:
            self.register_buffer("weight", None, persistent=False)
        else:
            self.register_buffer("weight", weight.to(dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.weight is not None:
            weight = self.weight.to(device=logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        log_pt = -F.cross_entropy(logits, targets, reduction="none")
        pt = log_pt.exp()
        loss = (1.0 - pt).pow(self.gamma) * ce
        return loss.mean()


def severe_aware_cost_matrix(
    num_classes: int = NUM_SEVERITY_CLASSES,
    *,
    fn_severe: float = 10.0,
    fn_moderate: float = 2.0,
    over_call: float = 1.0,
) -> torch.Tensor:
    """Asymmetric cost matrix ``C[true, pred]`` for ordinal severity grading.

    Under-grading a severe finding (true severe, predicted normal/mild) is the most
    costly entry; under-grading moderate is next; over-grading (predicting a higher
    severity than truth) costs ``over_call`` per step. The diagonal is 0. Used by
    :class:`ExpectedCostLoss` to make severe false-negatives expensive at training
    time, complementing the inference-time Safety Mode.
    """
    c = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    severe = num_classes - 1
    for true in range(num_classes):
        for pred in range(num_classes):
            if pred == true:
                continue
            if pred < true:  # under-grading
                gap = true - pred
                if true == severe:
                    c[true, pred] = fn_severe if pred == 0 else fn_moderate * gap
                else:
                    c[true, pred] = fn_moderate * gap
            else:  # over-grading (false-alarm direction)
                c[true, pred] = over_call * (pred - true)
    return c


class ExpectedCostLoss(nn.Module):
    """Differentiable expected-cost loss: ``mean_i sum_j p_i[j] * C[y_i, j]``.

    Minimises the expected misclassification cost under an asymmetric cost matrix
    (see :func:`severe_aware_cost_matrix`), so the model is penalised in proportion to
    how much probability it puts on costly (e.g. severe-miss) predictions.
    """

    def __init__(self, cost_matrix: torch.Tensor, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("cost", cost_matrix.to(dtype=torch.float32))
        if weight is None:
            self.register_buffer("weight", None, persistent=False)
        else:
            self.register_buffer("weight", weight.to(dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        costs = self.cost.to(device=logits.device)[targets.long()]  # (B, C)
        per_sample = (probs * costs).sum(dim=1)
        # Per-class weighting is essential on imbalanced data: without it the expected
        # cost is minimised by a degenerate "predict the cheap middle class" hedge.
        if self.weight is not None:
            w = self.weight.to(device=logits.device, dtype=logits.dtype)[targets.long()]
            return (per_sample * w).sum() / w.sum().clamp_min(1e-8)
        return per_sample.mean()


def _to_onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert an index label map to a one-hot float tensor with class dim at 1.

    Accepts either an index tensor (no class channel) or an already one-hot
    tensor whose channel dimension equals ``num_classes``.
    """
    if target.dim() >= 1 and target.shape[1:2] == (num_classes,):
        return target.to(dtype=torch.float32)
    onehot = F.one_hot(target.long(), num_classes=num_classes)
    # Move the new last dim to position 1 (channel-first).
    dims = list(range(onehot.dim()))
    perm = [dims[0], dims[-1], *dims[1:-1]]
    return onehot.permute(perm).to(dtype=torch.float32)


def soft_dice_loss(
    logits: torch.Tensor,
    targets_onehot_or_index: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
    ignore_background: bool = False,
) -> torch.Tensor:
    """Soft (differentiable) multiclass Dice loss.

    ``logits`` is ``(B, C, ...)``; the target is either an index map ``(B, ...)``
    or a one-hot ``(B, C, ...)`` tensor. Returns ``1 - mean_dice`` averaged over
    the retained classes. When ``ignore_background`` is True, class 0 is dropped.
    """
    probs = torch.softmax(logits, dim=1)
    target = _to_onehot(targets_onehot_or_index, num_classes).to(
        device=logits.device, dtype=probs.dtype
    )
    if ignore_background:
        probs = probs[:, 1:, ...]
        target = target[:, 1:, ...]
    reduce_dims = tuple(range(2, probs.dim()))
    intersection = (probs * target).sum(dim=reduce_dims)
    cardinality = probs.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Sum of soft Dice and (optionally weighted) cross-entropy for segmentation."""

    def __init__(
        self,
        num_classes: int,
        weight: torch.Tensor | None = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ignore_background: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.ignore_background = bool(ignore_background)
        self.eps = float(eps)
        if weight is None:
            self.register_buffer("weight", None, persistent=False)
        else:
            self.register_buffer("weight", weight.to(dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.weight is not None:
            weight = self.weight.to(device=logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, target.long(), weight=weight)
        dice = soft_dice_loss(
            logits,
            target,
            self.num_classes,
            eps=self.eps,
            ignore_background=self.ignore_background,
        )
        return self.ce_weight * ce + self.dice_weight * dice


class DiceFocalLoss(nn.Module):
    """Sum of soft Dice and multiclass focal loss for segmentation."""

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        ignore_background: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.ignore_background = bool(ignore_background)
        self.eps = float(eps)
        self.focal = FocalLoss(gamma=gamma, weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c = logits.shape[0], logits.shape[1]
        flat_logits = logits.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c)
        flat_target = target.long().reshape(-1)
        focal = self.focal(flat_logits, flat_target)
        dice = soft_dice_loss(
            logits,
            target,
            self.num_classes,
            eps=self.eps,
            ignore_background=self.ignore_background,
        )
        return self.focal_weight * focal + self.dice_weight * dice


def build_classification_loss(
    train_cfg: TrainConfig,
    class_weights: torch.Tensor | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a callable ``(logits, targets) -> scalar`` per ``train_cfg.loss``.

    Supported ``loss`` values: ``"weighted_ce"`` (default) and ``"focal"``.
    Class weights are only applied when ``train_cfg.class_weighted_loss`` is set.
    """
    weights = class_weights if train_cfg.class_weighted_loss else None
    loss_name = train_cfg.loss

    if loss_name == "cost_sensitive":
        # Pass class weights (when enabled) so the expected-cost objective does not
        # collapse to the cheap middle class on imbalanced severity data.
        module = ExpectedCostLoss(severe_aware_cost_matrix(), weight=weights)

        def _cost(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return module(logits, targets)

        return _cost

    if loss_name == "focal":
        module = FocalLoss(weight=weights)

        def _focal(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return module(logits, targets)

        return _focal

    def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return weighted_cross_entropy(logits, targets, weights)

    return _ce


def build_segmentation_loss(train_cfg: TrainConfig, num_classes: int) -> nn.Module:
    """Return a segmentation loss module per ``train_cfg.loss``.

    Supported ``loss`` values: ``"dice_focal"`` -> :class:`DiceFocalLoss`;
    anything else (including ``"dice_ce"``) -> :class:`DiceCELoss`.
    """
    if train_cfg.loss == "dice_focal":
        return DiceFocalLoss(num_classes=num_classes)
    return DiceCELoss(num_classes=num_classes)


# Re-exported for callers that want the default severity class dimension.
__all__ = [
    "NUM_SEVERITY_CLASSES",
    "severity_class_weights",
    "weighted_cross_entropy",
    "FocalLoss",
    "ExpectedCostLoss",
    "severe_aware_cost_matrix",
    "soft_dice_loss",
    "DiceCELoss",
    "DiceFocalLoss",
    "build_classification_loss",
    "build_segmentation_loss",
]
