"""Device selection, optimizer / scheduler builders, and early stopping.

Pure ``torch``; no training loop and no I/O. Deterministic and free of global
mutable state.
"""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def select_device(pref: str = "auto") -> torch.device:
    """Resolve a torch device from a preference string.

    ``"auto"`` picks CUDA when available, else CPU. ``"cuda"`` is honored only
    when CUDA is actually available; otherwise it falls back to CPU. ``"cpu"``
    always returns CPU.
    """
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> Optimizer:
    """Build an AdamW optimizer over the model's trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Optimizer,
    epochs: int,
    steps_per_epoch: int,
) -> LRScheduler | None:
    """Build a cosine-annealing scheduler over ``epochs * steps_per_epoch`` steps.

    Returns ``None`` when the total number of steps is zero (or non-positive), so
    callers can skip ``scheduler.step()`` safely.
    """
    total_steps = int(epochs) * int(steps_per_epoch)
    if total_steps <= 0:
        return None
    return CosineAnnealingLR(optimizer, T_max=total_steps)


class EarlyStopping:
    """Track a monitored metric and signal when to stop training.

    ``mode="min"`` treats lower values as better; ``mode="max"`` treats higher
    values as better. ``min_delta`` is the minimum change that counts as an
    improvement. :meth:`step` returns ``True`` when the new value is a new best.
    """

    def __init__(self, patience: int, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: float = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Record ``value``; return ``True`` if it is a new best."""
        value = float(value)
        if self._is_improvement(value):
            self.best = value
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        """True once the metric has failed to improve for ``patience`` steps."""
        return self.num_bad_epochs >= self.patience


__all__ = [
    "select_device",
    "build_optimizer",
    "build_scheduler",
    "EarlyStopping",
]
