"""Deterministic seeding for reproducible runs."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np


def seed_everything(seed: int = 1337, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and Torch RNGs and return the seed.

    With ``deterministic=True`` we also request deterministic cuDNN kernels. This
    can slow training slightly but is required for reproducible research numbers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # use_deterministic_algorithms can raise on ops with no deterministic
            # impl; warn-only keeps training robust without silently hiding it.
            with contextlib.suppress(RuntimeError, AttributeError):
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so each worker is deterministically seeded."""
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
