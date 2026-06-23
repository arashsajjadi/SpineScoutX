"""Shared synthetic fixtures for the SpineScoutX test suite.

Every fixture here is deterministic (seeded ``numpy.random.default_rng``) and uses
tiny sizes so the whole suite stays fast on CPU with no real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spinescoutx.constants import NUM_SEVERITY_CLASSES, SEVERE_INDEX


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the suite."""
    config.addinivalue_line("markers", "slow: end-to-end synthetic training smoke tests")


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic numpy random generator."""
    return np.random.default_rng(1337)


@pytest.fixture
def repo_root_path() -> Path:
    """Absolute path to the repository root (parent of the tests directory)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def perfect_probs() -> tuple[np.ndarray, np.ndarray]:
    """(y_true, probs) where probs put ~all mass on the true class."""
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    probs = np.full((y_true.size, NUM_SEVERITY_CLASSES), 1e-9, dtype=np.float64)
    probs[np.arange(y_true.size), y_true] = 1.0 - 2e-9
    return y_true, probs


@pytest.fixture
def overconfident_logits() -> tuple[np.ndarray, np.ndarray]:
    """Over-confident logits + labels: predictions are right ~half the time.

    Logits are large in magnitude (so softmax confidence is high) but the argmax
    only matches the label for half the samples, which makes the classifier badly
    calibrated and leaves room for temperature scaling to reduce the NLL.
    """
    rng = np.random.default_rng(7)
    n = 60
    labels = rng.integers(0, NUM_SEVERITY_CLASSES, size=n).astype(np.int64)
    logits = np.zeros((n, NUM_SEVERITY_CLASSES), dtype=np.float64)
    for i in range(n):
        # Make the model confidently predict label i for the first half and a
        # deterministically wrong class for the second half.
        if i % 2 == 0:
            peak = int(labels[i])
        else:
            peak = int((labels[i] + 1) % NUM_SEVERITY_CLASSES)
        logits[i, peak] = 8.0
    return logits, labels


@pytest.fixture
def severe_index() -> int:
    """The integer index of the severe severity class."""
    return SEVERE_INDEX
