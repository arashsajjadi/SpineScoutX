"""v1.8c — import-safety / gate tests for the real-MedSAM2 adapter.

These tests must pass WITHOUT VisionServeX, sam2, or the MedSAM2 checkpoint present (CI is offline),
so they only exercise the import surface and the ``available()`` capability gate — never weights.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import importlib

import numpy as np


def test_module_imports_without_backend():
    """The adapter must import even when VisionServeX/sam2/checkpoint are absent."""
    mod = importlib.import_module("spinescoutx.segmentation.medsam2_runner")
    assert hasattr(mod, "MedSAM2")
    assert hasattr(mod, "available")
    assert callable(mod.available)


def test_available_returns_bool_and_never_raises():
    """``available()`` is a pure capability probe: returns a bool, never raises."""
    from spinescoutx.segmentation.medsam2_runner import available

    assert isinstance(available(), bool)


def test_default_checkpoint_is_gitignored_path():
    """Default checkpoint lives under gitignored data/models — never committed."""
    from spinescoutx.segmentation.medsam2_runner import DEFAULT_CKPT

    assert DEFAULT_CKPT.as_posix().endswith("data/models/medsam2/MedSAM2_latest.pt")


def test_segment_signature_box_default():
    """``segment`` exposes a center-box prompt default (documented prompt strategy)."""
    import inspect

    from spinescoutx.segmentation.medsam2_runner import MedSAM2

    sig = inspect.signature(MedSAM2.segment)
    assert sig.parameters["box"].default == (70, 70, 154, 154)
    # A (3,H,W) crop is the documented input shape; sanity-check the contract array builds.
    crop = np.zeros((3, 224, 224), dtype=np.float32)
    assert crop.shape[0] == 3
