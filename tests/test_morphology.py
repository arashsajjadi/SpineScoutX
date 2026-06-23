"""Tests for the morphology feature engine."""

from __future__ import annotations

import numpy as np

from spinescoutx.features.morphology import (
    FEATURE_NAMES,
    NUM_FEATURES,
    feature_vector,
    morphology_features,
)


def _mask_with_canal(size: int, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    m = np.zeros((3, size, size), dtype=np.float32)
    m[1, y0:y1, x0:x1] = 1.0  # channel 1 = spinal_canal
    return m


def test_feature_vector_shape_and_order() -> None:
    m = _mask_with_canal(64, 28, 36, 10, 54)
    vec = feature_vector(m)
    assert vec.shape == (NUM_FEATURES,)
    feats = morphology_features(m)
    assert [feats[n] for n in FEATURE_NAMES] == list(vec)


def test_empty_canal_is_zero_but_valid() -> None:
    m = np.zeros((3, 32, 32), dtype=np.float32)
    f = morphology_features(m)
    assert f["canal_present"] == 0.0
    assert f["canal_area"] == 0.0
    assert f["min_canal_width"] == 0.0
    assert np.isfinite(feature_vector(m)).all()


def test_wider_canal_has_larger_width_and_area() -> None:
    narrow = _mask_with_canal(64, 30, 34, 10, 54)  # width 4
    wide = _mask_with_canal(64, 24, 40, 10, 54)  # width 16
    fn, fw = morphology_features(narrow), morphology_features(wide)
    assert fw["canal_area"] > fn["canal_area"]
    assert fw["min_canal_width"] > fn["min_canal_width"]
    assert fw["canal_present"] == fn["canal_present"] == 1.0


def test_centroid_and_symmetry() -> None:
    # a centred rectangular canal is left/right symmetric about its centroid
    m = _mask_with_canal(64, 24, 40, 8, 56)
    f = morphology_features(m)
    assert abs(f["canal_cx"] - 0.5) < 0.05
    assert f["canal_lr_asymmetry"] < 0.1
    assert 0.0 <= f["canal_compactness"] <= 1.0


def test_2d_mask_treated_as_canal() -> None:
    canal2d = np.zeros((40, 40), dtype=np.float32)
    canal2d[10:30, 18:22] = 1.0
    f = morphology_features(canal2d)
    assert f["canal_present"] == 1.0
    assert f["canal_area"] > 0.0
