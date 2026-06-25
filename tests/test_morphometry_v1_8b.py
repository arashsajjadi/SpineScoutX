"""Unit tests for v1.8b segmentation-morphometry features (pure functions, synthetic masks)."""

from __future__ import annotations

import numpy as np
import pytest

from spinescoutx.features.morphometry import (
    FEATURE_COLS,
    foraminal_features,
    stability_iou,
)


def _crop(val=0.5):
    return np.full((3, 64, 64), val, dtype=np.float32)


def test_features_present_and_finite():
    mask = np.zeros((64, 64), bool)
    mask[20:40, 24:40] = True
    f = foraminal_features(mask, _crop(), iou=0.8)
    for c in FEATURE_COLS:
        assert c in f
        assert np.isfinite(f[c])
    assert f["m_seg_fail"] == 0.0
    assert 0 < f["m_area_frac"] < 1
    assert abs(f["m_compactness"] - 1.0) < 1e-6  # filled rectangle fills its bbox


def test_empty_mask_flags_seg_fail():
    f = foraminal_features(np.zeros((64, 64), bool), _crop(), iou=0.1)
    assert f["m_seg_fail"] == 1.0
    assert f["m_area_frac"] == 0.0


def test_contrast_sign():
    # bright object on dark background -> positive contrast
    crop = np.zeros((3, 64, 64), dtype=np.float32)
    crop[1, 20:40, 24:40] = 1.0
    mask = np.zeros((64, 64), bool)
    mask[20:40, 24:40] = True
    f = foraminal_features(mask, crop, iou=0.9)
    assert f["m_contrast"] > 0.5
    assert f["m_intensity_mean"] > 0.5


def test_min_opening_tracks_extent():
    wide = np.zeros((64, 64), bool)
    wide[28:36, 8:56] = True  # short height -> small min_open
    tall = np.zeros((64, 64), bool)
    tall[8:56, 28:36] = True
    fw = foraminal_features(wide, _crop(), iou=0.7)
    ft = foraminal_features(tall, _crop(), iou=0.7)
    assert fw["m_min_open"] == pytest.approx(ft["m_min_open"], abs=0.02)  # symmetric extents
    assert fw["m_aspect"] < 1 < ft["m_aspect"]  # wide is landscape, tall is portrait


def test_stability_iou():
    a = np.zeros((10, 10), bool)
    a[2:8, 2:8] = True
    assert stability_iou(a, a) == pytest.approx(1.0)
    assert stability_iou(a, np.zeros((10, 10), bool)) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
