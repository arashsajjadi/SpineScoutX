"""Tests for the disc-level localizer + auto-crop honesty guard."""

from __future__ import annotations

import inspect

import numpy as np

from spinescoutx.data import auto_localize
from spinescoutx.data.localizer import (
    extract_peaks,
    gaussian_heatmaps,
    pck,
    peak_confidence,
)


def test_gaussian_heatmap_peaks_at_point() -> None:
    pts = np.array([[10, 20], [40, 50]], dtype=np.float32)
    hm = gaussian_heatmaps(pts, size=64, sigma=3.0)
    assert hm.shape == (2, 64, 64)
    for i, (x, y) in enumerate(pts):
        yy, xx = np.unravel_index(int(np.argmax(hm[i])), hm[i].shape)
        assert abs(xx - x) <= 1 and abs(yy - y) <= 1


def test_extract_peaks_roundtrip() -> None:
    pts = np.array([[5, 5], [30, 40], [60, 10]], dtype=np.float32)
    hm = gaussian_heatmaps(pts, size=64, sigma=2.0)
    rec = extract_peaks(hm)
    assert np.allclose(rec, pts, atol=1.0)


def test_five_distinct_level_peaks() -> None:
    pts = np.array([[10, 10], [10, 25], [10, 40], [10, 55], [30, 60]], dtype=np.float32)
    hm = gaussian_heatmaps(pts, size=64, sigma=2.0)
    rec = extract_peaks(hm)
    assert rec.shape == (5, 2)
    # all peaks distinct (one per level channel)
    assert len({tuple(p) for p in rec}) == 5


def test_missing_keypoint_is_zero_channel() -> None:
    pts = np.array([[10, 10], [np.nan, np.nan]], dtype=np.float32)
    hm = gaussian_heatmaps(pts, size=32, sigma=2.0)
    assert hm[1].sum() == 0.0
    assert peak_confidence(hm)[1] == 0.0


def test_pck() -> None:
    pred = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float32)
    gt = np.array([[5, 0], [15, 0], [40, 0]], dtype=np.float32)
    out = pck(pred, gt, (10, 20, 32))
    assert out["pck@10"] == 1 / 3  # only the dist-5 point is within 10
    assert out["pck@20"] == 2 / 3
    assert out["pck@32"] == 2 / 3


def test_auto_path_never_reads_gt_coordinates() -> None:
    """The auto-crop module must not depend on train_label_coordinates.csv."""
    src = inspect.getsource(auto_localize)
    # No functional use of the GT-coordinate loader anywhere in the auto path.
    assert "load_coordinates" not in src
    # It is allowed to read labels (severity target) and the series index.
    assert "load_labels" in src


def test_disc_localizer_forward() -> None:
    import torch

    from spinescoutx.config import ModelConfig
    from spinescoutx.models.disc_localizer import build_disc_localizer

    m = build_disc_localizer(ModelConfig(kind="disc_localizer", in_chans=1, embed_dim=16))
    out = m(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 5, 64, 64)
    assert torch.isfinite(m.heatmaps(torch.randn(1, 1, 64, 64))).all()
