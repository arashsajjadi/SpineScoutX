"""Tests for evidence-consistency metrics."""

from __future__ import annotations

import math

import numpy as np

from spinescoutx.evaluation.evidence_metrics import (
    anatomical_evidence_consistency,
    evidence_consistency_batch,
    evidence_leakage,
    normalize_map,
    peak_to_localizer_distance,
)


def test_aec_all_mass_in_region() -> None:
    heatmap = np.zeros((8, 8), dtype=np.float64)
    heatmap[2:4, 2:4] = 1.0
    mask = np.zeros((8, 8), dtype=np.float64)
    mask[2:4, 2:4] = 1.0
    aec = anatomical_evidence_consistency(heatmap, mask)
    assert aec is not None
    assert math.isclose(aec, 1.0, abs_tol=1e-9)


def test_aec_no_mass_in_region() -> None:
    heatmap = np.zeros((8, 8), dtype=np.float64)
    heatmap[0:2, 0:2] = 1.0
    mask = np.zeros((8, 8), dtype=np.float64)
    mask[6:8, 6:8] = 1.0
    aec = anatomical_evidence_consistency(heatmap, mask)
    assert aec is not None
    assert math.isclose(aec, 0.0, abs_tol=1e-9)


def test_aec_empty_mask_returns_none() -> None:
    heatmap = np.ones((8, 8), dtype=np.float64)
    mask = np.zeros((8, 8), dtype=np.float64)
    assert anatomical_evidence_consistency(heatmap, mask) is None


def test_leakage_is_one_minus_aec() -> None:
    rng = np.random.default_rng(0)
    heatmap = rng.random((8, 8))
    mask = np.zeros((8, 8), dtype=np.float64)
    mask[3:6, 3:6] = 1.0
    aec = anatomical_evidence_consistency(heatmap, mask)
    leak = evidence_leakage(heatmap, mask)
    assert aec is not None and leak is not None
    assert math.isclose(leak, 1.0 - aec, abs_tol=1e-9)


def test_peak_to_localizer_distance_zero_at_peak() -> None:
    heatmap = np.zeros((10, 10), dtype=np.float64)
    # Peak at row=4, col=7 -> localizer x=7, y=4 should give distance 0.
    heatmap[4, 7] = 5.0
    dist = peak_to_localizer_distance(heatmap, x=7.0, y=4.0)
    assert math.isclose(dist, 0.0, abs_tol=1e-9)


def test_peak_to_localizer_distance_normalized_in_unit_range() -> None:
    heatmap = np.zeros((10, 10), dtype=np.float64)
    heatmap[0, 0] = 1.0
    dist = peak_to_localizer_distance(heatmap, x=9.0, y=9.0, normalize=True)
    assert 0.0 <= dist <= 1.0


def test_normalize_map_zeros_stay_zeros() -> None:
    arr = np.zeros((4, 4), dtype=np.float64)
    out = normalize_map(arr)
    assert np.all(out == 0.0)


def test_evidence_consistency_batch() -> None:
    hm = np.zeros((4, 4), dtype=np.float64)
    hm[1:3, 1:3] = 1.0
    mask_full = np.ones((4, 4), dtype=np.float64)
    mask_empty = np.zeros((4, 4), dtype=np.float64)
    out = evidence_consistency_batch([hm, hm], [mask_full, mask_empty])
    assert out[0] is not None and math.isclose(out[0], 1.0, abs_tol=1e-9)
    assert out[1] is None
