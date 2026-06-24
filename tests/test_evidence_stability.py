"""Unit tests for the pure evidence-stability scoring/grading core."""

from __future__ import annotations

import numpy as np
import pytest

from spinescoutx.evaluation import evidence_stability as es


def test_normalized_entropy_bounds():
    assert es.normalized_entropy([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(1.0, abs=1e-9)
    assert es.normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    mid = es.normalized_entropy([0.8, 0.1, 0.1])
    assert 0.0 < mid < 1.0


def test_stability_stats_invariant_prediction():
    # identical rows -> zero dispersion, zero flips, full agreement
    p = np.tile([0.7, 0.2, 0.1], (9, 1))
    st = es.stability_stats(p)
    assert st["baseline_pred"] == 0
    assert st["p_severe_std"] == pytest.approx(0.0)
    assert st["p_severe_range"] == pytest.approx(0.0)
    assert st["severity_flip_rate"] == 0.0
    assert st["agreement_rate"] == 1.0
    assert st["k_perturb"] == 8


def test_stability_stats_flip_detected():
    # baseline says normal; half the perturbations flip to severe
    rows = [[0.6, 0.1, 0.3]] + [[0.1, 0.1, 0.8]] * 4 + [[0.6, 0.1, 0.3]] * 4
    st = es.stability_stats(np.array(rows))
    assert st["baseline_pred"] == 0
    assert st["severity_flip_rate"] == pytest.approx(0.5)
    assert st["p_severe_range"] > 0.4


def test_stability_stats_rejects_bad_shape():
    with pytest.raises(ValueError):
        es.stability_stats(np.zeros((1, 3)))  # no perturbations
    with pytest.raises(ValueError):
        es.stability_stats(np.zeros((4, 2)))  # not 3-class


def test_stability_grade_transitions():
    cfg = es.PerturbConfig()
    stable = {"severity_flip_rate": 0.0, "p_severe_range": 0.01, "p_severe_std": 0.01}
    mild = {"severity_flip_rate": 0.1, "p_severe_range": 0.2, "p_severe_std": 0.1}
    unstable = {"severity_flip_rate": 0.5, "p_severe_range": 0.6, "p_severe_std": 0.3}
    assert es.stability_grade(stable, cfg) == "stable"
    assert es.stability_grade(mild, cfg) == "mildly_unstable"
    assert es.stability_grade(unstable, cfg) == "unstable"


def test_instability_score_monotonic():
    low = es.instability_score({"severity_flip_rate": 0.0, "p_severe_range": 0.0})
    high = es.instability_score({"severity_flip_rate": 1.0, "p_severe_range": 1.0})
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)
    assert low < es.instability_score({"severity_flip_rate": 0.3, "p_severe_range": 0.2}) < high


def test_sample_offsets_deterministic_and_bounded():
    cfg = es.PerturbConfig(k=8, xy_sigma=4.0, slice_jitter=2, max_offset=48.0)
    a = es.sample_offsets(cfg, np.random.default_rng(0))
    b = es.sample_offsets(cfg, np.random.default_rng(0))
    assert a == b  # same seed -> same offsets
    assert len(a) == 8
    for dx, dy, ds in a:
        assert abs(dx) <= 48.0 and abs(dy) <= 48.0
        assert -2 <= ds <= 2


def test_review_reasons_route_aware():
    # only the strong `unstable` grade raises review reasons
    assert es.stability_review_reasons("stable", "spinal_canal_stenosis") == ()
    assert es.stability_review_reasons("mildly_unstable", "spinal_canal_stenosis") == ()
    canal = es.stability_review_reasons("unstable", "spinal_canal_stenosis")
    assert "evidence_unstable" in canal and "route_unstable" in canal
    sub = es.stability_review_reasons("unstable", "left_subarticular_stenosis")
    assert "axial_candidate_disagreement" in sub
    for_ = es.stability_review_reasons("unstable", "right_neural_foraminal_narrowing")
    assert "foraminal_slice_disagreement" in for_


def test_route_quality_levels():
    assert es.route_quality("stable", 0.9) == "good"
    assert es.route_quality("unstable", 0.9) == "weak"
    assert es.route_quality("stable", 0.2) == "weak"  # very low localizer conf -> weak
    assert es.route_quality("stable", 0.45) == "fair"  # mid conf demotes good->fair
    assert es.route_quality("mildly_unstable", None) == "fair"
    assert es.route_quality("stable", None) == "good"


def test_config_for_known_conditions():
    assert es.config_for("left_subarticular_stenosis").slice_jitter == 2
    assert es.config_for("spinal_canal_stenosis").slice_jitter == 1
    assert es.config_for("unknown_condition").k == 8  # falls back to default


def test_classify_instability_type():
    # stable / low instability -> stable
    assert (
        es.classify_instability_type(0.02, 0.01, 0.01, route="axial_t2", grade="stable") == "stable"
    )
    assert (
        es.classify_instability_type(0.05, 0.03, 0.02, route="axial_t2", grade="unstable")
        == "stable"
    )  # below min_unstable
    # slice dominant on axial -> axial_candidate_sensitive
    assert (
        es.classify_instability_type(0.5, 0.45, 0.05, route="axial_t2", grade="unstable")
        == "axial_candidate_sensitive"
    )
    # slice dominant on sagittal -> slice_sensitive
    assert (
        es.classify_instability_type(0.5, 0.45, 0.05, route="sagittal_t1", grade="unstable")
        == "slice_sensitive"
    )
    # in-plane dominant -> crop_sensitive
    assert (
        es.classify_instability_type(0.5, 0.05, 0.45, route="sagittal_t2", grade="unstable")
        == "crop_sensitive"
    )
    # balanced -> route_sensitive
    assert (
        es.classify_instability_type(0.5, 0.25, 0.25, route="sagittal_t2", grade="unstable")
        == "route_sensitive"
    )
    assert es.classify_instability_type(0.5, 0.0, 0.0, route="axial_t2", grade="unstable") == (
        "route_sensitive"
    )
