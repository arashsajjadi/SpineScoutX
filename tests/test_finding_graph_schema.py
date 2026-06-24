"""Tests for the study-level finding-graph schema (build, validate, render)."""

from __future__ import annotations

import pytest

from spinescoutx.reporting import finding_graph_schema as fg


def _graph(probs=(0.05, 0.15, 0.80), **kw):
    f = fg.build_finding("spinal_canal_stenosis", "l4_l5", list(probs), **kw)
    return fg.build_study_graph("12345", split="test", findings=[f], model_version="vTest")


def test_case_id_is_hashed_and_deterministic():
    a, b = fg.case_id("12345"), fg.case_id("12345")
    assert a == b and a.startswith("case_") and "12345" not in a


def test_severity_is_argmax_and_probs_preserved():
    f = fg.build_finding("spinal_canal_stenosis", "l4_l5", [0.1, 0.2, 0.7])
    assert f["severity_estimate"] == "severe"
    assert f["probabilities"]["P(severe)"] == 0.7
    assert f["view_route"] == "sagittal_t2"


def test_view_route_matches_condition():
    assert (
        fg.build_finding("left_neural_foraminal_narrowing", "l5_s1", [0.6, 0.3, 0.1])["view_route"]
        == "sagittal_t1"
    )
    assert (
        fg.build_finding("right_subarticular_stenosis", "l4_l5", [0.6, 0.3, 0.1])["view_route"]
        == "axial_t2"
    )


def test_review_reasons_from_real_signals():
    # low confidence + near-severe + disagreement
    f = fg.build_finding(
        "spinal_canal_stenosis", "l4_l5", [0.40, 0.35, 0.25], model_disagreement=True
    )
    assert f["review_required"] is True
    assert "low_confidence" in f["review_reasons"]
    assert "near_severe_threshold" in f["review_reasons"]
    assert "model_disagreement" in f["review_reasons"]
    # confident, clean -> no review
    g = fg.build_finding("spinal_canal_stenosis", "l4_l5", [0.97, 0.02, 0.01])
    assert g["review_required"] is False and g["review_reasons"] == []


def test_axial_level_uncertainty_reason():
    f = fg.build_finding(
        "left_subarticular_stenosis", "l4_l5", [0.9, 0.06, 0.04], axial_level_score=0.3
    )
    assert "axial_level_uncertainty" in f["review_reasons"]
    g = fg.build_finding(
        "left_subarticular_stenosis", "l4_l5", [0.9, 0.06, 0.04], axial_level_score=0.9
    )
    assert "axial_level_uncertainty" not in g["review_reasons"]


def test_validate_passes_on_good_graph():
    fg.validate_finding_graph(_graph())


def test_validate_rejects_bad_probabilities():
    g = _graph()
    g["findings"][0]["probabilities"]["P(severe)"] = 0.4  # now sums to 0.6
    with pytest.raises(ValueError):
        fg.validate_finding_graph(g)


def test_validate_rejects_forbidden_wording():
    g = _graph()
    g["findings"][0]["evidence"]["notes"] = "this is a diagnosis of severe stenosis"
    with pytest.raises(ValueError):
        fg.validate_finding_graph(g)


def test_validate_allows_not_diagnostic_disclaimer():
    # the disclaimer contains "not diagnostic" which must be permitted
    fg.validate_finding_graph(_graph())  # disclaimer is present by construction


def test_render_markdown_reflects_json():
    g = _graph(probs=(0.05, 0.15, 0.80))
    md = fg.render_markdown(g)
    assert g["case_id"] in md
    assert "0.8" in md  # P(severe)
    assert "severe" in md
    assert "not diagnostic" in md.lower()


def test_every_finding_has_provenance():
    g = _graph()
    assert all("crop_provenance" in f for f in g["findings"])
