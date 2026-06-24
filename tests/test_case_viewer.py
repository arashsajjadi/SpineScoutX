"""Tests for the real case-viewer layer (correctness derived, no leakage)."""

from __future__ import annotations

import pytest

from spinescoutx.reporting import case_viewer as cv
from spinescoutx.reporting import finding_graph_schema as fg


def _graph(findings):
    return fg.build_study_graph("100206310", split="test", findings=findings, model_version="v1.2")


def test_finding_correctness_severe_axis():
    assert cv.finding_correctness("severe", "severe") == "severe_correct"
    assert cv.finding_correctness("moderate", "severe") == "severe_false_negative"
    assert cv.finding_correctness("severe", "normal_mild") == "severe_false_positive"
    assert cv.finding_correctness("moderate", "moderate") == "exact_correct"
    assert cv.finding_correctness("moderate", "normal_mild") == "non_severe_mismatch"
    assert cv.finding_correctness("severe", None) == "no_reference"


def test_correctness_is_derived_not_hardcoded():
    # ref severe but model predicts normal -> must be a severe FN
    f = fg.build_finding(
        "spinal_canal_stenosis", "l4_l5", [0.9, 0.05, 0.05], reference_label="severe"
    )
    g = _graph([f])
    view = cv.build_case_viewer(g)
    cv.validate_case_viewer(view)
    vf = view["findings"][0]
    assert vf["severity_estimate"] == "normal_mild"
    assert vf["correctness"] == "severe_false_negative"
    assert vf["held_out_reference_label"] == "severe"
    assert view["case_category"] == "false_negative"


def test_correct_severe_category_and_summary():
    f = fg.build_finding(
        "spinal_canal_stenosis", "l4_l5", [0.05, 0.1, 0.85], reference_label="severe"
    )
    view = cv.build_case_viewer(_graph([f]))
    cv.validate_case_viewer(view)
    assert view["findings"][0]["correctness"] == "severe_correct"
    assert view["case_category"] == "correct_severe"
    assert view["study_summary"]["n_exact_correct"] == 1
    assert view["study_summary"]["n_severe_errors"] == 0
    assert view["study_summary"]["worst_failure_mode"] == "none"


def test_hard_right_foraminal_category():
    f = fg.build_finding(
        "right_neural_foraminal_narrowing", "l5_s1", [0.8, 0.15, 0.05], reference_label="severe"
    )
    view = cv.build_case_viewer(_graph([f]))
    assert view["case_category"] == "hard_right_foraminal"
    assert view["study_summary"]["worst_failure_mode"] == "severe_false_negative"


def test_mostly_normal_category():
    fs = [
        fg.build_finding(
            "spinal_canal_stenosis", lv, [0.95, 0.04, 0.01], reference_label="normal_mild"
        )
        for lv in ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")
    ]
    view = cv.build_case_viewer(_graph(fs))
    cv.validate_case_viewer(view)
    assert view["case_category"] == "mostly_normal"
    assert view["study_summary"]["n_review_required"] == 0


def test_validator_catches_tampered_correctness():
    f = fg.build_finding(
        "spinal_canal_stenosis", "l4_l5", [0.05, 0.1, 0.85], reference_label="normal_mild"
    )
    view = cv.build_case_viewer(_graph([f]))
    # genuine: severe prediction vs normal ref -> severe_false_positive
    assert view["findings"][0]["correctness"] == "severe_false_positive"
    view["findings"][0]["correctness"] = "severe_correct"  # tamper
    with pytest.raises(ValueError, match="derived"):
        cv.validate_case_viewer(view)


def test_no_dicom_path_leak():
    f = fg.build_finding(
        "spinal_canal_stenosis", "l4_l5", [0.6, 0.3, 0.1], reference_label="moderate"
    )
    view = cv.build_case_viewer(_graph([f]))
    view["findings"][0]["leak"] = "/data/raw/rsna/x/y/10.dcm"  # inject a leak
    with pytest.raises(ValueError, match="DICOM|leak"):
        cv.validate_case_viewer(view)


def test_markdown_reflects_json():
    f1 = fg.build_finding(
        "spinal_canal_stenosis", "l4_l5", [0.05, 0.1, 0.85], reference_label="severe"
    )
    f2 = fg.build_finding(
        "left_subarticular_stenosis", "l5_s1", [0.7, 0.2, 0.1], reference_label="moderate"
    )
    view = cv.build_case_viewer(_graph([f1, f2]))
    md = cv.render_case_markdown(view)
    assert view["case_id"] in md
    assert "held-out ref" in md.lower()
    for f in view["findings"]:
        assert f["held_out_reference_label"] in md
        assert str(f["probabilities"]["P(severe)"]) in md
    # correctness marks present
    assert "severe-correct" in md


def test_instability_type_threaded():
    f = fg.build_finding(
        "left_subarticular_stenosis", "l4_l5", [0.3, 0.2, 0.5], reference_label="severe"
    )
    view = cv.build_case_viewer(
        _graph([f]),
        instability_types={
            ("left_subarticular_stenosis", "l4_l5", "left"): "axial_candidate_sensitive"
        },
    )
    cv.validate_case_viewer(view)
    assert view["findings"][0]["instability_type"] == "axial_candidate_sensitive"
