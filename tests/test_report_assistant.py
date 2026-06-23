"""Tests for the research report-assistant markdown builder."""

from __future__ import annotations

from spinescoutx.reporting.assistant import build_assistant_markdown

GRAPH = {
    "study_id": "1",
    "dataset_source": "rsna",
    "model_version": "0.1.0",
    "run_id": "e2",
    "limitations": ["Not diagnostic."],
    "findings": [
        {"level": "l4_l5", "condition": "spinal_canal_stenosis", "side": None, "grade": "severe",
         "calibrated_confidence": 0.7, "uncertainty_flag": "moderate_confidence",
         "evidence_region": "spinal_canal", "evidence_region_source": "anatomy"},
        {"level": "l4_l5", "condition": "left_subarticular_stenosis", "side": "left",
         "grade": "normal_mild", "calibrated_confidence": 0.9, "uncertainty_flag": "high_confidence",
         "evidence_region": "lateral_recess_left", "evidence_region_source": "approximate"},
    ],
}
BASELINE = {
    "findings": [
        {"level": "l4_l5", "condition": "spinal_canal_stenosis", "grade": "moderate"},
        {"level": "l4_l5", "condition": "left_subarticular_stenosis", "grade": "normal_mild"},
    ]
}


def test_assistant_has_disclaimer_and_summary() -> None:
    md = build_assistant_markdown(GRAPH)
    assert "not diagnostic" in md.lower()
    assert "No treatment recommendation" in md
    assert "Severity grade summary" in md
    assert "severe" in md.lower()


def test_assistant_evidence_validity_counts() -> None:
    md = build_assistant_markdown(GRAPH)
    # one anatomy region (canal) + one approximate (lateral recess)
    assert "anatomy (validated region" in md
    assert "approximate" in md


def test_assistant_cross_model_severe_disagreement() -> None:
    md = build_assistant_markdown(GRAPH, baseline_graph=BASELINE, baseline_name="E0", model_name="E2")
    assert "Cross-model check" in md
    # canal stenosis: E0=moderate vs E2=severe -> a severe disagreement must be flagged
    assert "severe-grade disagreements" in md
    assert "E0=moderate vs E2=severe" in md
