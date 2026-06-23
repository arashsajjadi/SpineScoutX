"""Tests for the fail-closed LLM report safety filter (no Ollama required)."""

from __future__ import annotations

from spinescoutx.reporting.llm_report import (
    check_llm_safety,
    generate_safe_llm_report,
)

GRAPH = {
    "study_id": "123",
    "model_version": "0.1.0",
    "dataset_source": "rsna",
    "findings": [
        {"level": "l4_l5", "condition": "spinal_canal_stenosis", "side": None, "grade": "moderate"},
    ],
}

SAFE = (
    "Research-only, not diagnostic, not clinically validated, not for medical decision-making. "
    "At level l4_l5, the spinal canal stenosis finding was graded moderate. "
    "Research-only, not diagnostic."
)


def test_safe_text_passes() -> None:
    ok, reasons = check_llm_safety(SAFE, GRAPH)
    assert ok, reasons


def test_missing_disclaimer_rejected() -> None:
    ok, reasons = check_llm_safety("At l4_l5 the finding was moderate.", GRAPH)
    assert not ok
    assert any("disclaimer" in r for r in reasons)


def test_treatment_claim_rejected() -> None:
    txt = SAFE + " The patient should start treatment and we recommend surgery."
    ok, reasons = check_llm_safety(txt, GRAPH)
    assert not ok


def test_out_of_scope_pathology_rejected() -> None:
    txt = SAFE + " There is evidence of a tumor."
    ok, reasons = check_llm_safety(txt, GRAPH)
    assert not ok


def test_positive_diagnosis_rejected() -> None:
    txt = SAFE + " The diagnosis is canal stenosis."
    ok, reasons = check_llm_safety(txt, GRAPH)
    assert not ok


def test_hallucinated_grade_rejected() -> None:
    # 'severe' is not present in the graph (only 'moderate').
    txt = SAFE + " The stenosis is severe."
    ok, reasons = check_llm_safety(txt, GRAPH)
    assert not ok
    assert any("severe" in r for r in reasons)


def test_generate_fails_closed_when_ollama_unavailable() -> None:
    # Unreachable host -> ok False, no fabricated text.
    result = generate_safe_llm_report(GRAPH, "no-such-model", host="http://127.0.0.1:1")
    assert result["ok"] is False
    assert result["text"] is None
