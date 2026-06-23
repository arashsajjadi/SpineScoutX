"""Tests for JSON + Markdown report generation from a finding graph."""

from __future__ import annotations

from spinescoutx.reporting.finding_graph import (
    build_finding_graph,
    finding_graph_to_dict,
    validate_finding_graph,
)
from spinescoutx.reporting.json_report import read_json_report, write_json_report
from spinescoutx.reporting.markdown_report import (
    TOP_DISCLAIMER,
    render_markdown_report,
    write_markdown_report,
)


def _graph():
    predictions = [
        {
            "level": "l4_l5",
            "condition": "spinal_canal_stenosis",
            "side": None,
            "probs": [0.05, 0.15, 0.80],
            "evidence_consistency": 0.72,
            "evidence_image_path": "outputs/evidence/l4l5.png",
        },
        {
            "level": "l5_s1",
            "condition": "right_subarticular_stenosis",
            "side": "right",
            "probs": [0.6, 0.3, 0.1],
        },
    ]
    return build_finding_graph(
        "study_report",
        predictions,
        run_id="run_report",
        model_version="0.1.0",
        dataset_source="rsna",
    )


def test_write_json_report_validates(tmp_path) -> None:
    graph = _graph()
    path = write_json_report(graph, tmp_path / "report.json")
    assert path.exists()
    data = read_json_report(path)
    assert validate_finding_graph(data) == []
    assert data == finding_graph_to_dict(graph)


def test_markdown_contains_disclaimers(tmp_path) -> None:
    graph = _graph()
    md = render_markdown_report(graph)
    assert TOP_DISCLAIMER in md
    assert "Not diagnostic" in md or "not diagnostic" in md
    assert "No medical advice" in md

    path = write_markdown_report(graph, tmp_path / "report.md")
    assert path.exists()
    written = path.read_text(encoding="utf-8")
    assert TOP_DISCLAIMER in written


def test_markdown_lists_findings(tmp_path) -> None:
    graph = _graph()
    md = render_markdown_report(graph)
    assert "spinal_canal_stenosis" in md
    assert "right_subarticular_stenosis" in md
