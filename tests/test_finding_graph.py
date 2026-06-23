"""Tests for deterministic finding-graph construction and validation."""

from __future__ import annotations

import math

from spinescoutx.constants import SEVERITIES
from spinescoutx.reporting.finding_graph import (
    REQUIRED_FINDING_KEYS,
    REQUIRED_GRAPH_KEYS,
    build_finding_graph,
    finding_graph_from_dict,
    finding_graph_to_dict,
    grade_from_probs,
    validate_finding_graph,
)


def _sample_predictions() -> list[dict]:
    return [
        {
            "level": "l4_l5",
            "condition": "spinal_canal_stenosis",
            "side": None,
            "probs": [0.1, 0.2, 0.7],
            "evidence_consistency": 0.8,
            "crop_path": "cache/crop0.npy",
        },
        {
            "level": "l5_s1",
            "condition": "left_neural_foraminal_narrowing",
            "side": "left",
            "probs": [0.9, 0.05, 0.05],
        },
    ]


def test_grade_from_probs() -> None:
    grade, conf = grade_from_probs([0.1, 0.2, 0.7])
    assert grade == SEVERITIES[2]
    assert math.isclose(conf, 0.7, abs_tol=1e-9)


def test_build_and_validate_graph() -> None:
    graph = build_finding_graph(
        "study_001",
        _sample_predictions(),
        run_id="run_abc",
        model_version="0.1.0",
        dataset_source="rsna",
    )
    d = finding_graph_to_dict(graph)
    assert validate_finding_graph(d) == []

    for key in REQUIRED_GRAPH_KEYS:
        assert key in d
    assert d["research_only"] is True
    assert d["not_diagnostic"] is True

    for finding in d["findings"]:
        for key in REQUIRED_FINDING_KEYS:
            assert key in finding


def test_finding_graph_round_trip() -> None:
    graph = build_finding_graph(
        "study_002",
        _sample_predictions(),
        run_id="run_xyz",
        model_version="0.1.0",
        dataset_source="rsna",
    )
    d = finding_graph_to_dict(graph)
    restored = finding_graph_from_dict(d)
    assert restored.study_id == graph.study_id
    assert len(restored.findings) == len(graph.findings)
    assert finding_graph_to_dict(restored) == d


def test_validate_finding_graph_rejects_bad_enum() -> None:
    graph = build_finding_graph(
        "study_003",
        _sample_predictions(),
        run_id="run_bad",
        model_version="0.1.0",
        dataset_source="rsna",
    )
    d = finding_graph_to_dict(graph)
    d["findings"][0]["grade"] = "not_a_grade"
    problems = validate_finding_graph(d)
    assert any("invalid grade" in p for p in problems)


def test_validate_finding_graph_requires_research_only() -> None:
    graph = build_finding_graph(
        "study_004",
        _sample_predictions(),
        run_id="run_r",
        model_version="0.1.0",
        dataset_source="rsna",
    )
    d = finding_graph_to_dict(graph)
    d["research_only"] = False
    problems = validate_finding_graph(d)
    assert any("research_only" in p for p in problems)
