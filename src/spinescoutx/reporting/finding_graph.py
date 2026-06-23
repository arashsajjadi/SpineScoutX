"""Deterministic finding-graph construction for SpineScoutX research reports.

A :class:`FindingGraph` is a plain-data, model-free aggregation of per-finding
severity predictions for a single study. It exists so that downstream JSON and
Markdown reporting can be generated reproducibly and validated against the
shared vocabulary in :mod:`spinescoutx.constants`.

Research-only: this module produces no clinical or diagnostic output. Every
graph carries ``research_only`` / ``not_diagnostic`` flags and the standard
:data:`~spinescoutx.constants.RESEARCH_LIMITATIONS`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..constants import (
    CONDITIONS,
    LEVELS,
    RESEARCH_LIMITATIONS,
    SEVERITIES,
    UNCERTAINTY_FLAGS,
    evidence_region_for,
)
from ..evaluation.calibration import confidence_to_uncertainty_flag


@dataclass
class Finding:
    """A single per-(level, condition) severity finding."""

    level: str
    condition: str
    side: str | None
    grade: str
    confidence: float
    calibrated_confidence: float
    uncertainty_flag: str
    evidence_consistency: float | None
    evidence_region: str
    evidence_region_source: str
    evidence_image_path: str | None
    crop_path: str | None
    notes: str


@dataclass
class FindingGraph:
    """All findings for one study plus provenance and research disclaimers."""

    study_id: str
    dataset_source: str
    model_version: str
    run_id: str
    findings: list[Finding]
    limitations: list[str]
    research_only: bool = True
    not_diagnostic: bool = True


REQUIRED_FINDING_KEYS: tuple[str, ...] = (
    "level",
    "condition",
    "side",
    "grade",
    "confidence",
    "calibrated_confidence",
    "uncertainty_flag",
    "evidence_consistency",
    "evidence_region",
    "evidence_region_source",
    "evidence_image_path",
    "crop_path",
    "notes",
)

REQUIRED_GRAPH_KEYS: tuple[str, ...] = (
    "study_id",
    "research_only",
    "not_diagnostic",
    "dataset_source",
    "model_version",
    "run_id",
    "findings",
    "limitations",
)


def grade_from_probs(probs: Sequence[float]) -> tuple[str, float]:
    """Return (severity_label, probability) for the argmax of ``probs``."""
    if len(probs) != len(SEVERITIES):
        raise ValueError(f"Expected {len(SEVERITIES)} severity probabilities, got {len(probs)}.")
    best_index = max(range(len(probs)), key=lambda i: float(probs[i]))
    return SEVERITIES[best_index], float(probs[best_index])


def _finding_from_prediction(
    prediction: dict[str, Any],
    calibrator: Callable[[Sequence[float]], Sequence[float]] | None,
) -> Finding:
    """Build one :class:`Finding` from a prediction dict (deterministically)."""
    level = str(prediction["level"])
    condition = str(prediction["condition"])
    side = prediction.get("side")
    side_value = None if side is None else str(side)

    probs = list(prediction["probs"])
    grade, confidence = grade_from_probs(probs)

    calibrated_probs = prediction.get("calibrated_probs")
    if calibrated_probs is None and calibrator is not None:
        calibrated_probs = list(calibrator(probs))
    if calibrated_probs is None:
        calibrated_probs = probs
    _, calibrated_confidence = grade_from_probs(list(calibrated_probs))

    uncertainty_flag = confidence_to_uncertainty_flag(calibrated_confidence)

    region_name, _region_side, region_source = evidence_region_for(condition)

    evidence_consistency = prediction.get("evidence_consistency")
    evidence_value = None if evidence_consistency is None else float(evidence_consistency)

    evidence_image_path = prediction.get("evidence_image_path")
    crop_path = prediction.get("crop_path")

    return Finding(
        level=level,
        condition=condition,
        side=side_value,
        grade=grade,
        confidence=confidence,
        calibrated_confidence=calibrated_confidence,
        uncertainty_flag=uncertainty_flag,
        evidence_consistency=evidence_value,
        evidence_region=region_name,
        evidence_region_source=region_source,
        evidence_image_path=None if evidence_image_path is None else str(evidence_image_path),
        crop_path=None if crop_path is None else str(crop_path),
        notes=str(prediction.get("notes", "")),
    )


def build_finding_graph(
    study_id: str,
    predictions: list[dict[str, Any]],
    *,
    run_id: str,
    model_version: str,
    dataset_source: str,
    calibrator: Callable[[Sequence[float]], Sequence[float]] | None = None,
) -> FindingGraph:
    """Build a deterministic :class:`FindingGraph` from raw predictions.

    Each prediction dict must provide ``level``, ``condition``, ``side`` and
    ``probs`` (length-3 severity probabilities). Optional keys are
    ``calibrated_probs``, ``evidence_consistency``, ``evidence_image_path``,
    ``crop_path`` and ``notes``. The uncertainty flag is derived from the
    calibrated confidence and the evidence region from
    :func:`~spinescoutx.constants.evidence_region_for`.
    """
    findings = [_finding_from_prediction(pred, calibrator) for pred in predictions]
    return FindingGraph(
        study_id=str(study_id),
        dataset_source=str(dataset_source),
        model_version=str(model_version),
        run_id=str(run_id),
        findings=findings,
        limitations=list(RESEARCH_LIMITATIONS),
    )


def finding_graph_to_dict(graph: FindingGraph) -> dict[str, Any]:
    """Serialise a :class:`FindingGraph` to a plain dict."""
    return dataclasses.asdict(graph)


def finding_graph_from_dict(d: dict[str, Any]) -> FindingGraph:
    """Reconstruct a :class:`FindingGraph` from a plain dict."""
    findings = [
        Finding(**{key: raw.get(key) for key in REQUIRED_FINDING_KEYS})
        for raw in d.get("findings", [])
    ]
    return FindingGraph(
        study_id=str(d["study_id"]),
        dataset_source=str(d.get("dataset_source", "")),
        model_version=str(d.get("model_version", "")),
        run_id=str(d.get("run_id", "")),
        findings=findings,
        limitations=list(d.get("limitations", [])),
        research_only=bool(d.get("research_only", True)),
        not_diagnostic=bool(d.get("not_diagnostic", True)),
    )


def _validate_finding(raw: dict[str, Any], index: int) -> list[str]:
    """Return a list of problems for a single finding dict."""
    problems: list[str] = []
    prefix = f"findings[{index}]"

    for key in REQUIRED_FINDING_KEYS:
        if key not in raw:
            problems.append(f"{prefix}: missing required key {key!r}")

    level = raw.get("level")
    if level is not None and level not in LEVELS:
        problems.append(f"{prefix}: invalid level {level!r}")

    condition = raw.get("condition")
    if condition is not None and condition not in CONDITIONS:
        problems.append(f"{prefix}: invalid condition {condition!r}")

    grade = raw.get("grade")
    if grade is not None and grade not in SEVERITIES:
        problems.append(f"{prefix}: invalid grade {grade!r}")

    flag = raw.get("uncertainty_flag")
    if flag is not None and flag not in UNCERTAINTY_FLAGS:
        problems.append(f"{prefix}: invalid uncertainty_flag {flag!r}")

    return problems


def validate_finding_graph(d: dict[str, Any]) -> list[str]:
    """Return a list of validation problems; an empty list means valid."""
    problems: list[str] = []

    for key in REQUIRED_GRAPH_KEYS:
        if key not in d:
            problems.append(f"missing required key {key!r}")

    if d.get("research_only") is not True:
        problems.append("research_only must be True")
    if d.get("not_diagnostic") is not True:
        problems.append("not_diagnostic must be True")

    findings = d.get("findings")
    if not isinstance(findings, list):
        problems.append("findings must be a list")
    else:
        for index, raw in enumerate(findings):
            if not isinstance(raw, dict):
                problems.append(f"findings[{index}]: must be a mapping")
                continue
            problems.extend(_validate_finding(raw, index))

    limitations = d.get("limitations")
    if limitations is not None and not isinstance(limitations, list):
        problems.append("limitations must be a list")

    return problems


__all__ = [
    "Finding",
    "FindingGraph",
    "REQUIRED_FINDING_KEYS",
    "REQUIRED_GRAPH_KEYS",
    "build_finding_graph",
    "finding_graph_from_dict",
    "finding_graph_to_dict",
    "grade_from_probs",
    "validate_finding_graph",
]
