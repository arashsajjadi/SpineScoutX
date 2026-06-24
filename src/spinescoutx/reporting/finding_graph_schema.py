"""Study-level non-diagnostic finding-graph schema (v5): build, validate, render.

The "model output" of SpineScoutX is a structured **research finding graph**: per
(condition, level, side) a severity estimate (argmax of the real softmax), class
probabilities, calibrated confidence, an uncertainty flag, and `review_required` with
reasons derived from *real* model signals (low confidence, high entropy, model
disagreement, axial level-scorer uncertainty, near-severe threshold) — plus the view route
and crop provenance.

v5 adds **evidence-aware** fields: an `evidence_stability` block (how much P(severe) moves
when the same grader is re-run on plausible localizer perturbations — see
`evaluation.evidence_stability`) and a `route_quality` signal (good/fair/weak from stability
+ localizer confidence). Instability can raise `review_required` via `evidence_unstable` /
route-specific reasons. This module builds that dict, validates it (probabilities sum ≈ 1,
every finding has provenance, allowed review reasons, NO diagnosis/treatment wording, no
obvious identifiers), and renders a deterministic Markdown view that exactly reflects the
JSON. Research-only. Not diagnostic. Not for medical decision-making.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from ..constants import SEVERITIES, split_condition

SCHEMA_VERSION = "finding_graph_v5"
DISCLAIMER = (
    "research-only · not diagnostic · not clinically validated · "
    "no medical decision-making · no treatment recommendation"
)

VIEW_ROUTE = {
    "spinal_canal_stenosis": "sagittal_t2",
    "neural_foraminal_narrowing": "sagittal_t1",
    "subarticular_stenosis": "axial_t2",
}
CROP_CENTER_SOURCE = {
    "sagittal_t2": "disc_localizer (auto)",
    "sagittal_t1": "foraminal_localizer best-slice (auto)",
    "axial_t2": "axial_level_scorer + fixed paramedian offset (auto)",
}
ALLOWED_REVIEW_REASONS = frozenset(
    {
        "low_confidence",
        "high_entropy",
        "model_disagreement",
        "localizer_uncertainty",
        "axial_level_uncertainty",
        "view_missing",
        "morphology_disagreement",
        "near_severe_threshold",
        # v5 evidence-stability reasons (see evaluation.evidence_stability)
        "evidence_unstable",
        "route_unstable",
        "axial_candidate_disagreement",
        "foraminal_slice_disagreement",
    }
)
ALLOWED_STABILITY_GRADES = frozenset({"stable", "mildly_unstable", "unstable"})
ALLOWED_ROUTE_QUALITY = frozenset({"good", "fair", "weak"})
ALLOWED_PROVENANCE = frozenset({"auto", "oracle", "blocked"})
# Disclaimer/negated phrases that are explicitly allowed (stripped before the scan, longest
# first so the longer form is removed before its prefix).
ALLOWED_PHRASES = (
    "not for medical decision-making",
    "no medical decision-making",
    "no treatment recommendation",
    "not clinically validated",
    "research findings, not a diagnosis",
    "non_diagnostic_research_finding_graph",
    "non-diagnostic research finding graph",
    "non_diagnostic",
    "non-diagnostic",
    "not a diagnosis",
    "not diagnostic",
    "does not diagnose",
    "not diagnose",
    "no treatment",
)
# Positive-claim roots that must never remain after the allowed phrases are removed.
# (Roots only — avoid "fda"/"cure" which can collide with hex case_id / common words.)
FORBIDDEN_ROOTS = ("diagnos", "treatment", "prescrib", "doctor replacement", "fda-clear")


def case_id(study_id: str) -> str:
    """Stable anonymized case id (study_ids are already de-identified; hash anyway)."""
    return "case_" + hashlib.sha1(str(study_id).encode()).hexdigest()[:10]


def _uncertainty_flag(conf: float, high: float = 0.85, moderate: float = 0.60) -> str:
    if conf >= high:
        return "high_confidence"
    if conf >= moderate:
        return "moderate_confidence"
    return "review_required"


def _entropy_norm(probs: list[float]) -> float:
    eps = 1e-12
    h = -sum(p * math.log(p + eps) for p in probs)
    return h / math.log(len(probs))


def build_finding(
    condition: str,
    level: str,
    probs: list[float],
    *,
    side: str | None = None,
    crop_provenance: str = "auto",
    reference_label: str | None = None,
    model_disagreement: bool = False,
    localizer_confidence: float | None = None,
    axial_level_score: float | None = None,
    evidence_stability: dict[str, Any] | None = None,
    route_quality_grade: str | None = None,
    extra_review_reasons: tuple[str, ...] = (),
    conf_threshold: float = 0.60,
    entropy_threshold: float = 0.85,
    near_severe: float = 0.20,
    axial_level_uncertain_below: float = 0.50,
) -> dict[str, Any]:
    """Build one finding dict from a real 3-class probability vector. No invented values.

    ``evidence_stability`` (optional) is the dict from
    ``evaluation.evidence_stability.stability_stats`` augmented with a ``grade`` key;
    when its grade is not ``stable`` the route-aware reasons (``evidence_unstable`` +
    ``axial_candidate_disagreement`` / ``foraminal_slice_disagreement`` / ``route_unstable``)
    are added to ``review_reasons``. ``route_quality_grade`` in {good, fair, weak}.
    """
    # Round first, then take argmax, so severity_estimate is always the argmax of the
    # *stored* probabilities (avoids a rounding flip on near-tie cases vs the validator).
    p = [round(float(x), 4) for x in probs]
    pred = int(max(range(len(p)), key=lambda i: p[i]))
    conf = float(p[pred])
    base, parsed_side = split_condition(condition)
    side = side or parsed_side
    route = VIEW_ROUTE[base]

    reasons: list[str] = []
    if conf < conf_threshold:
        reasons.append("low_confidence")
    if _entropy_norm(p) > entropy_threshold:
        reasons.append("high_entropy")
    if model_disagreement:
        reasons.append("model_disagreement")
    if pred != 2 and p[2] >= near_severe:
        reasons.append("near_severe_threshold")
    if (
        route == "axial_t2"
        and axial_level_score is not None
        and axial_level_score < axial_level_uncertain_below
    ):
        reasons.append("axial_level_uncertainty")
    # v5: evidence-stability-driven review reasons (route-aware), inlined to keep the
    # schema layer free of an evaluation-module import. Only the strong `unstable` grade
    # raises review (mildly_unstable informs route_quality but does not flood review).
    stab_grade = (evidence_stability or {}).get("grade")
    if stab_grade == "unstable":
        reasons.append("evidence_unstable")
        if route == "axial_t2":
            tail = "axial_candidate_disagreement"
        elif route == "sagittal_t1":
            tail = "foraminal_slice_disagreement"
        else:
            tail = "route_unstable"
        reasons.append(tail)
    for r in extra_review_reasons:
        if r not in reasons:
            reasons.append(r)

    finding = {
        "condition": condition,
        "base_condition": base,
        "level": level,
        "side": side,
        "view_route": route,
        "crop_provenance": crop_provenance,
        "severity_estimate": SEVERITIES[pred],
        "probabilities": {
            "P(normal_mild)": p[0],
            "P(moderate)": p[1],
            "P(severe)": p[2],
        },
        "calibrated_confidence": round(conf, 4),
        "uncertainty_flag": _uncertainty_flag(conf),
        "review_required": bool(reasons),
        "review_reasons": reasons,
        "localizer": {
            "route": route,
            "confidence": round(float(localizer_confidence), 4)
            if localizer_confidence is not None
            else None,
            "axial_level_scorer_score": round(float(axial_level_score), 4)
            if axial_level_score is not None
            else None,
        },
        "evidence": {
            "view_used": route,
            "crop_center_source": CROP_CENTER_SOURCE[route],
            "notes": "auto inference — no ground-truth coordinates read at inference",
        },
        # v5 evidence-aware reliability signals
        "evidence_stability": _stability_block(evidence_stability),
        "route_quality": route_quality_grade,
        # held-out research target, shown for transparency; NOT a model input or output
        "reference_label": reference_label,
    }
    return finding


def _stability_block(stab: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact, JSON-safe subset of stability stats for the finding graph."""
    if not stab:
        return None
    keys = (
        "grade",
        "instability",
        "p_severe_range",
        "p_severe_std",
        "severity_flip_rate",
        "agreement_rate",
        "k_perturb",
    )
    out = {k: stab[k] for k in keys if k in stab}
    for k in (
        "instability",
        "p_severe_range",
        "p_severe_std",
        "severity_flip_rate",
        "agreement_rate",
    ):
        if k in out and out[k] is not None:
            out[k] = round(float(out[k]), 4)
    return out


def build_study_graph(
    study_id: str,
    *,
    split: str,
    findings: list[dict[str, Any]],
    model_version: str,
    route_version: str = SCHEMA_VERSION,
    generated_at: str = "",
    blocked_findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the full study-level finding graph + summary."""
    blocked = blocked_findings or []
    p_sev = [f["probabilities"]["P(severe)"] for f in findings]
    supported = sorted({f["condition"] for f in findings})
    review = [
        {
            "condition": f["condition"],
            "level": f["level"],
            "side": f["side"],
            "reasons": f["review_reasons"],
        }
        for f in findings
        if f["review_required"]
    ]
    summary = {
        "max_p_severe": round(max(p_sev), 4) if p_sev else 0.0,
        "n_findings": len(findings),
        "n_severe_estimates": sum(1 for f in findings if f["severity_estimate"] == "severe"),
        "n_review_required": len(review),
        "n_high_confidence": sum(1 for f in findings if f["uncertainty_flag"] == "high_confidence"),
        "n_low_confidence": sum(1 for f in findings if f["uncertainty_flag"] == "review_required"),
        "n_unstable": sum(
            1 for f in findings if (f.get("evidence_stability") or {}).get("grade") == "unstable"
        ),
        "n_weak_route": sum(1 for f in findings if f.get("route_quality") == "weak"),
        "n_blocked": len(blocked),
        "findings_requiring_review": review,
        "warnings": ["severity values are research findings, not a diagnosis"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "non_diagnostic_research_finding_graph",
        "case_id": case_id(study_id),
        "split": split,
        "research_only": True,
        "disclaimer": DISCLAIMER,
        "generated_at": generated_at,
        "model_version": model_version,
        "route_version": route_version,
        "supported_findings": supported,
        "unsupported_or_oracle_only_findings": sorted({b["condition"] for b in blocked}),
        "findings": findings,
        "blocked_findings": blocked,
        "study_summary": summary,
    }


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def validate_finding_graph(graph: dict[str, Any]) -> None:
    """Raise ValueError on any schema / safety violation."""
    if graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad schema_version: {graph.get('schema_version')}")
    if not str(graph.get("case_id", "")).startswith("case_"):
        raise ValueError("case_id must be an anonymized hash (case_*)")
    for f in graph["findings"]:
        pr = f["probabilities"]
        s = pr["P(normal_mild)"] + pr["P(moderate)"] + pr["P(severe)"]
        if abs(s - 1.0) > 0.02:
            raise ValueError(f"probabilities sum {s} != 1 for {f['condition']} {f['level']}")
        if f["crop_provenance"] not in ALLOWED_PROVENANCE:
            raise ValueError(f"bad provenance {f['crop_provenance']}")
        if f["severity_estimate"] not in SEVERITIES:
            raise ValueError(f"bad severity {f['severity_estimate']}")
        for r in f["review_reasons"]:
            if r not in ALLOWED_REVIEW_REASONS:
                raise ValueError(f"unknown review reason {r}")
        # review_required must be consistent with the presence of reasons
        if f["review_required"] != bool(f["review_reasons"]):
            raise ValueError("review_required inconsistent with review_reasons")
        # v5 evidence-aware fields (optional, but constrained when present)
        stab = f.get("evidence_stability")
        if stab is not None and stab.get("grade") not in ALLOWED_STABILITY_GRADES:
            raise ValueError(f"bad evidence_stability grade {stab.get('grade')}")
        rq = f.get("route_quality")
        if rq is not None and rq not in ALLOWED_ROUTE_QUALITY:
            raise ValueError(f"bad route_quality {rq}")
        # severity must equal the argmax of the stated probabilities
        ordered = [pr["P(normal_mild)"], pr["P(moderate)"], pr["P(severe)"]]
        if SEVERITIES[int(max(range(3), key=lambda i: ordered[i]))] != f["severity_estimate"]:
            raise ValueError("severity_estimate is not the argmax of probabilities")
    # safety: no diagnosis/treatment wording anywhere. Strip the explicitly-allowed
    # disclaimer/negated phrases first, then scan for positive-claim roots.
    blob = " ".join(_iter_strings(graph)).lower()
    for phrase in ALLOWED_PHRASES:
        blob = blob.replace(phrase, " ")
    for root in FORBIDDEN_ROOTS:
        if root in blob:
            raise ValueError(f"forbidden wording present (root {root!r})")


_LEVEL_ORDER = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")


def render_markdown(graph: dict[str, Any]) -> str:
    """Deterministic Markdown rendering that exactly reflects the JSON (no extra claims)."""
    s = graph["study_summary"]
    lines = [
        f"# Research finding graph — {graph['case_id']} (NON-DIAGNOSTIC)",
        "",
        f"> {graph['disclaimer']}",
        "",
        f"- **schema** `{graph['schema_version']}` · **model** `{graph['model_version']}` · "
        f"**split** `{graph['split']}` · **report** `{graph['report_type']}`",
        f"- **supported findings (auto):** {', '.join(graph['supported_findings'])}",
        f"- **study summary:** max P(severe) **{s['max_p_severe']}**, "
        f"{s['n_severe_estimates']} severe estimate(s), {s['n_review_required']} need review, "
        f"{s['n_high_confidence']} high-confidence, {s.get('n_unstable', 0)} unstable, "
        f"{s.get('n_weak_route', 0)} weak-route, {s['n_blocked']} blocked",
        "",
        "| condition | side | level | view route | severity estimate | P(severe) | "
        "confidence | flag | stability | route_q | review | ref |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in sorted(
        graph["findings"],
        key=lambda x: (
            x["condition"],
            _LEVEL_ORDER.index(x["level"]) if x["level"] in _LEVEL_ORDER else 9,
        ),
    ):
        stab = (f.get("evidence_stability") or {}).get("grade", "-")
        lines.append(
            f"| {f['condition']} | {f['side'] or '-'} | {f['level']} | {f['view_route']} | "
            f"{f['severity_estimate']} | {f['probabilities']['P(severe)']} | "
            f"{f['calibrated_confidence']} | {f['uncertainty_flag']} | {stab} | "
            f"{f.get('route_quality') or '-'} | "
            f"{','.join(f['review_reasons']) or '-'} | {f['reference_label'] or '-'} |"
        )
    if graph["blocked_findings"]:
        lines += [
            "",
            "**blocked / oracle-only:** "
            + ", ".join(b["condition"] for b in graph["blocked_findings"]),
        ]
    lines += [
        "",
        "_Provenance: auto = localizer/scorer-placed crop, no ground-truth coordinates at "
        "inference. `reference_label` is a held-out research target shown for transparency, "
        "not a model input/output. Severity values are research findings, not a diagnosis._",
    ]
    return "\n".join(lines) + "\n"
