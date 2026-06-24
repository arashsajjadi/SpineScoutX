"""Real case-viewer layer over a finding graph (v1.2).

A finding graph says what the model output. The **case viewer** adds the missing
piece for a reader: it places each prediction next to its **held-out reference
label** and derives a **correctness** verdict (by code, never hardcoded), assigns a
**case category**, and summarises where the study is right / uncertain / wrong.

Strict rules (enforced by ``validate_case_viewer``):
- the held-out reference label is **reference only**, never a model input; it is
  clearly tagged ``held_out_reference_label`` and lives in its own column;
- correctness is **derived** from ``severity_estimate`` vs the reference label;
- no patient identifiers / raw DICOM paths; no diagnosis/treatment wording.

Research-only. Not diagnostic. Not for medical decision-making.
"""

from __future__ import annotations

from typing import Any

from ..constants import SEVERITIES
from . import finding_graph_schema as fg

CASE_VIEWER_VERSION = "case_viewer_v1"

# GT-derived correctness (severe-first). `review_status` is the orthogonal flag.
CORRECTNESS = frozenset(
    {
        "severe_correct",  # ref severe & pred severe — the key success
        "exact_correct",  # pred == ref (non-severe)
        "non_severe_mismatch",  # pred != ref, neither side severe
        "severe_false_negative",  # ref severe, pred not severe — worst
        "severe_false_positive",  # ref not severe, pred severe
        "no_reference",  # no held-out label available
    }
)
REVIEW_STATUS = frozenset({"uncertain_review", "auto_accepted"})
CASE_CATEGORIES = frozenset(
    {
        "hard_right_foraminal",
        "false_negative",
        "false_positive",
        "unstable_review",
        "model_disagreement",
        "axial_uncertain",
        "correct_severe",
        "mostly_normal",
        "routine",
    }
)
# worst-first ordering for "worst_failure_mode" + category precedence
_SEVERITY_ERR_ORDER = ("severe_false_negative", "severe_false_positive", "non_severe_mismatch")


def finding_correctness(severity_estimate: str, reference_label: str | None) -> str:
    """Derive the severe-first correctness verdict (pure, GT-relative)."""
    if reference_label is None or reference_label not in SEVERITIES:
        return "no_reference"
    pred_sev = severity_estimate == "severe"
    ref_sev = reference_label == "severe"
    if ref_sev and pred_sev:
        return "severe_correct"
    if ref_sev and not pred_sev:
        return "severe_false_negative"
    if pred_sev and not ref_sev:
        return "severe_false_positive"
    if severity_estimate == reference_label:
        return "exact_correct"
    return "non_severe_mismatch"


def _is_correct(c: str) -> bool:
    return c in ("severe_correct", "exact_correct")


def _viewer_finding(
    f: dict[str, Any], instability_type: str | None, retrieval: dict[str, Any] | None
) -> dict[str, Any]:
    ref = f.get("reference_label")
    correctness = finding_correctness(f["severity_estimate"], ref)
    stab = f.get("evidence_stability") or {}
    sim = None
    if retrieval:
        sim = {
            "k": int(retrieval.get("k", 0)),
            "severity_distribution": retrieval.get("severity_distribution", {}),
            "majority_severity": retrieval.get("majority_severity"),
            "retrieval_warning": (
                "similar research cases (nearest grader-embedding neighbours), "
                "NOT a clinical reference; does not change the prediction"
            ),
        }
    return {
        "condition": f["condition"],
        "level": f["level"],
        "side": f.get("side"),
        "view_route": f["view_route"],
        "crop_provenance": f["crop_provenance"],
        "severity_estimate": f["severity_estimate"],
        "probabilities": f["probabilities"],
        "calibrated_confidence": f["calibrated_confidence"],
        "evidence_stability_grade": stab.get("grade"),
        "instability_type": instability_type,
        "route_quality": f.get("route_quality"),
        "review_required": f["review_required"],
        "review_reasons": f["review_reasons"],
        "held_out_reference_label": ref,  # reference only — NOT a model input
        "correctness": correctness,
        "review_status": "uncertain_review" if f["review_required"] else "auto_accepted",
        "similar_research_cases": sim,  # explanation-only; never changes the prediction
    }


def _case_category(vfindings: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Single most-salient category. Severe errors first; a clean correct-severe study is
    ``correct_severe`` even if some finding is unstable (instability categories only when
    there is no severe error and no severe-correct headline)."""
    corr = [f["correctness"] for f in vfindings]
    # the right-foraminal "hard case" is specifically a right-foraminal SEVERE miss
    has_rf_fn = any(
        f["condition"] == "right_neural_foraminal_narrowing"
        and f["correctness"] == "severe_false_negative"
        for f in vfindings
    )
    has_unstable = any(f["evidence_stability_grade"] == "unstable" for f in vfindings)
    has_disagree = any("model_disagreement" in f["review_reasons"] for f in vfindings)
    has_axial_unc = any(
        f["view_route"] == "axial_t2"
        and (
            "axial_level_uncertainty" in f["review_reasons"]
            or "axial_candidate_disagreement" in f["review_reasons"]
        )
        for f in vfindings
    )
    if has_rf_fn:
        return "hard_right_foraminal"
    if "severe_false_negative" in corr:
        return "false_negative"
    if "severe_false_positive" in corr:
        return "false_positive"
    if "severe_correct" in corr:
        return "correct_severe"  # ranked above instability: a caught severe is the headline
    if has_disagree:
        return "model_disagreement"
    if has_axial_unc:
        return "axial_uncertain"
    if has_unstable:
        return "unstable_review"
    if summary["highest_p_severe"] < 0.20 and summary["n_severe_estimates"] == 0:
        return "mostly_normal"
    return "routine"


def build_case_viewer(
    graph: dict[str, Any],
    *,
    instability_types: dict[tuple[str, str, str], str] | None = None,
    retrieval: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a case-viewer object from a validated finding graph.

    ``instability_types`` (optional) maps ``(condition, level, side)`` to an
    instability type from evidence-intelligence v2 (crop/slice/axial_candidate/route
    sensitive). ``retrieval`` (optional) maps the same key to a similar-research-cases
    summary (explanation only — never changes the prediction). Correctness is derived here
    from prediction vs held-out reference.
    """
    itypes = instability_types or {}
    retr = retrieval or {}
    vfindings = [
        _viewer_finding(
            f,
            itypes.get((f["condition"], f["level"], f.get("side") or "")),
            retr.get((f["condition"], f["level"], f.get("side") or "")),
        )
        for f in graph["findings"]
    ]
    p_sev = [f["probabilities"]["P(severe)"] for f in graph["findings"]]
    n_exact = sum(1 for f in vfindings if _is_correct(f["correctness"]))
    n_sev_err = sum(
        1
        for f in vfindings
        if f["correctness"] in ("severe_false_negative", "severe_false_positive")
    )
    worst = "none"
    for mode in _SEVERITY_ERR_ORDER:
        if any(f["correctness"] == mode for f in vfindings):
            worst = mode
            break
    rq = [f["route_quality"] for f in vfindings if f["route_quality"]]
    summary = {
        "highest_p_severe": round(max(p_sev), 4) if p_sev else 0.0,
        "n_findings": len(vfindings),
        "n_severe_estimates": sum(1 for f in vfindings if f["severity_estimate"] == "severe"),
        "n_review_required": sum(1 for f in vfindings if f["review_required"]),
        "n_exact_correct": n_exact,
        "n_severe_errors": n_sev_err,
        "n_with_reference": sum(1 for f in vfindings if f["correctness"] != "no_reference"),
        "worst_failure_mode": worst,
        "route_quality_summary": {q: rq.count(q) for q in ("good", "fair", "weak")},
    }
    summary["case_category"] = _case_category(vfindings, summary)
    return {
        "viewer_version": CASE_VIEWER_VERSION,
        "case_id": graph["case_id"],
        "split": graph["split"],
        "model_version": graph["model_version"],
        "route_version": graph["route_version"],
        "disclaimer": graph["disclaimer"],
        "case_category": summary["case_category"],
        "findings": vfindings,
        "study_summary": summary,
        "reference_note": (
            "held_out_reference_label is a held-out research target shown for transparency; "
            "it is NOT a model input and was NOT used at auto inference. correctness is derived."
        ),
    }


_LEVEL_ORDER = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")
_CORRECT_MARK = {
    "severe_correct": "✓ severe-correct",
    "exact_correct": "✓ exact",
    "non_severe_mismatch": "~ non-severe miss",
    "severe_false_negative": "✗ severe FN",
    "severe_false_positive": "✗ severe FP",
    "no_reference": "– no ref",
}


def render_case_markdown(cv: dict[str, Any]) -> str:
    """Deterministic Markdown that reflects the case-viewer JSON exactly."""
    s = cv["study_summary"]
    rq = s["route_quality_summary"]
    lines = [
        f"# Real case viewer — {cv['case_id']} (NON-DIAGNOSTIC)",
        "",
        f"> {cv['disclaimer']}",
        "",
        f"- **category:** `{cv['case_category']}` · **model** `{cv['model_version']}` · "
        f"**split** `{cv['split']}`",
        f"- **summary:** highest P(severe) **{s['highest_p_severe']}** · "
        f"{s['n_exact_correct']}/{s['n_with_reference']} correct vs reference · "
        f"{s['n_severe_errors']} severe error(s) · {s['n_review_required']} review · "
        f"worst: `{s['worst_failure_mode']}` · route quality "
        f"good {rq['good']}/fair {rq['fair']}/weak {rq['weak']}",
        "",
        "| condition | side | level | route | severity est. | P(severe) | conf | stability | "
        "route_q | review | **held-out ref** | **correctness** |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in sorted(
        cv["findings"],
        key=lambda x: (
            x["condition"],
            _LEVEL_ORDER.index(x["level"]) if x["level"] in _LEVEL_ORDER else 9,
        ),
    ):
        lines.append(
            f"| {f['condition']} | {f['side'] or '-'} | {f['level']} | {f['view_route']} | "
            f"{f['severity_estimate']} | {f['probabilities']['P(severe)']} | "
            f"{f['calibrated_confidence']} | {f['evidence_stability_grade'] or '-'} | "
            f"{f['route_quality'] or '-'} | {','.join(f['review_reasons']) or '-'} | "
            f"{f['held_out_reference_label'] or '-'} | {_CORRECT_MARK[f['correctness']]} |"
        )
    sims = [f for f in cv["findings"] if f.get("similar_research_cases")]
    if sims:
        lines += ["", "**Similar research cases** (explanation only — never changes a prediction):"]
        for f in sims[:4]:
            sc = f["similar_research_cases"]
            dist = ", ".join(f"{k}:{v}" for k, v in sc["severity_distribution"].items())
            lines.append(
                f"- {f['condition']} {f['level']}: top-{sc['k']} → majority "
                f"`{sc['majority_severity']}` ({dist})"
            )
    lines += [
        "",
        f"_{cv['reference_note']}_",
    ]
    return "\n".join(lines) + "\n"


def validate_case_viewer(cv: dict[str, Any]) -> None:
    """Raise ValueError on schema / safety / leakage violations."""
    if cv.get("viewer_version") != CASE_VIEWER_VERSION:
        raise ValueError(f"bad viewer_version {cv.get('viewer_version')}")
    if not str(cv.get("case_id", "")).startswith("case_"):
        raise ValueError("case_id must be an anonymized hash (case_*)")
    if cv["case_category"] not in CASE_CATEGORIES:
        raise ValueError(f"bad case_category {cv['case_category']}")
    for f in cv["findings"]:
        pr = f["probabilities"]
        s = pr["P(normal_mild)"] + pr["P(moderate)"] + pr["P(severe)"]
        if abs(s - 1.0) > 0.02:
            raise ValueError(f"probabilities sum {s} != 1")
        if f["correctness"] not in CORRECTNESS:
            raise ValueError(f"bad correctness {f['correctness']}")
        if f["review_status"] not in REVIEW_STATUS:
            raise ValueError(f"bad review_status {f['review_status']}")
        # correctness must equal the code-derived value (no hardcoding)
        if f["correctness"] != finding_correctness(
            f["severity_estimate"], f["held_out_reference_label"]
        ):
            raise ValueError("correctness is not derived from prediction vs reference")
        # the reference label must be present as its own clearly-named field
        if "held_out_reference_label" not in f:
            raise ValueError("missing held_out_reference_label")
    # no raw dicom path leakage anywhere
    blob = " ".join(fg._iter_strings(cv))
    if ".dcm" in blob.lower() or "/data/raw/" in blob.lower():
        raise ValueError("raw DICOM path / data-root leak in case viewer")
    # no diagnosis/treatment wording (reuse the finding-graph guard)
    low = blob.lower()
    for phrase in fg.ALLOWED_PHRASES:
        low = low.replace(phrase, " ")
    for root in fg.FORBIDDEN_ROOTS:
        if root in low:
            raise ValueError(f"forbidden wording present (root {root!r})")
