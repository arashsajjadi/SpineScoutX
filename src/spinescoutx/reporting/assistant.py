"""Research report assistant: a richer, still-deterministic finding-graph report.

The structured finding graph remains authoritative. This adds a readable summary,
severe-finding and uncertainty highlights, evidence-region validity, optional
cross-model (E0 vs E2) disagreement + severe false-negative warnings, and an
optional fail-closed LLM wording section.

Research-only. Not diagnostic. Not for medical decision-making.
"""

from __future__ import annotations

from typing import Any

_DISCLAIMER = (
    "> **Research-only — not diagnostic — not clinically validated — not for "
    "medical decision-making. No treatment recommendation.**"
)


def _by_key(findings: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(f["level"], f["condition"]): f for f in findings}


def build_assistant_markdown(
    graph: dict[str, Any],
    *,
    baseline_graph: dict[str, Any] | None = None,
    baseline_name: str = "E0",
    model_name: str = "E2",
    llm_text: str | None = None,
) -> str:
    """Render a research-assistant Markdown report from a finding-graph dict."""
    findings = graph.get("findings", [])
    lines: list[str] = [
        f"# SpineScoutX research assistant — study {graph.get('study_id')}",
        "",
        _DISCLAIMER,
        "",
        f"- dataset: `{graph.get('dataset_source')}` · model: `{graph.get('model_version')}` "
        f"· run: `{graph.get('run_id')}` · findings: {len(findings)}",
        "",
        "## Severity grade summary",
    ]
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["grade"]] = counts.get(f["grade"], 0) + 1
    lines.append(
        "  ".join(f"**{g}**: {counts.get(g, 0)}" for g in ("normal_mild", "moderate", "severe"))
    )

    severe = [f for f in findings if f["grade"] == "severe"]
    review = [f for f in findings if f.get("uncertainty_flag") == "review_required"]
    lines += ["", "## Highlighted research findings (non-diagnostic)"]
    if severe:
        lines.append("**Severe-grade findings (research estimate only):**")
        for f in severe:
            side = f" ({f['side']})" if f.get("side") else ""
            lines.append(
                f"- {f['level']} {f['condition']}{side}: grade=severe, "
                f"calibrated_confidence={f.get('calibrated_confidence')}, "
                f"flag={f.get('uncertainty_flag')}, evidence_region="
                f"{f.get('evidence_region')} ({f.get('evidence_region_source')})"
            )
    else:
        lines.append("- No severe-grade findings in this study (research estimate).")
    if review:
        lines.append(f"- {len(review)} finding(s) flagged `review_required` (low confidence).")

    # Evidence-region validity rollup.
    valid = sum(1 for f in findings if f.get("evidence_region_source") == "anatomy")
    approx = sum(1 for f in findings if f.get("evidence_region_source") == "approximate")
    lines += [
        "",
        "## Evidence-region validity",
        f"- anatomy (validated region, e.g. spinal canal): {valid}",
        f"- approximate (foraminal/lateral-recess; SPIDER has no such labels): {approx}",
    ]

    # Optional cross-model disagreement + severe FN warnings.
    if baseline_graph is not None:
        base = _by_key(baseline_graph.get("findings", []))
        cur = _by_key(findings)
        disagree = []
        severe_warn = []
        for key, f in cur.items():
            b = base.get(key)
            if b is None:
                continue
            if b["grade"] != f["grade"]:
                disagree.append((key, b["grade"], f["grade"]))
            # severe FN warning: either model says severe but the other does not.
            if "severe" in (b["grade"], f["grade"]) and b["grade"] != f["grade"]:
                severe_warn.append((key, b["grade"], f["grade"]))
        lines += [
            "",
            f"## Cross-model check ({baseline_name} vs {model_name})",
            f"- disagreements: {len(disagree)} of {len(cur)} (level,condition) pairs",
        ]
        if severe_warn:
            lines.append(
                "- ⚠ severe-grade disagreements (treat as research uncertainty, not risk):"
            )
            for (lvl, cond), bg, cg in severe_warn[:10]:
                lines.append(f"  - {lvl} {cond}: {baseline_name}={bg} vs {model_name}={cg}")

    if llm_text:
        lines += ["", "## LLM-polished wording (non-authoritative)", "", llm_text]

    lines += [
        "",
        "## Limitations",
        *[f"- {lim}" for lim in graph.get("limitations", [])],
        "",
        "_No medical advice. No treatment recommendation. The structured finding graph "
        "above is the authoritative output; any prose is a convenience rendering._",
    ]
    return "\n".join(lines) + "\n"
