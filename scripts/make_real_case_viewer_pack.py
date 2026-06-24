#!/usr/bin/env python3
"""Generate the REAL case-viewer pack: input/evidence route -> model prediction ->
held-out reference label -> correctness -> safety, as WIDE readable cards.

Reuses the showcase finding-graph builder (deployable graders on locked-test auto, schema
v5 with evidence_stability + route_quality), wraps each study in the case-viewer layer
(correctness derived from prediction vs held-out reference), joins the evidence-intelligence
v2 instability *type*, selects curated cases by category, and renders a large 1800x1000
sectioned card per case (Summary / Evidence route / Prediction vs held-out reference /
Safety & review / Correctness note) plus a legend. NO DICOM pixels, no identifiers
(hashed case_*); the held-out reference is shown for transparency only, never an input.

Research-only. Not diagnostic. Reproduce: `python scripts/make_real_case_viewer_pack.py`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from spinescoutx.reporting import case_viewer as cvm
from spinescoutx.training.optim import select_device

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
PACK = ROOT / "outputs/real/case_viewer_pack"
ASSETS = ROOT / "docs/assets/cases"
DOC = ROOT / "docs/run_logs/real_case_viewer.md"
V2_RECORDS = ROOT / "outputs/real/evidence_intel_v2_records.parquet"
RETR_RECORDS = ROOT / "outputs/real/similar_case_retrieval_records.parquet"

_spec = importlib.util.spec_from_file_location(
    "show", ROOT / "scripts/make_model_output_showcase.py"
)
_show = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_show)

SEV_COLOR = {"normal_mild": "#2e7d32", "moderate": "#f9a825", "severe": "#c62828"}
ROUTE_COLOR = {"sagittal_t2": "#2a9d8f", "sagittal_t1": "#457b9d", "axial_t2": "#9d4edd"}
CORRECT_STYLE = {
    "severe_correct": ("#2e7d32", "✓ SEVERE CORRECT"),
    "exact_correct": ("#2e7d32", "✓ EXACT"),
    "non_severe_mismatch": ("#f9a825", "~ NON-SEVERE MISS"),
    "severe_false_negative": ("#c62828", "✗ SEVERE FALSE NEG"),
    "severe_false_positive": ("#e65100", "✗ SEVERE FALSE POS"),
    "no_reference": ("#9e9e9e", "– NO REF"),
}


def _load_instability_types():
    if not V2_RECORDS.exists():
        return {}
    import pandas as pd

    df = pd.read_parquet(V2_RECORDS)
    out: dict[tuple[str, str, str], str] = {}
    for r in df.itertuples():
        out[(r.condition, str(r.study_id), r.level, str(r.side or ""))] = r.instability_type
    return out


def _load_retrieval():
    if not RETR_RECORDS.exists():
        return {}
    import pandas as pd

    df = pd.read_parquet(RETR_RECORDS)
    out: dict[tuple[str, str, str, str], dict] = {}
    for r in df.itertuples():
        out[(r.condition, str(r.study_id), r.level, str(r.side or ""))] = {
            "k": int(r.k),
            "severity_distribution": dict(r.severity_distribution),
            "majority_severity": r.majority_severity,
        }
    return out


def _build_viewers(device):
    graphs = _show._build_graphs(device)
    itypes_flat = _load_instability_types()
    retr_flat = _load_retrieval()
    studies_with_types = {k[1] for k in itypes_flat}
    viewers = {}
    for study, g in graphs.items():
        per = {(c, lv, sd): t for (c, st, lv, sd), t in itypes_flat.items() if st == study}
        per_r = {(c, lv, sd): d for (c, st, lv, sd), d in retr_flat.items() if st == study}
        viewers[study] = cvm.build_case_viewer(g, instability_types=per, retrieval=per_r)
        cvm.validate_case_viewer(viewers[study])
    return viewers, studies_with_types


def _has(
    v, *, cond=None, correctness=None, category=None, route=None, unstable=False, review=False
):
    if category is not None and v["case_category"] != category:
        return False
    for f in v["findings"]:
        if cond and cond not in f["condition"]:
            continue
        if correctness and f["correctness"] != correctness:
            continue
        if route and f["view_route"] != route:
            continue
        if unstable and f["evidence_stability_grade"] != "unstable":
            continue
        if review and not f["review_required"]:
            continue
        return True
    return False


def _select(viewers, prefer):
    """Pick one viewer per category -> {asset_name: viewer}. Prefer studies with types."""
    chosen, used = {}, set()
    ordered = sorted(
        viewers.items(),
        key=lambda kv: (kv[0] not in prefer, -kv[1]["study_summary"]["highest_p_severe"]),
    )

    def pick(name, pred):
        for study, v in ordered:
            if study in used:
                continue
            if pred(v):
                used.add(study)
                chosen[name] = v
                return

    def clean(v):  # no severe errors -> a clean positive exemplar
        return v["study_summary"]["n_severe_errors"] == 0

    pick(
        "case_canal_correct_severe",
        lambda v: clean(v) and _has(v, cond="spinal_canal", correctness="severe_correct"),
    )
    pick(
        "case_left_foraminal_correct_severe",
        lambda v: clean(v) and _has(v, cond="left_neural_foraminal", correctness="severe_correct"),
    )
    # right-foraminal HARD case: specifically a right-foraminal severe miss (the weakest route)
    pick(
        "case_right_foraminal_hard",
        lambda v: _has(v, cond="right_neural_foraminal", correctness="severe_false_negative"),
    )
    pick(
        "case_subarticular_correct",
        lambda v: clean(v) and _has(v, cond="subarticular", correctness="severe_correct"),
    )
    pick("case_axial_unstable", lambda v: _has(v, route="axial_t2", unstable=True))
    pick("case_model_disagreement", lambda v: v["case_category"] == "model_disagreement")
    # review_required catching a severe false negative (the safety win)
    pick(
        "case_review_required",
        lambda v: any(
            f["review_required"] and f["correctness"] == "severe_false_negative"
            for f in v["findings"]
        ),
    )
    pick(
        "case_mostly_normal",
        lambda v: (
            v["case_category"] == "mostly_normal" and v["study_summary"]["n_review_required"] == 0
        ),
    )
    return chosen


# --------------------------------------------------------------------------- #
# wide readable card
# --------------------------------------------------------------------------- #
def _relevant_findings(v, k=7):
    """Severe / wrong / review findings first, then fill — at most k rows."""
    order = {
        "severe_false_negative": 0,
        "severe_false_positive": 1,
        "severe_correct": 2,
        "non_severe_mismatch": 3,
        "exact_correct": 4,
        "no_reference": 5,
    }
    fs = sorted(
        v["findings"],
        key=lambda f: (
            order.get(f["correctness"], 9),
            -f["probabilities"]["P(severe)"],
        ),
    )
    return fs[:k]


def render_card(v, path, title):
    fig = plt.figure(figsize=(18, 10), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    s = v["study_summary"]

    # header
    ax.add_patch(Rectangle((0, 92), 100, 8, fc="#0d3b66", ec="none"))
    ax.text(
        1.5,
        96,
        f"Real case viewer — {title}",
        fontsize=22,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(98.5, 96, v["case_id"], fontsize=15, color="#cfe3ff", va="center", ha="right")
    ax.text(
        1.5,
        93,
        "RESEARCH-ONLY · NOT DIAGNOSTIC · auto inference (no ground-truth coordinates)",
        fontsize=11,
        color="#cfe3ff",
        va="center",
    )

    # 1) summary band
    cat = v["case_category"]
    ax.text(1.5, 88.5, "1 · CASE SUMMARY", fontsize=14, fontweight="bold", color="#0d3b66")
    rq = s["route_quality_summary"]
    chips = [
        (f"category: {cat}", "#0d3b66"),
        (f"highest P(severe): {s['highest_p_severe']:.2f}", "#c62828"),
        (f"correct vs reference: {s['n_exact_correct']}/{s['n_with_reference']}", "#2e7d32"),
        (
            f"severe errors: {s['n_severe_errors']}",
            "#c62828" if s["n_severe_errors"] else "#2e7d32",
        ),
        (f"review_required: {s['n_review_required']}", "#e65100"),
        (f"route quality g/f/w: {rq['good']}/{rq['fair']}/{rq['weak']}", "#457b9d"),
    ]
    x = 1.5
    for txt, col in chips:
        w = 0.62 * len(txt) + 2.0
        ax.add_patch(
            FancyBboxPatch(
                (x, 84.3), w, 3.0, boxstyle="round,pad=0.2", fc="#eef3f8", ec=col, lw=1.5
            )
        )
        ax.text(
            x + w / 2,
            85.8,
            txt,
            fontsize=11.5,
            color=col,
            ha="center",
            va="center",
            fontweight="bold",
        )
        x += w + 1.2

    # 2) evidence route
    ax.text(
        1.5,
        81.5,
        "2 · EVIDENCE ROUTE (how each crop was placed — auto, no GT)",
        fontsize=14,
        fontweight="bold",
        color="#0d3b66",
    )
    steps = [
        "MRI study",
        "view router",
        "per-condition localizer",
        "auto crop (no GT)",
        "robust grader",
        "finding graph",
    ]
    x = 2.0
    for i, st in enumerate(steps):
        w = 0.6 * len(st) + 3.5
        ax.add_patch(
            FancyBboxPatch(
                (x, 77.2), w, 3.2, boxstyle="round,pad=0.2", fc="#e0f2f1", ec="#2a9d8f", lw=1.5
            )
        )
        ax.text(x + w / 2, 78.8, st, fontsize=11.5, ha="center", va="center", color="#0d3b66")
        x += w
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 1.6, 78.8),
                xytext=(x, 78.8),
                arrowprops={"arrowstyle": "-|>", "color": "#2a9d8f", "lw": 2},
            )
            x += 1.9

    # 3) prediction vs held-out reference (the key panel)
    ax.text(
        1.5,
        74.0,
        "3 · PREDICTION vs HELD-OUT REFERENCE  (reference = transparency only, NOT a model input)",
        fontsize=14,
        fontweight="bold",
        color="#0d3b66",
    )
    cols = [
        (2.0, "finding", "left"),
        (19, "level", "left"),
        (25.5, "route", "left"),
        (35, "MODEL severity", "left"),
        (50, "P(severe)", "left"),
        (61, "conf", "left"),
        (66, "stability", "left"),
        (81, "HELD-OUT REF", "left"),
        (99, "correctness", "right"),
    ]
    yhead = 71.5
    ax.add_patch(Rectangle((1, yhead - 1.2), 98, 2.4, fc="#0d3b66", ec="none"))
    for cx, name, ha in cols:
        ax.text(cx, yhead, name, fontsize=11, fontweight="bold", color="white", va="center", ha=ha)
    # visual separator between model block and reference block
    ax.plot([79.5, 79.5], [12, yhead + 1.2], color="#888", lw=1.2, ls="--")
    ax.text(33, yhead + 2.0, "◀ MODEL OUTPUT", fontsize=10.5, color="#1565c0", fontweight="bold")
    ax.text(
        80.5, yhead + 2.0, "HELD-OUT REFERENCE ▶", fontsize=10.5, color="#555", fontweight="bold"
    )

    rows = _relevant_findings(v, k=7)
    rh = (yhead - 1.2 - 14) / max(len(rows), 1)
    rh = min(rh, 7.2)
    y = yhead - 1.2 - rh / 2
    for f in rows:
        ax.add_patch(Rectangle((1, y - rh / 2), 98, rh * 0.94, fc="#f6f8fb", ec="#dde3ea", lw=0.8))
        cond_short = (
            f["condition"]
            .replace("_stenosis", "")
            .replace("_narrowing", "")
            .replace("neural_", "")
            .replace("_", " ")
        )
        ax.text(2.0, y, cond_short, fontsize=11, va="center", color="#222")
        ax.text(19, y, f["level"], fontsize=11, va="center")
        ax.add_patch(
            Rectangle((25.5, y - 1.0), 8.0, 2.0, fc=ROUTE_COLOR[f["view_route"]], ec="none")
        )
        ax.text(
            25.8, y, f["view_route"].replace("_", "-"), fontsize=8.5, color="white", va="center"
        )
        # model severity (colored)
        ax.text(
            35,
            y,
            f["severity_estimate"],
            fontsize=11.5,
            va="center",
            fontweight="bold",
            color=SEV_COLOR[f["severity_estimate"]],
        )
        # P(severe) bar
        ps = f["probabilities"]["P(severe)"]
        ax.add_patch(Rectangle((50, y - 0.7), 8.5, 1.4, fc="#eceff1", ec="#cfd8dc"))
        ax.add_patch(Rectangle((50, y - 0.7), 8.5 * ps, 1.4, fc="#c62828", ec="none"))
        ax.text(58.8, y, f"{ps:.2f}", fontsize=9.5, va="center")
        ax.text(61, y, f"{f['calibrated_confidence']:.2f}", fontsize=10.5, va="center")
        grade_abbr = {"stable": "stable", "mildly_unstable": "mild", "unstable": "unstable"}
        stab = grade_abbr.get(f["evidence_stability_grade"], "-")
        itype = (
            (f["instability_type"] or "")
            .replace("_sensitive", "")
            .replace("axial_candidate", "axial")
        )
        ax.text(
            66,
            y,
            f"{stab}{(' / ' + itype) if itype and itype not in ('stable', '') else ''}",
            fontsize=9.5,
            va="center",
            color="#555",
        )
        # held-out reference (gray box) — clearly separated from the model block
        ref = f["held_out_reference_label"] or "—"
        ax.add_patch(
            FancyBboxPatch(
                (80.5, y - 1.0), 8.5, 2.0, boxstyle="round,pad=0.1", fc="#eceff1", ec="#9e9e9e"
            )
        )
        ax.text(84.75, y, f"REF: {ref}", fontsize=9.5, va="center", ha="center", color="#444")
        # correctness badge (right-aligned so it never runs off the edge)
        col, label = CORRECT_STYLE[f["correctness"]]
        ax.text(99, y, label, fontsize=10.5, va="center", ha="right", color=col, fontweight="bold")
        y -= rh

    # 4) safety/review + 5) note
    ax.text(1.5, 11.0, "4 · SAFETY & REVIEW", fontsize=14, fontweight="bold", color="#0d3b66")
    review = [f for f in v["findings"] if f["review_required"]]
    if review:
        reasons = sorted({r for f in review for r in f["review_reasons"]})
        rtxt = ", ".join(reasons[:6])
    else:
        rtxt = "no findings flagged for review (selective, not always-on)"
    ax.text(
        1.5,
        8.6,
        f"{len(review)} finding(s) flagged · reasons: {rtxt}",
        fontsize=11.5,
        color="#e65100",
        va="center",
    )
    # similar research cases (explanation only) — show for the top finding if available
    sim_f = next((f for f in _relevant_findings(v, 3) if f.get("similar_research_cases")), None)
    if sim_f:
        sc = sim_f["similar_research_cases"]
        dist = " ".join(f"{k[:4]}:{val}" for k, val in sc["severity_distribution"].items())
        ax.text(
            55,
            8.6,
            f"similar research cases (top-{sc['k']}): {dist}  → majority {sc['majority_severity']} "
            "(explanation only, no prediction change)",
            fontsize=10,
            color="#555",
            va="center",
        )

    ax.text(
        1.5, 5.5, "5 · CORRECTNESS / FAILURE NOTE", fontsize=14, fontweight="bold", color="#0d3b66"
    )
    note = _note(v)
    ax.text(1.5, 3.1, note, fontsize=12, color="#222", va="center", wrap=True)
    ax.text(98.5, 1.0, v["reference_note"], fontsize=8, color="#888", ha="right", va="center")

    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _note(v):
    s = v["study_summary"]
    cat = v["case_category"]
    if cat == "false_negative" or s["worst_failure_mode"] == "severe_false_negative":
        return (
            "A severe finding was MISSED (severe false negative). This is the costliest error; "
            "evidence stability / review flags it for human research review where possible."
        )
    if cat == "hard_right_foraminal":
        return (
            "Right-foraminal hard case — the weakest route (signal/sample-limited). Shown with its "
            "held-out reference so the miss/uncertainty is explicit, not hidden."
        )
    if cat == "false_positive":
        return "A non-severe finding was over-called severe (false positive) — review-flagged."
    if cat == "correct_severe":
        return (
            "The model correctly identified the severe finding(s) and the prediction matches the "
            "held-out reference. Confident, stable, high route quality."
        )
    if cat == "mostly_normal":
        return (
            "Mostly normal/mild study with NO review flags — demonstrates the review layer is "
            "selective, not always-on."
        )
    if cat in ("unstable_review", "axial_uncertain"):
        return (
            "At least one finding is unstable under localizer perturbation and flagged for review; "
            "the instability TYPE names the cause (e.g. axial leveling)."
        )
    if cat == "model_disagreement":
        return "Deployed and comparison graders disagree on a finding → flagged for review."
    return "Mixed study; see the prediction-vs-reference table for per-finding correctness."


def render_legend(path):
    fig = plt.figure(figsize=(18, 5.5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 86), 100, 14, fc="#0d3b66", ec="none"))
    ax.text(
        2,
        93,
        "How to read a case viewer card",
        fontsize=22,
        fontweight="bold",
        color="white",
        va="center",
    )
    items = [
        (
            "MODEL OUTPUT (left of dashed line)",
            "severity estimate + P(severe) + calibrated confidence + evidence stability/type",
            "#1565c0",
        ),
        (
            "HELD-OUT REFERENCE (right)",
            "the RSNA research target — shown for transparency, NEVER a model input",
            "#555",
        ),
        (
            "correctness",
            "✓ severe-correct/exact · ✗ severe false-neg/pos · ~ non-severe miss (code-derived)",
            "#0d3b66",
        ),
        ("severity colors", "green = normal_mild · amber = moderate · red = severe", "#c62828"),
        (
            "evidence stability",
            "stable / mildly / unstable; type = crop / slice / axial-candidate / route sensitive",
            "#9d4edd",
        ),
        (
            "review_required",
            "low_confidence · model_disagreement · evidence_unstable · axial_level_uncertainty",
            "#e65100",
        ),
    ]
    y = 80
    for head, body, col in items:
        ax.add_patch(Rectangle((2, y - 1.2), 2, 2.4, fc=col, ec="none"))
        ax.text(5.5, y, head + ":", fontsize=13, fontweight="bold", color=col, va="center")
        ax.text(34, y, body, fontsize=12, color="#222", va="center")
        y -= 12.5
    ax.text(
        2,
        2,
        "Research-only · not diagnostic. Cards render structured outputs, not DICOM pixels.",
        fontsize=10,
        color="#888",
        va="center",
    )
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    device = select_device("auto")
    PACK.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("[case-viewer] building finding graphs + viewers on locked-test auto ...")
    viewers, prefer = _build_viewers(device)
    print(
        f"[case-viewer] {len(viewers)} viewers built ({len(prefer)} studies have instability types)"
    )
    chosen = _select(viewers, prefer)
    titles = {
        "case_canal_correct_severe": "Canal — correct severe",
        "case_left_foraminal_correct_severe": "Left foraminal — correct severe",
        "case_right_foraminal_hard": "Right foraminal — hard case",
        "case_subarticular_correct": "Subarticular — correct severe",
        "case_axial_unstable": "Axial — unstable (flagged)",
        "case_model_disagreement": "Model disagreement",
        "case_review_required": "Review catches a severe FN",
        "case_mostly_normal": "Mostly normal — 0 review",
    }
    index = []
    for name, v in chosen.items():
        (PACK / f"{v['case_id']}.json").write_text(json.dumps(v, indent=2))
        (PACK / f"{v['case_id']}.md").write_text(cvm.render_case_markdown(v))
        render_card(v, ASSETS / f"{name}.png", titles.get(name, name))
        index.append((name, v["case_id"], v["case_category"], v["study_summary"]))
        print(f"[case-viewer] {name}: {v['case_id']} [{v['case_category']}]")
    render_legend(ASSETS / "prediction_vs_reference_legend.png")
    _doc(index)
    print(f"[case-viewer] wrote {len(chosen)} cards + legend to {ASSETS}")
    return 0


def _doc(index):
    lines = [
        "# Real case viewer pack",
        "",
        "> Research-only · not diagnostic. Each card shows, for one anonymized locked-test study:",
        "> input/evidence route → model prediction → **held-out reference label** → derived",
        "> correctness → safety/review. No DICOM pixels, no identifiers (`case_*`). The held-out",
        "> reference is shown for transparency only and is NEVER a model input.",
        "",
        "| card | case_id | category | highest P(severe) | correct/ref | severe errors |",
        "|---|---|---|---|---|---|",
    ]
    for name, cid, cat, s in index:
        lines.append(
            f"| `{name}` | {cid} | {cat} | {s['highest_p_severe']:.2f} | "
            f"{s['n_exact_correct']}/{s['n_with_reference']} | {s['n_severe_errors']} |"
        )
    lines += [
        "",
        "Cards: `docs/assets/cases/*.png` (1800×1000, large-font, sectioned). Legend:",
        "`prediction_vs_reference_legend.png`. Full JSON/MD pack: `outputs/real/case_viewer_pack/`",
        "(gitignored). Schema: `case_viewer_v1` (`reporting/case_viewer.py`). Reproduce:",
        "`python scripts/make_real_case_viewer_pack.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
