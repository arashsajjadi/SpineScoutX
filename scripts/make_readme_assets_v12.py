#!/usr/bin/env python3
"""Generate readable, wide, one-concept-per-image README assets (v1.2 redesign).

Fixes the "too compressed / unreadable" problem: each asset is wide, large-font, high
contrast, and understandable standalone. The hero is a REAL case (loaded from the
gitignored case-viewer pack JSON — no recompute, no fabrication); the explainers are clearly
schematic KEYS (how to read a card), not claimed model outputs.

Outputs (committed, lightweight): docs/assets/readme/*.png. Research-only, not diagnostic.
Reproduce: run `make_real_case_viewer_pack.py` then `make_readme_assets_v12.py`.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
PACK = ROOT / "outputs/real/case_viewer_pack"
OUT = ROOT / "docs/assets/readme"
SEV_COLOR = {"normal_mild": "#2e7d32", "moderate": "#f9a825", "severe": "#c62828"}


def _fig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _header(ax, title, sub=""):
    ax.add_patch(Rectangle((0, 80), 100, 20, fc="#0d3b66", ec="none"))
    ax.text(2.5, 90, title, fontsize=27, fontweight="bold", color="white", va="center")
    if sub:
        ax.text(2.5, 83.5, sub, fontsize=13, color="#cfe3ff", va="center")


def _load_hero_case():
    """A real correct-severe case from the pack (prefer canal)."""
    files = sorted(glob.glob(str(PACK / "*.json")))
    best = None
    for fp in files:
        v = json.loads(Path(fp).read_text())
        if v["case_category"] == "correct_severe":
            best = v
            if any("canal" in f["condition"] for f in v["findings"]):
                return v
    return best or (json.loads(Path(files[0]).read_text()) if files else None)


def hero_case_viewer():
    v = _load_hero_case()
    fig, ax = _fig(18, 7.5)
    if v is None:  # pack not generated -> skip gracefully
        _header(ax, "Real case viewer", "run make_real_case_viewer_pack.py")
        fig.savefig(OUT / "hero_case_viewer.png", facecolor="white")
        plt.close(fig)
        return
    s = v["study_summary"]
    _header(
        ax,
        "What SpineScoutX outputs: a real case",
        f"{v['case_id']} · {v['case_category']} · auto (no GT) · RESEARCH-ONLY, NOT DIAGNOSTIC",
    )
    ax.text(2.5, 75, "MODEL PREDICTION", fontsize=15, fontweight="bold", color="#1565c0")
    ax.text(64, 75, "HELD-OUT REFERENCE", fontsize=15, fontweight="bold", color="#555")
    ax.text(85, 75, "CORRECT?", fontsize=15, fontweight="bold", color="#0d3b66")
    rows = sorted(v["findings"], key=lambda f: -f["probabilities"]["P(severe)"])[:4]
    y = 66
    for f in rows:
        cond = (
            f["condition"].replace("_stenosis", "").replace("_narrowing", "").replace("neural_", "")
        )
        ax.text(2.5, y, f"{cond} {f['level']}", fontsize=13, va="center", color="#222")
        ax.text(
            30,
            y,
            f["severity_estimate"],
            fontsize=15,
            fontweight="bold",
            color=SEV_COLOR[f["severity_estimate"]],
            va="center",
        )
        ps = f["probabilities"]["P(severe)"]
        ax.add_patch(Rectangle((44, y - 1.4), 14, 2.8, fc="#eceff1", ec="#cfd8dc"))
        ax.add_patch(Rectangle((44, y - 1.4), 14 * ps, 2.8, fc="#c62828", ec="none"))
        ax.text(58.5, y, f"P(sev) {ps:.2f}", fontsize=11, va="center")
        ref = f["held_out_reference_label"] or "—"
        ax.add_patch(
            FancyBboxPatch(
                (64, y - 1.5), 16, 3.0, boxstyle="round,pad=0.1", fc="#eceff1", ec="#9e9e9e"
            )
        )
        ax.text(72, y, ref, fontsize=14, va="center", ha="center", color="#444", fontweight="bold")
        ok = f["correctness"] in ("severe_correct", "exact_correct")
        ax.text(
            85,
            y,
            "✓" if ok else "✗",
            fontsize=22,
            va="center",
            color="#2e7d32" if ok else "#c62828",
            fontweight="bold",
        )
        y -= 11
    ax.text(
        2.5,
        5,
        f"This study: {s['n_exact_correct']}/{s['n_with_reference']} findings match the "
        f"held-out reference · highest P(severe) {s['highest_p_severe']:.2f} · "
        f"{s['n_review_required']} flagged for review.",
        fontsize=12,
        color="#222",
        va="center",
    )
    fig.savefig(OUT / "hero_case_viewer.png", facecolor="white")
    plt.close(fig)


def pipeline():
    fig, ax = _fig(18, 4.6)
    _header(
        ax,
        "Input → output",
        "one MRI study in, a non-diagnostic finding graph out — no ground-truth coordinates used",
    )
    steps = [
        ("MRI study", "#2a9d8f"),
        ("view router", "#2a9d8f"),
        ("per-condition\nlocalizer", "#457b9d"),
        ("auto crop\n(no GT)", "#457b9d"),
        ("robust grader", "#9d4edd"),
        ("finding graph\n+ review", "#0d3b66"),
    ]
    x = 3
    for i, (st, col) in enumerate(steps):
        w = 12.5
        ax.add_patch(
            FancyBboxPatch((x, 30), w, 28, boxstyle="round,pad=0.3", fc="#eef3f8", ec=col, lw=2.5)
        )
        ax.text(
            x + w / 2,
            44,
            st,
            fontsize=14,
            ha="center",
            va="center",
            color="#0d3b66",
            fontweight="bold",
        )
        x += w
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 2.2, 44),
                xytext=(x, 44),
                arrowprops={"arrowstyle": "-|>", "color": col, "lw": 3},
            )
            x += 2.6
    ax.text(
        3,
        12,
        "5/5 findings have a real auto route: canal · L/R foraminal · L/R subarticular.",
        fontsize=13,
        color="#222",
    )
    fig.savefig(OUT / "pipeline_input_to_output.png", facecolor="white")
    plt.close(fig)


def pred_vs_ref():
    fig, ax = _fig(18, 5.5)
    _header(
        ax,
        "How to read prediction vs reference",
        "the model output is placed next to the held-out reference; correctness is code-derived",
    )
    rowsdata = [
        ("severe", 0.86, "severe", True, "✓ severe-correct"),
        ("moderate", 0.34, "severe", False, "✗ severe false-negative (missed)"),
        ("severe", 0.61, "normal_mild", False, "✗ severe false-positive (over-call)"),
        ("normal_mild", 0.04, "normal_mild", True, "✓ exact"),
    ]
    ax.text(3, 70, "MODEL", fontsize=14, fontweight="bold", color="#1565c0")
    ax.text(52, 70, "HELD-OUT REF", fontsize=14, fontweight="bold", color="#555")
    ax.text(72, 70, "verdict (illustrative key)", fontsize=14, fontweight="bold", color="#0d3b66")
    y = 60
    for sev, ps, ref, ok, verdict in rowsdata:
        ax.text(3, y, sev, fontsize=15, fontweight="bold", color=SEV_COLOR[sev], va="center")
        ax.add_patch(Rectangle((20, y - 1.3), 16, 2.6, fc="#eceff1", ec="#cfd8dc"))
        ax.add_patch(Rectangle((20, y - 1.3), 16 * ps, 2.6, fc="#c62828", ec="none"))
        ax.text(37, y, f"P(sev) {ps:.2f}", fontsize=11, va="center")
        ax.add_patch(
            FancyBboxPatch(
                (52, y - 1.4), 16, 2.8, boxstyle="round,pad=0.1", fc="#eceff1", ec="#9e9e9e"
            )
        )
        ax.text(60, y, ref, fontsize=13, va="center", ha="center", color="#444")
        ax.text(
            72,
            y,
            verdict,
            fontsize=13,
            va="center",
            color="#2e7d32" if ok else "#c62828",
            fontweight="bold",
        )
        y -= 12
    ax.text(
        3,
        4,
        "Illustrative key. The reference is for transparency only — NEVER a model input.",
        fontsize=11,
        color="#888",
    )
    fig.savefig(OUT / "prediction_vs_reference_card.png", facecolor="white")
    plt.close(fig)


def stability_explainer():
    fig, ax = _fig(18, 5.5)
    _header(
        ax,
        "Evidence stability",
        "re-run the same grader on plausible localizer perturbations (no GT): does P(severe) move?",
    )
    panels = [
        ("stable", "#2e7d32", "prediction barely moves\n→ trustworthy"),
        ("mildly_unstable", "#f9a825", "moves a little\n→ route quality = fair"),
        ("unstable", "#c62828", "prediction flips\n→ review_required + typed cause"),
    ]
    x = 4
    for name, col, desc in panels:
        ax.add_patch(
            FancyBboxPatch((x, 18), 27, 48, boxstyle="round,pad=0.4", fc="#f6f8fb", ec=col, lw=3)
        )
        ax.add_patch(Rectangle((x, 58), 27, 8, fc=col, ec="none"))
        ax.text(
            x + 13.5,
            62,
            name,
            fontsize=16,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        ax.text(x + 13.5, 40, desc, fontsize=13, ha="center", va="center", color="#222")
        x += 31
    ax.text(
        4,
        8,
        "Type names the cause: crop / slice / axial-candidate / route sensitive. "
        "Unstable findings carry several× the severe-FN rate of stable ones.",
        fontsize=11.5,
        color="#222",
    )
    fig.savefig(OUT / "evidence_stability_explainer.png", facecolor="white")
    plt.close(fig)


def safety_explainer():
    fig, ax = _fig(18, 5.5)
    _header(
        ax,
        "Safety mode: review_required",
        "severe-first; flags low-confidence / unstable / disagreeing findings for human review",
    )
    reasons = [
        "low_confidence",
        "model_disagreement",
        "evidence_unstable",
        "axial_level_uncertainty",
        "near_severe_threshold",
    ]
    x = 3
    for r in reasons:
        w = 0.55 * len(r) + 4
        ax.add_patch(
            FancyBboxPatch(
                (x, 52), w, 6, boxstyle="round,pad=0.2", fc="#fff3e0", ec="#e65100", lw=2
            )
        )
        ax.text(
            x + w / 2,
            55,
            r,
            fontsize=12.5,
            ha="center",
            va="center",
            color="#e65100",
            fontweight="bold",
        )
        x += w + 2
    ax.text(3, 40, "Selective, not always-on:", fontsize=15, fontweight="bold", color="#0d3b66")
    ax.text(
        3,
        32,
        "• a mostly-normal study gets 0 reviews;   • a severe/uncertain study gets several.",
        fontsize=13.5,
        color="#222",
    )
    ax.text(
        3,
        22,
        "Calibration is honest: the graders are already well-calibrated (test ECE 0.03–0.08),",
        fontsize=13.5,
        color="#222",
    )
    ax.text(
        3,
        15,
        "so no temperature is applied — raw probabilities are kept (documented, not hidden).",
        fontsize=13.5,
        color="#222",
    )
    fig.savefig(OUT / "safety_mode_explainer.png", facecolor="white")
    plt.close(fig)


def failure_preview():
    fig, ax = _fig(18, 5.5)
    _header(
        ax,
        "Where it fails (shown, not hidden)",
        "the honest limitations, each with a real measurement",
    )
    items = [
        (
            "Right foraminal",
            "weakest route (severe recall ~0.66); 56% of misses are confidently-normal",
        ),
        (
            "Axial leveling",
            "±1-slice hit ~0.43; robust grader tolerates it but it drives subarticular instability",
        ),
        ("L5–S1 level", "severe recall 0.58 vs 0.87 at L4–L5 (internal domain-shift)"),
        (
            "No external validation",
            "single dataset; no other-institution / prospective / reader-study evidence",
        ),
    ]
    y = 64
    for head, body in items:
        ax.add_patch(Rectangle((3, y - 2.2), 2, 4.4, fc="#c62828", ec="none"))
        ax.text(6.5, y, head + ":", fontsize=14.5, fontweight="bold", color="#c62828", va="center")
        ax.text(28, y, body, fontsize=12.5, color="#222", va="center")
        y -= 14
    fig.savefig(OUT / "failure_gallery_preview.png", facecolor="white")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hero_case_viewer()
    pipeline()
    pred_vs_ref()
    stability_explainer()
    safety_explainer()
    failure_preview()
    print(f"wrote README assets to {OUT}:")
    for f in sorted(glob.glob(str(OUT / "*.png"))):
        print("  ", Path(f).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
