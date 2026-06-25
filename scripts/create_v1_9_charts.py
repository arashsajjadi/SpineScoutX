"""Create clean v1.9 research charts (Phase 2).

Generates 6 clean PNG (and SVG where practical) charts for the README and docs.
No clutter, large readable fonts, consistent colors. Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
OUTDIR = ROOT / "docs/assets/v1_9"
OUTDIR.mkdir(parents=True, exist_ok=True)

# === Colors ===
C_GREEN = "#2a9d5c"   # deployed / raw useful
C_BLUE = "#2563c4"    # safety / triage win
C_GRAY = "#b0b0b0"    # executed negative
C_RED = "#d64545"     # decisive negative
C_AMBER = "#e07b39"   # localization win (not deployed)
C_LIGHT = "#f4f4f4"
C_DARK = "#222222"


def save(fig, stem: str) -> None:
    for ext in ("png", "svg"):
        p = OUTDIR / f"{stem}.{ext}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {stem}.png / .svg")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Model journey timeline
# ─────────────────────────────────────────────────────────────────────────────
def chart_timeline() -> None:
    versions = [
        ("v1.0", "Auto-robust\n5-finding system", C_GREEN, "Deployed\n(best raw)"),
        ("v1.1", "Evidence\nintelligence", C_BLUE, "Triage win"),
        ("v1.2", "Real case\nviewer", C_GRAY, "Negative"),
        ("v1.3", "Axial localization\n+ evidence v3", C_AMBER, "Localization win\n(not deployed)"),
        ("v1.4", "Accuracy\naudit", C_GRAY, "Negative"),
        ("v1.5", "MIL + BiGRU", C_AMBER, "Localization win\n(grading negative)"),
        ("v1.6", "External data\n+ SSL + ConvNeXt", C_RED, "Decisive\nnegative"),
        ("v1.7", "Label repair\n+ triage", C_BLUE, "Triage win\n+0.209 eff. recall"),
        ("v1.8b", "SAM2.1\nmorphometry", C_GRAY, "Negative"),
        ("v1.8c", "Real MedSAM2\n(correction)", C_GRAY, "Negative"),
    ]
    n = len(versions)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1.4, 1.4)
    # Draw timeline spine
    ax.axhline(0, color="#888888", lw=2, zorder=1)
    for i, (ver, label, color, outcome) in enumerate(versions):
        # Node
        ax.scatter(i, 0, color=color, s=240, zorder=4, edgecolors="white", linewidths=2)
        # Version label (below)
        ax.text(i, -0.32, ver, ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=C_DARK)
        # Name label (alternating above/below to avoid overlap)
        y_label = 0.55 if i % 2 == 0 else -0.82
        va = "bottom" if i % 2 == 0 else "top"
        ax.text(i, y_label, label, ha="center", va=va, fontsize=9.5, color=C_DARK,
                multialignment="center")
        # Outcome label (opposite side)
        y_out = -1.15 if i % 2 == 0 else 0.82
        va_out = "top" if i % 2 == 0 else "bottom"
        ax.text(i, y_out, outcome, ha="center", va=va_out, fontsize=8.5, color=color,
                multialignment="center", style="italic")
    # Legend
    legend_items = [
        mpatches.Patch(color=C_GREEN, label="Deployed / best raw"),
        mpatches.Patch(color=C_BLUE, label="Safety / triage win"),
        mpatches.Patch(color=C_AMBER, label="Localization win (not deployed)"),
        mpatches.Patch(color=C_GRAY, label="Executed negative"),
        mpatches.Patch(color=C_RED, label="Decisive negative"),
    ]
    ax.legend(handles=legend_items, loc="lower right", ncol=3, fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_title("SpineScoutX Research Journey — v1.0 to v1.8c", pad=10)
    ax.axis("off")
    save(fig, "model_journey_timeline")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Raw severe recall by finding (deployed model)
# ─────────────────────────────────────────────────────────────────────────────
def chart_raw_recall() -> None:
    routes = ["Canal", "L-foraminal", "R-foraminal\n(weak)", "L-subarticular", "R-subarticular"]
    recalls = [0.830, 0.788, 0.660, 0.746, 0.737]
    macro = 0.752
    colors = [C_GREEN if r >= macro else C_AMBER if r >= 0.70 else C_RED for r in recalls]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(routes, recalls, color=colors, height=0.55, edgecolor="white", linewidth=1.5)
    ax.axvline(macro, color=C_BLUE, lw=2, linestyle="--", label=f"5-route macro {macro:.3f}")
    for bar, r in zip(bars, recalls, strict=False):
        ax.text(r + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{r:.3f}", va="center", fontsize=11.5, fontweight="bold", color=C_DARK)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Locked-test severe recall (argmax)", labelpad=8)
    ax.set_title("Best Deployed Model — Locked-test Severe Recall by Finding", pad=10)
    ax.legend(loc="lower right", fontsize=11)
    ax.text(0.02, 0.03, "Research-only · not diagnostic", transform=ax.transAxes,
            fontsize=9, color="#999999", ha="left", va="bottom")
    fig.tight_layout()
    save(fig, "raw_severe_recall_by_finding")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Experiment outcome matrix
# ─────────────────────────────────────────────────────────────────────────────
def chart_outcome_matrix() -> None:
    versions = ["v1.0", "v1.1", "v1.2", "v1.3", "v1.4", "v1.5", "v1.6", "v1.7", "v1.8b", "v1.8c"]
    dims = ["Raw\nAccuracy", "Localization", "Evidence /\nExplainability", "Safety /\nTriage",
            "Data Insight"]
    # ✔ = win, – = neutral, ✘ = negative
    MATRIX = [
        ["✔", "✔", "–", "–", "✔"],    # v1.0  baseline deployed
        ["–", "–", "✔", "✔", "✔"],    # v1.1  evidence + stability
        ["–", "–", "✔", "–", "–"],    # v1.2  viewer
        ["–", "✔", "✔", "✔", "–"],    # v1.3  localization + evidence v3
        ["–", "–", "–", "–", "✔"],    # v1.4  audit
        ["–", "✔", "–", "–", "✔"],    # v1.5  MIL negative + BiGRU loc win
        ["✘", "–", "–", "–", "✔"],    # v1.6  all four decisive loss
        ["–", "–", "–", "✔", "✔"],    # v1.7  label repair + triage win
        ["–", "–", "–", "–", "✔"],    # v1.8b morphometry negative
        ["–", "–", "–", "–", "✔"],    # v1.8c real medsam2 negative
    ]
    COLOR_MAP = {"✔": C_GREEN, "–": "#e0e0e0", "✘": C_RED}

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for i, row in enumerate(MATRIX):
        for j, val in enumerate(row):
            bg = COLOR_MAP[val]
            rect = plt.Rectangle([j - 0.44, i - 0.42], 0.88, 0.84,
                                  color=bg, zorder=2, linewidth=0)
            ax.add_patch(rect)
            ax.text(j, i, val, ha="center", va="center", fontsize=15,
                    color="white" if val == "✘" else (C_DARK if val == "–" else "white"),
                    fontweight="bold" if val != "–" else "normal", zorder=3)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_yticks(range(len(versions)))
    ax.set_yticklabels(versions, fontsize=11, fontfamily="monospace")
    ax.set_xlim(-0.5, len(dims) - 0.5)
    ax.set_ylim(-0.5, len(versions) - 0.5)
    ax.invert_yaxis()
    ax.set_title("Experiment Outcome Matrix — SpineScoutX v1.0 → v1.8c", pad=10)
    ax.tick_params(bottom=False, left=False)
    legend_items = [
        mpatches.Patch(color=C_GREEN, label="✔  Positive outcome"),
        mpatches.Patch(color="#e0e0e0", label="–  Neutral / N/A"),
        mpatches.Patch(color=C_RED, label="✘  Negative outcome"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    save(fig, "experiment_outcome_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Foraminal attempts comparison
# ─────────────────────────────────────────────────────────────────────────────
def chart_foraminal_comparison() -> None:
    arms = [
        "Deployed\nreference\n(best raw)",
        "LSS pretrain\n(v1.6)",
        "LSS+RSNA\njoint (v1.6)",
        "ConvNeXt\n(v1.6)",
        "SAM2.1\nmorphometry\n(v1.8b)",
        "Real MedSAM2\nmorphometry\n(v1.8c)",
    ]
    left_recall = [0.788, 0.596, 0.788, 0.615, 0.788, 0.788]
    right_recall = [0.660, 0.509, 0.660, 0.452, 0.660, 0.660]
    colors = [C_GREEN, C_RED, C_GRAY, C_RED, C_GRAY, C_GRAY]

    x = np.arange(len(arms))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, left_recall, width, label="L-foraminal", color=colors,
                alpha=0.9, edgecolor="white", linewidth=1.2)
    ax.bar(x + width / 2, right_recall, width, label="R-foraminal", color=colors,
                alpha=0.6, edgecolor="white", linewidth=1.2, hatch="//")
    # Annotate best
    ax.annotate("← Best raw model\n   (both deployed &\n   all fusions = same)",
                xy=(0, 0.660 + 0.02), xytext=(1.2, 0.58),
                arrowprops={"arrowstyle": "->", "color": C_GREEN, "lw": 1.5},
                fontsize=10, color=C_GREEN, fontweight="bold")
    ax.axhline(0.660, color=C_GREEN, lw=1.2, linestyle=":", alpha=0.5)
    ax.axhline(0.788, color=C_GREEN, lw=1.2, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=10)
    ax.set_ylabel("Locked-test severe recall", labelpad=8)
    ax.set_title("Foraminal Grading Attempts — Best Raw = Deployed Reference", pad=10)
    ax.legend(["L-foraminal recall", "R-foraminal recall"], loc="lower right", fontsize=10)
    ax.text(0.02, 0.03, "Research-only · not diagnostic", transform=ax.transAxes,
            fontsize=9, color="#999999")
    fig.tight_layout()
    save(fig, "foraminal_attempts_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — Triage effective recall (v1.7)
# ─────────────────────────────────────────────────────────────────────────────
def chart_triage_recall() -> None:
    budgets = [0, 5, 10, 15, 20]
    raw = [0.724, 0.724, 0.724, 0.724, 0.724]
    triage = [0.724, 0.819, 0.867, 0.933, 0.952]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(budgets, raw, triage, alpha=0.15, color=C_BLUE, label="_nolabel")
    ax.plot(budgets, raw, "o--", color=C_GRAY, lw=2.5, markersize=7, label="Raw argmax only (0.724)")
    ax.plot(budgets, triage, "o-", color=C_BLUE, lw=2.5, markersize=8,
            label="v1.7 triage: effective recall")
    for b, t in zip(budgets, triage, strict=False):
        if b > 0:
            ax.annotate(f"{t:.3f}", (b, t), textcoords="offset points", xytext=(6, 5),
                        fontsize=10.5, color=C_BLUE, fontweight="bold")
    ax.annotate("← 15% budget:\n   0.724 → 0.933\n   (+0.209)", xy=(15, 0.933),
                xytext=(16.5, 0.80), arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 1.5},
                fontsize=11, color=C_BLUE, fontweight="bold")
    ax.set_xlim(0, 22)
    ax.set_ylim(0.65, 1.02)
    ax.set_xlabel("Human review budget (% of all findings flagged)", labelpad=8)
    ax.set_ylabel("Effective foraminal severe recall", labelpad=8)
    ax.set_title("v1.7 Triage — Effective Severe Recall vs Review Budget\n"
                 "(review/triage metric — NOT raw argmax grading)", pad=10)
    ax.legend(loc="lower right", fontsize=11)
    ax.text(0.02, 0.03, "Research-only · not diagnostic · deployed grader unchanged",
            transform=ax.transAxes, fontsize=9, color="#999999")
    fig.tight_layout()
    save(fig, "triage_effective_recall")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6 — What we learned
# ─────────────────────────────────────────────────────────────────────────────
def chart_what_we_learned() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    # Title
    ax.text(0.5, 0.97, "SpineScoutX — What We Learned (v1.0 – v1.8c)",
            ha="center", va="top", fontsize=16, fontweight="bold", color=C_DARK)

    # NOT the bottleneck — left panel
    neg_items = [
        "Pipeline bugs (v1.4 audit: 0/4384 alignment errors)",
        "Localizer / crop quality (v1.3/v1.5 BiGRU: ±1-hit 0.487→0.616, grading unchanged)",
        "MIL / attention pooling (v1.5: dev +0.125 R-for → test −0.207 collapse)",
        "External domain-shifted data (v1.6 LSS: decisive loss −0.151 to −0.192)",
        "SimCLR self-supervised pre-training (v1.6: non-convergent)",
        "Anatomy prior / channel injection (v1.6: no gain)",
        "Larger backbone — ConvNeXt (v1.6: decisive loss −0.173 to −0.208)",
        "Soft label cleaning / noise-aware training (v1.7: cleaning rejected by dev)",
        "SAM2.1 morphometry (v1.8b: AUROC 0.687 standalone but redundant; fusion Δ=0)",
        "Real MedSAM2 morphometry (v1.8c: AUROC 0.551, weaker; fusion Δ=0)",
    ]
    pos_items = [
        "In-domain severe-label quality is the binding ceiling",
        "704-case expert review pack created (v1.7; awaiting labels)",
        "Triage improves effective safety: 0.724 → 0.933 at 15% budget (v1.7)",
        "BiGRU axial localization: ±1-slice-hit 0.487→0.616 (v1.5, not deployed)",
        "VisionServeX MedSAM2 integration: reusable, CI-safe (v1.8c)",
        "Only remaining lever: expert re-annotation + clean-labelled held-out test",
    ]

    left_x, right_x = 0.03, 0.52
    # Left box
    ax.add_patch(mpatches.FancyBboxPatch([left_x - 0.01, 0.03], 0.46, 0.87,
                 boxstyle="round,pad=0.01", fc="#fff5f5", ec=C_RED, lw=1.5))
    ax.text(left_x + 0.21, 0.87, "NOT the bottleneck", ha="center", va="top",
            fontsize=13, fontweight="bold", color=C_RED)
    for k, item in enumerate(neg_items):
        ax.text(left_x, 0.82 - k * 0.075, f"✘  {item}", ha="left", va="top",
                fontsize=9.5, color=C_DARK, wrap=True)

    # Right box
    ax.add_patch(mpatches.FancyBboxPatch([right_x - 0.01, 0.03], 0.46, 0.87,
                 boxstyle="round,pad=0.01", fc="#f0fff4", ec=C_GREEN, lw=1.5))
    ax.text(right_x + 0.21, 0.87, "Genuine insights / next steps", ha="center", va="top",
            fontsize=13, fontweight="bold", color=C_GREEN)
    for k, item in enumerate(pos_items):
        ax.text(right_x, 0.82 - k * 0.11, f"✔  {item}", ha="left", va="top",
                fontsize=9.5, color=C_DARK, wrap=True)

    ax.text(0.5, 0.01, "Research-only · non-commercial · not diagnostic",
            ha="center", va="bottom", fontsize=9, color="#999999")
    fig.tight_layout()
    save(fig, "what_we_learned")


def main() -> int:
    print("generating v1.9 charts →", OUTDIR)
    chart_timeline()
    chart_raw_recall()
    chart_outcome_matrix()
    chart_foraminal_comparison()
    chart_triage_recall()
    chart_what_we_learned()
    print("all charts done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
