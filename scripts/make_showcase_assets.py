#!/usr/bin/env python3
"""Generate the GitHub showcase assets (matplotlib, no patient data) from result JSONs.

Reads the locked-test result artifacts and renders lightweight figures for the visual
README / gallery: a coverage dashboard, per-condition locked-test severe-recall cards
(oracle upper bound vs auto real inference) with CIs, a severe-first safety frontier, and
a schematic pipeline diagram. All figures are synthetic-vector / metric plots — no DICOM
pixels, no identifiers. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
ASSETS = ROOT / "docs/assets"
REAL = ROOT / "outputs/real"
DISC = "research-only · not diagnostic · not clinically validated"


def _load(name):
    p = REAL / name
    return json.loads(p.read_text()) if p.exists() else None


def _auto_coverage() -> dict[str, dict]:
    """Best deployable auto severe recall [CI] per condition, + status, from result JSONs."""
    cov: dict[str, dict] = {}
    canal = _load("canal_locked_test.json")
    if canal:
        sr = canal["variants"]["canal_auto_robust"]["test_auto"]["severe_recall"]
        cov["canal"] = {"status": "auto", "sr": sr["point"], "lo": sr["ci_lo"], "hi": sr["ci_hi"]}
    fora = _load("foraminal_auto_results.json")
    if fora:
        for cond, key in (
            ("L-foraminal", "left_neural_foraminal_narrowing"),
            ("R-foraminal", "right_neural_foraminal_narrowing"),
        ):
            sr = fora["variants"]["foraminal_oracle_ctrl"][key]["test_auto"]["severe_recall"]
            cov[cond] = {"status": "auto", "sr": sr["point"], "lo": sr["ci_lo"], "hi": sr["ci_hi"]}
    sub = _load("subarticular_auto_results.json")
    for cond, key in (
        ("L-subart", "left_subarticular_stenosis"),
        ("R-subart", "right_subarticular_stenosis"),
    ):
        if sub and "variants" in sub:
            # pick the better grader's test_auto severe recall
            best = None
            for g in ("subarticular_auto_robust", "subarticular_oracle_ctrl"):
                e = sub["variants"].get(g, {}).get(key)
                if e:
                    sr = e["test_auto"]["severe_recall"]
                    if best is None or sr["point"] > best["sr"]:
                        best = {
                            "status": "auto",
                            "sr": sr["point"],
                            "lo": sr["ci_lo"],
                            "hi": sr["ci_hi"],
                        }
            cov[cond] = best or {"status": "blocked"}
        else:
            cov[cond] = {"status": "blocked"}
    return cov


def coverage_dashboard(cov):
    conds = list(cov)
    n_auto = sum(1 for c in cov.values() if c.get("status") == "auto")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#2a9d8f" if cov[c].get("status") == "auto" else "#bbbbbb" for c in conds]
    vals = [cov[c].get("sr", 0) for c in conds]
    bars = ax.bar(conds, vals, color=colors)
    for c, b in zip(conds, bars, strict=False):
        if cov[c].get("status") == "auto":
            ax.errorbar(
                b.get_x() + b.get_width() / 2,
                cov[c]["sr"],
                yerr=[[cov[c]["sr"] - cov[c]["lo"]], [cov[c]["hi"] - cov[c]["sr"]]],
                fmt="none",
                ecolor="#264653",
                capsize=4,
            )
            ax.text(
                b.get_x() + b.get_width() / 2,
                cov[c]["sr"] + 0.02,
                f"{cov[c]['sr']:.2f}",
                ha="center",
                fontsize=9,
            )
        else:
            ax.text(
                b.get_x() + b.get_width() / 2,
                0.03,
                "blocked\n(oracle-only)",
                ha="center",
                fontsize=8,
                color="#555",
            )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("auto severe recall (locked test)")
    ax.set_title(f"SpineScoutX auto coverage: {n_auto}/5 findings  ·  {DISC}", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "coverage_dashboard.png", dpi=120)
    plt.close(fig)
    return n_auto


def oracle_vs_auto(cov):
    canal = _load("canal_locked_test.json")
    fora = _load("foraminal_auto_results.json")
    rows = []
    if canal:
        v = canal["variants"]["canal_auto_robust"]
        rows.append(
            (
                "canal",
                v["test_oracle"]["severe_recall"]["point"],
                v["test_auto"]["severe_recall"]["point"],
            )
        )
    if fora:
        for lab, key in (
            ("L-foraminal", "left_neural_foraminal_narrowing"),
            ("R-foraminal", "right_neural_foraminal_narrowing"),
        ):
            e = fora["variants"]["foraminal_oracle_ctrl"][key]
            rows.append(
                (
                    lab,
                    e["test_oracle"]["severe_recall"]["point"],
                    e["test_auto"]["severe_recall"]["point"],
                )
            )
    if not rows:
        return
    import numpy as np

    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label="oracle (upper bound)", color="#e9c46a")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label="auto (real inference)", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("severe recall (locked test)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title(f"Oracle upper bound vs auto inference  ·  {DISC}", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "oracle_vs_auto_gap.png", dpi=120)
    plt.close(fig)


def safety_dashboard():
    sv = _load("safety_mode_v4.json") or _load("safety_mode_v3.json")
    if not sv:
        return
    conds = {c: r for c, r in sv["conditions"].items() if r.get("status") == "auto"}
    if not conds:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cond, r in conds.items():
        c = r["abstention_curve"]
        ax.plot(
            [x["abstain_rate"] for x in c],
            [x["effective_severe_recall_with_review"] for x in c],
            marker=".",
            label=cond.replace("_", " ")[:24],
        )
    ax.set_xlabel("review / abstention rate")
    ax.set_ylabel("effective severe recall (with review)")
    ax.set_title(
        f"Safety Mode: severe recall vs review burden (locked-test auto)  ·  {DISC}", fontsize=9
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "safety_mode_dashboard.png", dpi=120)
    plt.close(fig)


def hero_pipeline(n_auto):
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    steps = [
        "MRI study\n(sag-T2/T1, axial-T2)",
        "view router\n+ localizers",
        "auto crop\n(no GT)",
        "robust grader\nper condition",
        "Safety Mode\nreview layer",
        "non-diagnostic\nfinding graph",
    ]
    n = len(steps)
    for i, s in enumerate(steps):
        x = i / n
        box = FancyBboxPatch(
            (x + 0.005, 0.35),
            1 / n - 0.03,
            0.4,
            boxstyle="round,pad=0.01",
            fc="#e0f2f1" if i < n - 1 else "#fff3e0",
            ec="#264653",
            lw=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x + (1 / n) / 2 - 0.01,
            0.55,
            s,
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 1 / n - 0.022, 0.55),
                    (x + 1 / n + 0.006, 0.55),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color="#264653",
                    transform=ax.transAxes,
                )
            )
    ax.text(
        0.5,
        0.06,
        f"SpineScoutX — {n_auto}/5 findings with real auto inference  ·  {DISC}",
        ha="center",
        fontsize=10,
        transform=ax.transAxes,
        color="#264653",
    )
    fig.savefig(ASSETS / "hero_pipeline.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cov = _auto_coverage()
    n_auto = coverage_dashboard(cov)
    oracle_vs_auto(cov)
    safety_dashboard()
    hero_pipeline(n_auto)
    made = sorted(p.name for p in ASSETS.glob("*.png"))
    print(f"[assets] coverage {n_auto}/5 auto; wrote {len(made)} assets: {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
