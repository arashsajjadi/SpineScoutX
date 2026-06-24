#!/usr/bin/env python3
"""Real-data evidence case viewer (v1.3) — what the model SAW → predicted → reference.

For real locked-test cases (hashed `case_*`) this renders three panels:
  A · REAL EVIDENCE SIGNALS — per-finding *derived* scalars from the actual auto crop
      (crop-centre x/y in px, slice/instance index, mean crop intensity) — pixel-free,
      non-reconstructive, metadata-free (per `real_evidence_asset_policy.md`);
  B · PREDICTION vs HELD-OUT REFERENCE — severity + P(severe) + code-derived correctness;
  C · SAFETY — evidence-v3 severe-FN risk + review reason + side-aware (v2) similar cases.

The full real-pixel viewer is written locally under gitignored
`outputs/real/evidence_case_viewer/`; committed assets (docs/assets/real_cases/) are PIXEL-FREE.
No DICOMs, no identifiers, no GT at inference; the reference is shown for transparency only.
Research-only. Not diagnostic. Reproduce: `python scripts/make_real_evidence_case_viewer.py`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
PACK = ROOT / "outputs/real/evidence_case_viewer"
ASSETS = ROOT / "docs/assets/real_cases"
DOC = ROOT / "docs/run_logs/real_evidence_case_viewer.md"
CACHES = {
    "spinal_canal_stenosis": "data/cache/rsna_auto_canal_all",
    "left_neural_foraminal_narrowing": "data/cache/rsna_auto_foraminal",
    "right_neural_foraminal_narrowing": "data/cache/rsna_auto_foraminal",
    "left_subarticular_stenosis": "data/cache/rsna_auto_subarticular",
    "right_subarticular_stenosis": "data/cache/rsna_auto_subarticular",
}
SEV_COLOR = {"normal_mild": "#2e7d32", "moderate": "#f9a825", "severe": "#c62828"}
CORRECT_STYLE = {
    "severe_correct": ("#2e7d32", "✓ SEVERE CORRECT"),
    "exact_correct": ("#2e7d32", "✓ EXACT"),
    "non_severe_mismatch": ("#f9a825", "~ NON-SEVERE MISS"),
    "severe_false_negative": ("#c62828", "✗ SEVERE FALSE NEG"),
    "severe_false_positive": ("#e65100", "✗ SEVERE FALSE POS"),
    "no_reference": ("#9e9e9e", "– NO REF"),
}

_cv = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("cv", ROOT / "scripts/make_real_case_viewer_pack.py")
)
importlib.util.spec_from_file_location(
    "cv", ROOT / "scripts/make_real_case_viewer_pack.py"
).loader.exec_module(_cv)


def _real_signals():
    """{(study, condition, level, side) -> dict of REAL derived crop scalars} (pixel-free)."""
    out = {}
    for cond, cpath in CACHES.items():
        man = pd.read_parquet(ROOT / cpath / "manifest.parquet")
        man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))]
        for r in man.itertuples():
            try:
                arr = np.load(ROOT / cpath / r.crop_path)
                mean_i = round(float(arr.mean()), 3)
            except Exception:  # noqa: BLE001
                mean_i = float("nan")
            out[(str(r.study_id), cond, str(r.level), str(getattr(r, "side", "") or ""))] = {
                "x": round(float(r.x), 1),
                "y": round(float(r.y), 1),
                "slice": int(r.instance_number),
                "mean_intensity": mean_i,
            }
    return out


def _v2_retrieval():
    p = ROOT / "outputs/real/similar_case_retrieval_v2_records.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return {
        (str(r.study_id), r.condition, r.level, str(r.side or "")): {
            "majority": r.majority_severity,
            "dist": dict(r.severity_distribution),
            "tier": r.tier,
        }
        for r in df.itertuples()
    }


def render_card(v, path, title):
    fig = plt.figure(figsize=(20, 11), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    study = v["case_id"]  # already hashed; the underlying study id is not exposed here
    findings = _cv._relevant_findings(v, k=6)

    ax.add_patch(Rectangle((0, 93), 100, 7, fc="#0d3b66", ec="none"))
    ax.text(
        1.5,
        96.3,
        f"Real evidence case — {title}",
        fontsize=23,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(98.5, 96.3, study, fontsize=14, color="#cfe3ff", va="center", ha="right")
    ax.text(
        1.5,
        94,
        "real locked-test case · auto (no GT) · pixel-free derived evidence · NOT DIAGNOSTIC",
        fontsize=10.5,
        color="#cfe3ff",
        va="center",
    )

    # Panel A — REAL EVIDENCE SIGNALS (derived scalars from the actual crop)
    ax.text(
        1.5,
        90,
        "A · REAL EVIDENCE SIGNALS  (derived from the actual auto crop — no pixels committed)",
        fontsize=14,
        fontweight="bold",
        color="#0d3b66",
    )
    acols = [
        (2, "finding"),
        (24, "route"),
        (37, "auto crop centre (x,y) px"),
        (62, "slice #"),
        (74, "mean intensity"),
    ]
    ax.add_patch(Rectangle((1, 86.4), 98, 2.0, fc="#e0f2f1", ec="none"))
    for cx, name in acols:
        ax.text(cx, 87.4, name, fontsize=10.5, fontweight="bold", color="#00695c", va="center")
    y = 84.5
    for f in findings:
        sig = v.get("_sig", {}).get((f["condition"], f["level"], f.get("side") or ""))
        cond = (
            f["condition"].replace("_stenosis", "").replace("_narrowing", "").replace("neural_", "")
        )
        ax.text(2, y, f"{cond} {f['level']}", fontsize=10.5, va="center")
        ax.text(24, y, f["view_route"].replace("_", "-"), fontsize=10, va="center", color="#457b9d")
        if sig:
            ax.text(37, y, f"({sig['x']:.0f}, {sig['y']:.0f})", fontsize=10.5, va="center")
            ax.text(62, y, f"{sig['slice']}", fontsize=10.5, va="center")
            ax.text(74, y, f"{sig['mean_intensity']:.3f}", fontsize=10.5, va="center")
        else:
            ax.text(37, y, "n/a", fontsize=10, va="center", color="#999")
        y -= 2.5

    # Panel B — PREDICTION vs HELD-OUT REFERENCE
    ax.text(
        1.5,
        67,
        "B · PREDICTION vs HELD-OUT REFERENCE  (reference = transparency only, NOT a model input)",
        fontsize=14,
        fontweight="bold",
        color="#0d3b66",
    )
    bcols = [
        (2, "finding"),
        (24, "MODEL severity"),
        (42, "P(severe)"),
        (57, "conf"),
        (66, "HELD-OUT REF"),
        (99, "correct?"),
    ]
    ax.add_patch(Rectangle((1, 63.4), 98, 2.0, fc="#0d3b66", ec="none"))
    ax.plot([64.5, 64.5], [40, 65.4], color="#888", lw=1.1, ls="--")
    for cx, name in bcols:
        ha = "right" if name == "correct?" else "left"
        ax.text(cx, 64.4, name, fontsize=10.5, fontweight="bold", color="white", va="center", ha=ha)
    y = 61.2
    for f in findings:
        cond = (
            f["condition"].replace("_stenosis", "").replace("_narrowing", "").replace("neural_", "")
        )
        ax.text(2, y, f"{cond} {f['level']}", fontsize=10.5, va="center")
        ax.text(
            24,
            y,
            f["severity_estimate"],
            fontsize=11,
            fontweight="bold",
            color=SEV_COLOR[f["severity_estimate"]],
            va="center",
        )
        ps = f["probabilities"]["P(severe)"]
        ax.add_patch(Rectangle((42, y - 0.8), 10, 1.6, fc="#eceff1", ec="#cfd8dc"))
        ax.add_patch(Rectangle((42, y - 0.8), 10 * ps, 1.6, fc="#c62828", ec="none"))
        ax.text(52.5, y, f"{ps:.2f}", fontsize=9.5, va="center")
        ax.text(57, y, f"{f['calibrated_confidence']:.2f}", fontsize=10, va="center")
        ref = f["held_out_reference_label"] or "—"
        ax.add_patch(
            FancyBboxPatch(
                (66, y - 0.9), 9, 1.8, boxstyle="round,pad=0.1", fc="#eceff1", ec="#9e9e9e"
            )
        )
        ax.text(70.5, y, f"REF: {ref}", fontsize=9.5, va="center", ha="center", color="#444")
        col, label = CORRECT_STYLE[f["correctness"]]
        ax.text(99, y, label, fontsize=10.5, va="center", ha="right", color=col, fontweight="bold")
        y -= 3.0

    # Panel C — SAFETY (v3 risk + review + side-aware retrieval)
    ax.text(
        1.5,
        21,
        "C · SAFETY & EVIDENCE INTELLIGENCE",
        fontsize=14,
        fontweight="bold",
        color="#0d3b66",
    )
    s = v["study_summary"]
    review = [f for f in v["findings"] if f["review_required"]]
    reasons = sorted({r for f in review for r in f["review_reasons"]})[:6]
    ax.text(
        1.5,
        17.8,
        f"review_required: {len(review)} finding(s) · reasons: {', '.join(reasons) or 'none'}",
        fontsize=11,
        color="#e65100",
        va="center",
    )
    ax.text(
        1.5,
        14.8,
        f"correct vs reference: {s['n_exact_correct']}/{s['n_with_reference']} · severe errors: "
        f"{s['n_severe_errors']} · worst: {s['worst_failure_mode']} · route quality g/f/w: "
        f"{s['route_quality_summary']['good']}/{s['route_quality_summary']['fair']}/{s['route_quality_summary']['weak']}",
        fontsize=11,
        color="#222",
        va="center",
    )
    # side-aware retrieval for the top finding
    rf = next(
        (
            f
            for f in findings
            if v.get("_retr", {}).get((f["condition"], f["level"], f.get("side") or ""))
        ),
        None,
    )
    if rf:
        rr = v["_retr"][(rf["condition"], rf["level"], rf.get("side") or "")]
        dist = " ".join(f"{k[:4]}:{val}" for k, val in rr["dist"].items())
        ax.text(
            1.5,
            11.8,
            f"similar research cases (side-aware v2, {rf['level']}): {dist} → majority "
            f"{rr['majority']} [tier {rr['tier']}] — explanation only, no prediction change",
            fontsize=10.5,
            color="#555",
            va="center",
        )
    ax.text(1.5, 2.0, v["reference_note"], fontsize=8.5, color="#888", va="center")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    from spinescoutx.training.optim import select_device

    device = select_device("auto")
    PACK.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("[real-evidence] building viewers (real locked-test) ...")
    viewers, _prefer = _cv._build_viewers(device)
    signals = _real_signals()
    retr = _v2_retrieval()
    # attach per-study real signals + retrieval (keyed by cond/level/side) onto each viewer
    # we need the underlying study id; rebuild it from the case_id mapping via the graphs build
    # (the viewer stores case_id only) -> we instead re-key signals by case_id below.
    from spinescoutx.reporting.finding_graph_schema import case_id as _cid

    sig_by_case, retr_by_case = {}, {}
    for (study, cond, lv, sd), d in signals.items():
        sig_by_case.setdefault(_cid(study), {})[(cond, lv, sd)] = d
    for (study, cond, lv, sd), d in retr.items():
        retr_by_case.setdefault(_cid(study), {})[(cond, lv, sd)] = d
    for v in viewers.values():
        v["_sig"] = sig_by_case.get(v["case_id"], {})
        v["_retr"] = retr_by_case.get(v["case_id"], {})

    chosen = _cv._select(viewers, _prefer)
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
        clean = {k: val for k, val in v.items() if not k.startswith("_")}
        (PACK / f"{v['case_id']}.json").write_text(json.dumps(clean, indent=2))
        render_card(v, ASSETS / f"{name}.png", titles.get(name, name))
        index.append((name, v["case_id"], v["case_category"]))
        print(f"[real-evidence] {name}: {v['case_id']} [{v['case_category']}]")
    _doc(index)
    print(f"[real-evidence] wrote {len(chosen)} pixel-free evidence cards to {ASSETS}")
    return 0


def _doc(index):
    lines = [
        "# Real-data evidence case viewer (v1.3)",
        "",
        "> Research-only · not diagnostic. Each card shows, for a real locked-test case (hashed",
        "> `case_*`): **A** real derived evidence signals (auto crop centre, slice, mean intensity",
        "> — pixel-free), **B** prediction vs **held-out reference** + code-derived correctness,",
        "> **C** evidence-v3 safety/review + side-aware (v2) similar cases. No DICOMs, no",
        "> identifiers, no GT at inference (`real_evidence_asset_policy.md`). The full real-pixel",
        "> viewer is generated locally under `outputs/real/evidence_case_viewer/` (gitignored).",
        "",
        "| card | case_id | category |",
        "|---|---|---|",
    ]
    for name, cid, cat in index:
        lines.append(f"| `{name}` | {cid} | {cat} |")
    lines += ["", "Reproduce: `python scripts/make_real_evidence_case_viewer.py`."]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
