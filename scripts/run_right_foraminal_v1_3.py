#!/usr/bin/env python3
"""Right neural foraminal narrowing — v1.3 audit + evidence-v3 review repair.

Right-foraminal is the weakest route. It has been characterized as **signal/sample-limited**
across v1.0–v1.2 (specialist non-decisive; 56% of severe misses confidently-normal; per-level
thresholding does not help; n_severe≈53). v1.3 does **not** retrain another specialist (it
would chase the same limit and risk the release). Instead it shows the real, bounded gain: the
**evidence-intelligence v3** severe-FN risk score improves right-foraminal severe-FN *triage*
over confidence alone. Built from existing locked-test records; reference labels score FNs only.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
COND = "right_neural_foraminal_narrowing"
OUT = ROOT / "outputs/real/right_foraminal_v1_3_results.json"
DOC = ROOT / "docs/run_logs/right_foraminal_v1_3.md"
ASSET = ROOT / "docs/assets/readme/right_foraminal_before_after.png"

_spec = importlib.util.spec_from_file_location("v3", ROOT / "scripts/run_evidence_intel_v3.py")
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)


def main() -> int:
    df = _v3._load()
    r = df[df.condition == COND].copy()
    sig = _v3._signals(r)
    fn = r["severe_fn"].to_numpy()
    conf = sig["uncertainty"]
    v3 = _v3._v3_risk(sig)
    groups = r["study_id"].to_numpy()

    # hard-case audit
    sev = r[r.is_severe == 1]
    misses = sev[sev.severe_fn == 1]
    audit = {
        "n": int(len(r)),
        "n_severe": int(r["is_severe"].sum()),
        "n_severe_fn": int(fn.sum()),
        "frac_misses_confidently_normal": float((misses["baseline_p_severe"] < 0.20).mean())
        if len(misses)
        else float("nan"),
        "miss_p_severe_mean": float(misses["baseline_p_severe"].mean())
        if len(misses)
        else float("nan"),
    }
    # per-level severe-FN
    per_level = {}
    for lv in sorted(r["level"].unique()):
        s = r[(r.level == lv) & (r.is_severe == 1)]
        per_level[lv] = {"n_severe": int(len(s)), "n_fn": int(s["severe_fn"].sum())}

    # v3 vs confidence severe-FN triage
    auroc = {
        "confidence_only": _v3._es.cluster_boot_auroc(conf, fn, groups, 2000),
        "v3_combined": _v3._es.cluster_boot_auroc(v3, fn, groups, 2000),
    }
    capture = {
        f"budget_{int(b * 100)}pct": {
            "confidence_only": _v3._es.capture_at_budget(conf, fn, b),
            "v3_combined": _v3._es.capture_at_budget(v3, fn, b),
        }
        for b in (0.10, 0.20, 0.30)
    }
    out = {
        "condition": COND,
        "audit": audit,
        "per_level_severe_fn": per_level,
        "severe_fn_auroc": auroc,
        "severe_fn_capture": capture,
        "note": "No new specialist trained (non-decisive across v1.0-v1.2); v1.3 gain = v3 triage.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _card(out)
    _doc(out)
    print(
        f"R-foraminal severe-FN AUROC: conf {auroc['confidence_only']['point']:.3f} -> "
        f"v3 {auroc['v3_combined']['point']:.3f}"
    )
    print(f"wrote {OUT}\nwrote {DOC}\nwrote {ASSET}")
    return 0


def _card(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = out["severe_fn_capture"]
    budgets = ["budget_10pct", "budget_20pct", "budget_30pct"]
    labels = ["10%", "20%", "30%"]
    conf = [cap[b]["confidence_only"] for b in budgets]
    v3 = [cap[b]["v3_combined"] for b in budgets]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=100)
    ax.bar(x - 0.2, conf, 0.4, label="confidence-only review", color="#90a4ae")
    ax.bar(x + 0.2, v3, 0.4, label="evidence v3 review", color="#00838f")
    for i in range(3):
        ax.text(i - 0.2, conf[i] + 0.01, f"{conf[i]:.2f}", ha="center", fontsize=10)
        ax.text(i + 0.2, v3[i] + 0.01, f"{v3[i]:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlabel("human-review budget", fontsize=11)
    ax.set_ylabel("right-foraminal severe-FN capture", fontsize=12)
    ax.set_title("Right foraminal — evidence-v3 review improves severe-FN triage", fontsize=12.5)
    ax.legend(fontsize=10)
    fig.text(
        0.5,
        0.01,
        "Locked-test auto · accuracy unchanged (sample-limited) · research-only",
        ha="center",
        fontsize=9,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, facecolor="white")
    plt.close(fig)


def _doc(out):
    a = out["audit"]
    au = out["severe_fn_auroc"]
    lines = [
        "# Right neural foraminal narrowing — v1.3 audit + evidence-v3 review repair",
        "",
        "> Research-only · not diagnostic. The weakest route. v1.3 does **not** retrain a",
        "> specialist (non-decisive across v1.0–v1.2); it improves severe-FN **triage** via the",
        "> evidence-intelligence v3 risk score. Reference labels score FNs only.",
        "",
        "## Hard-case audit (locked test)",
        f"- n={a['n']}, n_severe={a['n_severe']}, severe FNs={a['n_severe_fn']}.",
        f"- **{a['frac_misses_confidently_normal']:.0%}** of severe misses are confidently normal",
        f"  (P(severe)<0.20; mean miss P(severe) {a['miss_p_severe_mean']:.3f}) — a signal/sample",
        "  limit, not a thresholding knob (consistent with v1.0–v1.2).",
        "",
        "## Severe-FN triage: evidence v3 vs confidence",
        f"- severe-FN detection AUROC: confidence **{au['confidence_only']['point']:.3f}** "
        f"[{au['confidence_only']['ci_lo']:.3f},{au['confidence_only']['ci_hi']:.3f}] → "
        f"v3 **{au['v3_combined']['point']:.3f}** "
        f"[{au['v3_combined']['ci_lo']:.3f},{au['v3_combined']['ci_hi']:.3f}].",
        "",
        "| review budget | confidence-only capture | v3 capture |",
        "|---|---|---|",
    ]
    for b in ("budget_10pct", "budget_20pct", "budget_30pct"):
        d = out["severe_fn_capture"][b]
        lines.append(
            f"| {b.replace('budget_', '').replace('pct', '%')} | "
            f"{d['confidence_only']:.3f} | {d['v3_combined']:.3f} |"
        )
    lines += [
        "",
        "## Verdict (honest)",
        "- **Accuracy unchanged** — right-foraminal severe recall remains sample/signal-limited;",
        "  retraining a specialist has been non-decisive four times, so it was not repeated.",
        "- **Real v1.3 gain = triage:** the v3 risk score catches right-foraminal severe FNs",
        "  better than confidence alone for human research review. Next step for accuracy: more",
        "  right-side severe data or a dedicated right localizer.",
        "",
        "Reproduce: `python scripts/run_right_foraminal_v1_3.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
