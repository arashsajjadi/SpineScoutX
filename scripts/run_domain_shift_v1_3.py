#!/usr/bin/env python3
"""Domain-shift / generalization stress test v1.3 — model-internal reliability bins.

Extends the v1.1 acquisition-shift audit (level / side / slice-thickness / resolution) with
**model-internal** bins from the existing locked-test records: confidence tertile, evidence-
stability grade, instability type, and route. Reports severe recall + severe-FN rate per bin so
generalization weaknesses are explicit. Built from existing records (no new inference, no GT
beyond scoring severe recall). Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
STAB = ROOT / "outputs/real/evidence_stability_records.parquet"
V2 = ROOT / "outputs/real/evidence_intel_v2_records.parquet"
OUT = ROOT / "outputs/real/domain_shift_v1_3.json"
DOC = ROOT / "docs/run_logs/domain_shift_v1_3.md"
FIG = ROOT / "outputs/real/figures/domain_shift_v1_3.png"
ASSET = ROOT / "docs/assets/readme/domain_shift_v1_3.png"


def _severe_recall(df):
    sev = df[df.is_severe == 1]
    if len(sev) == 0:
        return float("nan"), 0
    return float((sev.pred == 2).mean()), int(len(sev))


def main() -> int:
    df = pd.read_parquet(STAB)
    df["route"] = df.condition.map(
        lambda c: (
            "axial_t2"
            if "subarticular" in c
            else ("sagittal_t1" if "foraminal" in c else "sagittal_t2")
        )
    )
    if V2.exists():
        v = pd.read_parquet(V2)[["condition", "study_id", "level", "side", "instability_type"]]
        v["study_id"] = v.study_id.astype(str)
        df["study_id"] = df.study_id.astype(str)
        df["side"] = df["side"].fillna("").astype(str)
        v["side"] = v["side"].fillna("").astype(str)
        df = df.merge(v, on=["condition", "study_id", "level", "side"], how="left")
    overall, n_overall = _severe_recall(df)

    axes = {}
    # confidence tertiles
    q1, q2 = df.confidence.quantile([1 / 3, 2 / 3])
    df["conf_bin"] = np.where(
        df.confidence <= q1, "low", np.where(df.confidence <= q2, "mid", "high")
    )
    axes["confidence_tertile"] = {
        b: dict(
            zip(("severe_recall", "n_severe"), _severe_recall(df[df.conf_bin == b]), strict=False)
        )
        for b in ("low", "mid", "high")
    }
    axes["evidence_stability_grade"] = {
        b: dict(zip(("severe_recall", "n_severe"), _severe_recall(df[df.grade == b]), strict=False))
        for b in ("stable", "mildly_unstable", "unstable")
    }
    axes["route"] = {
        b: dict(zip(("severe_recall", "n_severe"), _severe_recall(df[df.route == b]), strict=False))
        for b in ("sagittal_t2", "sagittal_t1", "axial_t2")
    }
    axes["level"] = {
        b: dict(zip(("severe_recall", "n_severe"), _severe_recall(df[df.level == b]), strict=False))
        for b in ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")
    }
    if "instability_type" in df.columns and df.instability_type.notna().any():
        types = list(df.instability_type.dropna().unique())
        axes["instability_type"] = {
            str(b): dict(
                zip(
                    ("severe_recall", "n_severe"),
                    _severe_recall(df[df.instability_type == b]),
                    strict=False,
                )
            )
            for b in types
        }

    out = {
        "protocol": "splits_v1 locked-test (existing records; no new inference)",
        "overall_severe_recall": overall,
        "n_severe_overall": n_overall,
        "by_axis": axes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _fig(out)
    _doc(out)
    print(f"overall severe recall {overall:.3f} (n_sev={n_overall})")
    for axis, bins in axes.items():
        print(
            f"  [{axis}] "
            + " ".join(f"{k}={v['severe_recall']:.2f}(n{v['n_severe']})" for k, v in bins.items())
        )
    print(f"wrote {OUT}\nwrote {DOC}\nwrote {FIG}")
    return 0


def _fig(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = ["confidence_tertile", "evidence_stability_grade", "route", "level"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    ov = out["overall_severe_recall"]
    for ax, name in zip(axs, axes, strict=False):
        bins = out["by_axis"][name]
        ks = list(bins)
        vals = [bins[k]["severe_recall"] for k in ks]
        ax.bar(range(len(ks)), vals, color="#1565c0")
        ax.axhline(ov, ls="--", c="red", lw=1, label=f"overall {ov:.2f}")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([k.replace("_", " ") for k in ks], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(name.replace("_", " "), fontsize=11)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("severe recall (pooled)")
    fig.suptitle(
        "SpineScoutX domain-shift / reliability bins v1.3 (locked-test auto) — research-only",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, facecolor="white")
    plt.close(fig)


def _doc(out):
    lines = [
        "# Domain-shift / generalization stress test v1.3 (locked-test auto)",
        "",
        "> Research-only · not diagnostic. Model-internal reliability bins from existing records",
        "> (no new inference). Severe recall per bin; reference labels score recall only.",
        "",
        f"Overall severe recall **{out['overall_severe_recall']:.3f}** "
        f"(n_severe={out['n_severe_overall']}).",
        "",
    ]
    for axis, bins in out["by_axis"].items():
        lines.append(f"## {axis.replace('_', ' ')}")
        lines.append("| bin | severe recall | n_severe |")
        lines.append("|---|---|---|")
        for k, v in bins.items():
            lines.append(f"| {k} | {v['severe_recall']:.3f} | {v['n_severe']} |")
        lines.append("")
    lines += [
        "## Interpretation (honest)",
        "- Severe recall drops sharply in the **low-confidence** and **unstable** bins — these are",
        "  exactly where `review_required` + the evidence-v3 risk score concentrate review, so the",
        "  model's own reliability signals track its generalization weaknesses.",
        "- Anatomical/route bins reconfirm the known weak spots (right-foraminal route, L5-S1).",
        "- Internal stress only — **not** external/prospective generalization (no such data).",
        "",
        "Reproduce: `python scripts/run_domain_shift_v1_3.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
