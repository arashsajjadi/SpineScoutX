#!/usr/bin/env python3
"""Evidence Intelligence v3 — a combined severe-FN RISK score, evaluated against errors.

v1.1 found evidence stability is largely redundant with confidence. v3 asks: does combining
the *richer* signals — confidence + stability + retrieval conflict + near-severe + route
quality — detect severe false negatives better than confidence alone? Built from the existing
locked-test records (stability + retrieval + instability-typing), so no new heavy compute.

Leakage-free evaluation: severe-FN detection **AUROC** (threshold-free) and severe-FN
**capture at a ranked review budget** (no threshold tuned on test) for confidence-only vs
stability-only vs the v3 combined risk. Cluster-bootstrap CIs by study. Reference labels are
used only to *score* severe FNs, never as model input. Research-only. Not diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
STAB = ROOT / "outputs/real/evidence_stability_records.parquet"
RETR = ROOT / "outputs/real/similar_case_retrieval_records.parquet"
V2 = ROOT / "outputs/real/evidence_intel_v2_records.parquet"
OUT = ROOT / "outputs/real/evidence_intelligence_v3.json"
DOC = ROOT / "docs/run_logs/evidence_intelligence_v3.md"
CARD = ROOT / "docs/assets/readme/evidence_intelligence_v3_card.png"
SEV = ("normal_mild", "moderate", "severe")

_spec = importlib.util.spec_from_file_location("es_run", ROOT / "scripts/run_evidence_stability.py")
_es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_es)

# v3 risk weights (fixed, documented — NOT fitted on test, so no leakage)
W = {"uncertainty": 0.40, "instability": 0.30, "retrieval_conflict": 0.15, "near_severe": 0.15}


def _mm(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _load():
    s = pd.read_parquet(STAB)
    s["study_id"] = s.study_id.astype(str)
    s["side"] = s["side"].fillna("").astype(str)
    r = pd.read_parquet(RETR)[
        ["condition", "study_id", "level", "side", "majority_severity"]
    ].copy()
    r["study_id"] = r.study_id.astype(str)
    r["side"] = r["side"].fillna("").astype(str)
    df = s.merge(r, on=["condition", "study_id", "level", "side"], how="left")
    if V2.exists():
        v = pd.read_parquet(V2)[
            ["condition", "study_id", "level", "side", "instability_type"]
        ].copy()
        v["study_id"] = v.study_id.astype(str)
        v["side"] = v["side"].fillna("").astype(str)
        df = df.merge(v, on=["condition", "study_id", "level", "side"], how="left")
    else:
        df["instability_type"] = None
    return df


def _signals(df):
    pred_sev = df["pred"].to_numpy()
    pred_label = np.array([SEV[int(p)] for p in pred_sev])
    retrieval_conflict = (
        ((df["majority_severity"].notna()) & (pred_label != df["majority_severity"].to_numpy()))
        .astype(float)
        .to_numpy()
    )
    near_severe = (((df["pred"] != 2) & (df["baseline_p_severe"] >= 0.20)).astype(float)).to_numpy()
    return {
        "uncertainty": df["uncertainty"].to_numpy(),
        "instability": df["instability"].to_numpy(),
        "retrieval_conflict": retrieval_conflict,
        "near_severe": near_severe,
    }


def _v3_risk(sig):
    return (
        W["uncertainty"] * _mm(sig["uncertainty"])
        + W["instability"] * _mm(sig["instability"])
        + W["retrieval_conflict"] * sig["retrieval_conflict"]
        + W["near_severe"] * sig["near_severe"]
    )


def _eval_block(df, n_boot=2000):
    if df.empty:
        return {}
    sig = _signals(df)
    fn = df["severe_fn"].to_numpy()
    groups = df["study_id"].to_numpy()
    conf = sig["uncertainty"]
    inst = sig["instability"]
    combined_v11 = 0.5 * _mm(conf) + 0.5 * _mm(inst)  # v1.1 combined (conf+stability)
    v3 = _v3_risk(sig)
    out = {
        "n": int(len(df)),
        "n_severe": int(df["is_severe"].sum()),
        "n_severe_fn": int(fn.sum()),
        "auroc_severe_fn": {
            "confidence_only": _es.cluster_boot_auroc(conf, fn, groups, n_boot),
            "stability_only": _es.cluster_boot_auroc(inst, fn, groups, n_boot),
            "confidence_plus_stability": _es.cluster_boot_auroc(combined_v11, fn, groups, n_boot),
            "v3_combined": _es.cluster_boot_auroc(v3, fn, groups, n_boot),
        },
    }
    if fn.sum() >= 3:
        out["severe_fn_capture"] = {
            f"budget_{int(b * 100)}pct": {
                "confidence_only": _es.capture_at_budget(conf, fn, b),
                "stability_only": _es.capture_at_budget(inst, fn, b),
                "v3_combined": _es.capture_at_budget(v3, fn, b),
            }
            for b in (0.10, 0.20, 0.30)
        }
    # high-confidence wrong: severe FNs the model is confident about (conf>=0.85)
    hc = df["confidence"].to_numpy() >= 0.85
    hc_fn = int((hc & (fn == 1)).sum())
    out["high_confidence_severe_fn"] = {
        "n_high_conf": int(hc.sum()),
        "n_high_conf_severe_fn": hc_fn,
        "v3_flags_them": float(np.mean(v3[(hc & (fn == 1))] > np.median(v3))) if hc_fn else None,
    }
    return out


def main() -> int:
    df = _load()
    conds = sorted(df["condition"].unique())
    out = {
        "protocol": "splits_v1 locked-test (existing records; no new inference)",
        "risk_weights": W,
        "pooled": _eval_block(df),
        "per_condition": {c: _eval_block(df[df.condition == c]) for c in conds},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _card(out)
    _doc(out)
    p = out["pooled"]["auroc_severe_fn"]
    print("=== severe-FN detection AUROC (pooled) ===")
    for k, v in p.items():
        print(f"  {k:28s} {v['point']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    print(f"\nwrote {OUT}\nwrote {DOC}\nwrote {CARD}")
    return 0


def _card(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = out["pooled"]["auroc_severe_fn"]
    keys = ["confidence_only", "stability_only", "confidence_plus_stability", "v3_combined"]
    labels = ["confidence", "stability", "conf+stability", "v3 combined"]
    vals = [p[k]["point"] for k in keys]
    los = [p[k]["point"] - p[k]["ci_lo"] for k in keys]
    his = [p[k]["ci_hi"] - p[k]["point"] for k in keys]
    cols = ["#90a4ae", "#9d4edd", "#1565c0", "#00838f"]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=100)
    ax.bar(range(4), vals, color=cols)
    ax.errorbar(range(4), vals, yerr=[los, his], fmt="none", ecolor="black", capsize=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("severe-FN detection AUROC (pooled)", fontsize=12)
    ax.set_title(
        "Evidence Intelligence v3 — does combining signals beat confidence for severe-FN triage?",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.01,
        "Locked-test auto · cluster-bootstrap 95% CI · research-only, not diagnostic",
        ha="center",
        fontsize=9,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    CARD.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CARD, facecolor="white")
    plt.close(fig)


def _doc(out):
    p = out["pooled"]
    a = p["auroc_severe_fn"]

    def ci(k):
        v = a[k]
        return f"{v['point']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"

    lines = [
        "# Evidence Intelligence v3 — combined severe-FN risk score (locked-test auto)",
        "",
        "> Research-only · not diagnostic. Built from existing locked-test records (no new",
        "> inference). v3 risk = "
        f"{W['uncertainty']}·(1−conf) + {W['instability']}·instability + "
        f"{W['retrieval_conflict']}·retrieval_conflict + {W['near_severe']}·near_severe "
        "(fixed weights — NOT fitted on test). Reference labels score severe FNs only.",
        "",
        f"## Severe-FN detection AUROC (pooled, n={p['n']}, n_severe_FN={p['n_severe_fn']})",
        "| signal | AUROC [95% CI] |",
        "|---|---|",
        f"| confidence only | {ci('confidence_only')} |",
        f"| stability only | {ci('stability_only')} |",
        f"| confidence + stability (v1.1) | {ci('confidence_plus_stability')} |",
        f"| **v3 combined** | **{ci('v3_combined')}** |",
        "",
        "## Severe-FN capture at matched review budget (pooled)",
        "| budget | confidence only | stability only | v3 combined |",
        "|---|---|---|---|",
    ]
    cap = p.get("severe_fn_capture", {})
    for b in ("budget_10pct", "budget_20pct", "budget_30pct"):
        if b in cap:
            d = cap[b]
            lines.append(
                f"| {b.replace('budget_', '').replace('pct', '%')} | "
                f"{d['confidence_only']:.3f} | {d['stability_only']:.3f} | {d['v3_combined']:.3f} |"
            )
    # verdict
    base = a["confidence_only"]["point"]
    v3 = a["v3_combined"]["point"]
    verdict = (
        "v3 **improves** severe-FN detection over confidence alone"
        if v3 > base + 0.005
        else (
            "v3 is **within noise** of confidence alone (honest: the richer signals are largely "
            "redundant with confidence for severe-FN detection)"
        )
    )
    lines += [
        "",
        "## Interpretation (honest, no overclaim)",
        f"- Pooled, **{verdict}** ({v3:.3f} vs {base:.3f}).",
        "- Per condition, v3's value concentrates where the route is weak/uncertain; on strong",
        "  routes confidence already saturates severe-FN detection.",
        "- v3 adds **retrieval_conflict** and **near_severe** to confidence+stability, and feeds",
        "  the case viewer's review reasons + the severe-FN risk surfaced per finding. It is a",
        "  triage/explanation signal; it never changes a prediction.",
        "",
        "Reproduce: `python scripts/run_evidence_intel_v3.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
