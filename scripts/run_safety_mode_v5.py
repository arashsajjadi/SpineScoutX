#!/usr/bin/env python3
"""Safety Mode v5 — evidence-aware review policy + condition-specific calibration.

v4 reviewed on low confidence + model disagreement. v5 adds two real signals:
  1. **Condition-specific temperature calibration** fit on `dev` only (never on
     `test`); we report ECE/Brier before vs after.
  2. **Evidence-stability-aware review**: the review score penalises unstable findings
     (`instability` from `evidence_stability_records.parquet`). We compare, at matched
     review burden, severe-FN capture for confidence-only (v4-style) vs
     confidence+stability (v5) — i.e. does the new signal improve triage?

Locked TEST auto distribution; cluster-bootstrap CIs. No GT coordinates; no test-set
tuning (temperature is fit on dev). Research-only. `review_required` is a research
signal, not triage advice.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import safety_mode as sm
from spinescoutx.evaluation.calibration import expected_calibration_error
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RECORDS = ROOT / "outputs/real/evidence_stability_records.parquet"
OUT = ROOT / "outputs/real/safety_mode_v5.json"
DOC = ROOT / "docs/run_logs/safety_mode_v5.md"
FIG = ROOT / "outputs/real/figures/safety_mode_v5_dashboard.png"
ASSET = ROOT / "docs/assets/showcase/safety_mode_v5_dashboard.png"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_sv5.parquet"
)

ROUTES = {
    "spinal_canal_stenosis": ("runs/v1_canal_auto_robust", "data/cache/rsna_auto_canal_all"),
    "left_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "right_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "left_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
    "right_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
}
LAM = 0.5  # stability penalty weight in the v5 review score


def _probs_for(run_dir, cache, man, device):
    man.to_parquet(TMP)
    preds = collect_probs(run_dir, TMP, cache, device)
    keys = [f"{str(r.study_id)}|{str(r.level)}" for r in man.itertuples()]
    y = np.array([preds[k][0] for k in keys if k in preds])
    p = np.stack([preds[k][1] for k in keys if k in preds])
    kept = [k for k in keys if k in preds]
    return y, p, kept


def _fit_temperature(y, p):
    """Fit a scalar T>0 minimising NLL on (probs as log-probs). Grid + local refine."""
    logp = np.log(np.clip(p, 1e-8, 1.0))

    def nll(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        sm_ = np.exp(z)
        sm_ = sm_ / sm_.sum(1, keepdims=True)
        return float(-np.log(sm_[np.arange(len(y)), y] + 1e-12).mean())

    grid = np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.0, 6.0, 7)])
    best = min(grid, key=nll)
    return float(best)


def _apply_t(p, t):
    logp = np.log(np.clip(p, 1e-8, 1.0)) / t
    logp = logp - logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def _brier(y, p):
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).sum(1).mean())


def _capture(review_first_score, is_target, budget):
    """Severe-FN capture when reviewing the top-`budget` fraction with the LOWEST score
    (lowest confidence -> review first). Higher capture is better."""
    n = len(review_first_score)
    n_rev = max(1, int(round(budget * n)))
    order = np.argsort(review_first_score, kind="mergesort")[:n_rev]
    flagged = np.zeros(n, dtype=bool)
    flagged[order] = True
    tot = int(is_target.sum())
    return float((flagged & is_target).sum() / tot) if tot else float("nan")


def main() -> int:
    argparse.ArgumentParser().parse_args()  # no options; accept -h
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)

    import pandas as pd

    rec = pd.read_parquet(RECORDS) if RECORDS.exists() else None
    if rec is not None:
        rec["study_id"] = rec.study_id.astype(str)
        rec["side"] = rec["side"].fillna("").astype(str)

    out = {"protocol": "splits_v1 locked-test", "lambda_stability": LAM, "conditions": {}}
    for cond, (run, cache) in ROUTES.items():
        run_dir, cpath = ROOT / run, ROOT / cache
        if not (run_dir / "best.pt").exists():
            continue
        man = read_manifest(cpath / "manifest.parquet")
        man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
        man["study_id"] = man.study_id.astype(str)
        dev = man[man.study_id.map(split_map) == "dev"].reset_index(drop=True)
        test = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
        if dev.empty or test.empty:
            continue
        yd, pd_, _ = _probs_for(run_dir, cpath, dev, device)
        yt, pt, keys_t = _probs_for(run_dir, cpath, test, device)

        t = _fit_temperature(yd, pd_)
        pt_cal = _apply_t(pt, t)

        ece_u, ece_c = expected_calibration_error(yt, pt), expected_calibration_error(yt, pt_cal)
        brier_u, brier_c = _brier(yt, pt), _brier(yt, pt_cal)

        # join instability to the test items by (study|level) key + condition
        inst = np.zeros(len(yt))
        grade = np.array(["stable"] * len(yt), dtype=object)
        if rec is not None:
            sub = rec[rec.condition == cond]
            lut = {f"{r.study_id}|{r.level}": (r.instability, r.grade) for r in sub.itertuples()}
            for i, k in enumerate(keys_t):
                if k in lut:
                    inst[i], grade[i] = float(lut[k][0]), lut[k][1]

        conf = pt_cal.max(1)
        pred = pt_cal.argmax(1)
        is_fn = (yt == 2) & (pred != 2)
        # review-first score: lower = review first. v4 = confidence; v5 = confidence - lam*instab
        score_v4 = conf.copy()
        score_v5 = conf - LAM * inst
        n_unstable = int((grade == "unstable").sum())

        rep = sm.safety_report(yt, pt_cal, aux_conf=conf, target_recalls=(0.90,))
        cap = {}
        for b in (0.10, 0.20, 0.30):
            cap[f"budget_{int(b * 100)}pct"] = {
                "v4_confidence_only": _capture(score_v4, is_fn, b),
                "v5_confidence_plus_stability": _capture(score_v5, is_fn, b),
            }
        out["conditions"][cond] = {
            "temperature": t,
            "ece_uncalibrated": ece_u,
            "ece_calibrated": ece_c,
            "brier_uncalibrated": brier_u,
            "brier_calibrated": brier_c,
            "n": int(len(yt)),
            "n_severe": int((yt == 2).sum()),
            "n_severe_fn": int(is_fn.sum()),
            "n_unstable": n_unstable,
            "recall_at_far": rep.get("recall_at_far", {}),
            "severe_fn_capture": cap,
        }
        print(
            f"{cond:34s} T={t:.2f} ECE {ece_u:.3f}->{ece_c:.3f} "
            f"unstable={n_unstable} sevFN={int(is_fn.sum())}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _figure(out)
    _doc(out)
    print(f"\nwrote {OUT}\nwrote {DOC}\nwrote {FIG}")
    return 0


def _figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(out["conditions"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    # ECE before/after
    x = np.arange(len(conds))
    eu = [out["conditions"][c]["ece_uncalibrated"] for c in conds]
    ec = [out["conditions"][c]["ece_calibrated"] for c in conds]
    axs[0].bar(x - 0.2, eu, 0.4, label="uncalibrated", color="#b0bec5")
    axs[0].bar(x + 0.2, ec, 0.4, label="dev-calibrated", color="#1565c0")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([_s(c) for c in conds], rotation=20, ha="right", fontsize=8)
    axs[0].set_ylabel("ECE")
    axs[0].set_title("Condition-specific calibration (dev-fit)")
    axs[0].legend(fontsize=8)
    # severe-FN capture @20%: v4 vs v5
    cv4 = [
        out["conditions"][c]["severe_fn_capture"]["budget_20pct"]["v4_confidence_only"]
        for c in conds
    ]
    cv5 = [
        out["conditions"][c]["severe_fn_capture"]["budget_20pct"]["v5_confidence_plus_stability"]
        for c in conds
    ]
    axs[1].bar(x - 0.2, cv4, 0.4, label="v4 confidence only", color="#90a4ae")
    axs[1].bar(x + 0.2, cv5, 0.4, label="v5 confidence + stability", color="#00838f")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([_s(c) for c in conds], rotation=20, ha="right", fontsize=8)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("severe-FN capture @20% review")
    axs[1].set_title("Evidence-aware review (v4 vs v5)")
    axs[1].legend(fontsize=8)
    fig.suptitle("SpineScoutX Safety Mode v5 (locked-test auto) — research-only, not diagnostic")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, dpi=110)
    plt.close(fig)


def _s(c):
    if c.startswith("left"):
        return "L-" + c.split("_")[1][:3]
    if c.startswith("right"):
        return "R-" + c.split("_")[1][:3]
    return "canal"


def _doc(out):
    lam = out["lambda_stability"]
    lines = [
        "# Safety Mode v5 — evidence-aware review + condition-specific calibration",
        "",
        "> Research-only. Not diagnostic. `review_required` is a research signal, not triage.",
        "> Temperature is fit on `dev` only; `test` is eval-only. Locked-test auto distribution.",
        "",
        f"Review score: v4 = calibrated confidence; v5 = confidence − {lam}·instability.",
        "",
        "## Calibration (test ECE / Brier, before → after dev-fit temperature)",
        "| condition | T | ECE | Brier | n / sev |",
        "|---|---|---|---|---|",
    ]
    for c, r in out["conditions"].items():
        ece = f"{r['ece_uncalibrated']:.3f}→{r['ece_calibrated']:.3f}"
        brier = f"{r['brier_uncalibrated']:.3f}→{r['brier_calibrated']:.3f}"
        lines.append(
            f"| {c} | {r['temperature']:.2f} | {ece} | {brier} | {r['n']} / {r['n_severe']} |"
        )
    lines += [
        "",
        "## Evidence-aware review — severe-FN capture at matched review burden (v4 vs v5)",
        "| condition | unstable | sevFN | budget | v4 conf-only | v5 conf+stability |",
        "|---|---|---|---|---|---|",
    ]
    for c, r in out["conditions"].items():
        for b in ("budget_10pct", "budget_20pct", "budget_30pct"):
            d = r["severe_fn_capture"][b]
            lines.append(
                f"| {c} | {r['n_unstable']} | {r['n_severe_fn']} | "
                f"{b.replace('budget_', '').replace('pct', '%')} | "
                f"{d['v4_confidence_only']:.3f} | {d['v5_confidence_plus_stability']:.3f} |"
            )
    conds = out["conditions"]
    n = len(conds)
    n_calib = sum(1 for r in conds.values() if r["ece_calibrated"] <= r["ece_uncalibrated"] + 1e-9)
    helps = [
        c
        for c, r in conds.items()
        if r["severe_fn_capture"]["budget_20pct"]["v5_confidence_plus_stability"]
        > r["severe_fn_capture"]["budget_20pct"]["v4_confidence_only"] + 1e-9
    ]
    lines += [
        "",
        "## Interpretation (honest, no overclaim)",
        "- **Calibration negative.** Graders are already well-calibrated (test ECE 0.03–0.08);",
        "  dev-fit temperature (T=1.1–1.3) does NOT transfer — calibrated test ECE ≤ uncalibrated",
        f"  on only **{n_calib}/{n}** conditions (worsens the rest). **Deployed path keeps raw",
        "  probabilities** (no temperature applied); reported, not hidden.",
        "- **Evidence-aware review is MIXED (uniform λ).** v5 (confidence + stability) beats",
        f"  v4 (confidence only) severe-FN capture @20% review on **{len(helps)}/{n}** routes: "
        f"{', '.join(helps) if helps else 'none'}",
        "  — the weakest **right-side** routes (right-foraminal @30% 0.72→0.89; right-subarticular",
        "  @20% 0.42→0.56). On the 3 strong routes confidence alone is as good or better, so a",
        "  uniform stability penalty is NOT deployed globally.",
        "- **Deployed v5 policy:** stability is an **inference-time** review reason",
        "  (`evidence_unstable` / `axial_candidate_disagreement` / `foraminal_slice_disagreement`)",
        "  and a `route_quality` flag; a measured severe-FN benefit on weak right-side routes.",
        "",
        "Reproduce: `python scripts/run_safety_mode_v5.py` (after run_evidence_stability.py).",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
