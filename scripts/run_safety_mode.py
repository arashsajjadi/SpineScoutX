#!/usr/bin/env python3
"""Phase 6 runner: severe-first Safety Mode on the AUTO distribution.

Applies the Safety Mode decision layer (operating points + abstention/review +
cost-weighted score) to the best robust grader and the control, evaluated on the auto
(real-inference) canal val set (C4). Writes outputs/real/safety_mode_frontier.json
(+ figure). Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation import safety_mode as sm
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
GAP = ROOT / "data/cache/rsna_gap2x2"
C4_MAN = GAP / "c4_autoxy_midslice" / "manifest.parquet"
C4_CACHE = GAP / "c4_autoxy_midslice"
RESULTS = ROOT / "outputs/real/robust_auto_experiments.json"
OUT_JSON = ROOT / "outputs/real/safety_mode_frontier.json"
OUT_FIG = ROOT / "outputs/real/figures/safety_mode_frontier.png"


def _pick_best() -> str:
    """Best robust variant by auto severe-recall point (excluding the control)."""
    r = json.loads(RESULTS.read_text())["variants"]
    cands = {k: v for k, v in r.items() if k != "r_oracle_ctrl"}
    return max(cands, key=lambda k: cands[k]["auto_severe_recall_point"])


def _safety_for(run_dir: Path, device, n_boot: int) -> dict:
    preds = collect_probs(run_dir, C4_MAN, C4_CACHE, device)
    keys = sorted(preds)
    y = np.array([preds[k][0] for k in keys])
    p = np.stack([preds[k][1] for k in keys])
    studies = np.array([k.split("|")[0] for k in keys])
    report = sm.safety_report(y, p, target_recalls=(0.90, 0.95))
    # bootstrap CI on the headline safety metric: recall@FAR<=10%
    ci = bs.bootstrap_ci(y, p, studies, bs.make_recall_at_far(0.10), n_boot=n_boot)
    report["recall_at_far10_ci"] = ci
    report["n"], report["n_severe"] = int(len(y)), int((y == 2).sum())
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="best robust run dir (default: pick from results)")
    ap.add_argument("--baseline", default="runs/r_oracle_ctrl")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    device = select_device("auto")

    best = args.run or f"runs/{_pick_best()}"
    print(f"[safety] best robust run = {best}; baseline = {args.baseline}")

    out = {
        "condition": "spinal_canal_stenosis",
        "distribution": "auto (real inference, C4)",
        "models": {
            "baseline_oracle_trained": _safety_for(ROOT / args.baseline, device, args.n_boot),
            "robust_best": _safety_for(ROOT / best, device, args.n_boot),
        },
        "robust_best_run": best,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"[safety] wrote {OUT_JSON}")
    _print(out)
    _figure(out)
    return 0


def _print(out: dict) -> None:
    for name, rep in out["models"].items():
        op = rep["operating_points"]
        rb90 = rep["review_burden"]["eff_recall>=0.9"]
        print(f"\n=== {name} (auto, n={rep['n']}, severe={rep['n_severe']}) ===")
        print(
            f"  balanced(argmax) severe recall = {op['balanced_argmax']['severe_recall']:.3f} "
            f"(FAR {op['balanced_argmax']['false_alarm_rate']:.3f})"
        )
        s90 = op["safety"].get("recall>=0.9", {})
        if s90.get("reached"):
            print(f"  safety: severe recall>=0.90 reachable at FAR={s90['false_alarm_rate']:.3f}")
        else:
            print("  safety: severe recall>=0.90 NOT reachable by thresholding alone")
        far10 = rep["recall_at_far10_ci"]
        print(
            f"  recall@FAR<=10% = {far10['point']:.3f} [{far10['ci_lo']:.3f},{far10['ci_hi']:.3f}]"
        )
        if rb90.get("reached"):
            print(
                f"  review for eff. severe recall>=0.90: abstain {rb90['abstain_rate']:.1%} "
                f"(captures {rb90['severe_fn_capture_frac']:.1%} of model severe-FNs)"
            )


def _figure(out: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[safety] figure skipped: {exc}")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rep in out["models"].items():
        c = rep["abstention_curve"]
        ax.plot(
            [r["abstain_rate"] for r in c],
            [r["effective_severe_recall_with_review"] for r in c],
            marker=".",
            label=name,
        )
    ax.set_xlabel("abstention / review rate")
    ax.set_ylabel("effective severe recall (with review)")
    ax.set_title("Safety Mode: severe recall vs review burden (auto) — research-only")
    ax.grid(alpha=0.3)
    ax.legend()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=120)
    print(f"[safety] wrote {OUT_FIG}")


if __name__ == "__main__":
    raise SystemExit(main())
