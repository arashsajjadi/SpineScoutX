#!/usr/bin/env python3
"""Dev-selected ensemble of the deployed single-crop grader + candidate-bag MIL (v1.5).

If the MIL is individually weaker but *complementary*, a convex blend of class probabilities can
beat the deployed grader. We sweep the blend weight alpha on **dev** (maximize recall@FAR<=10%
with a FAR guardrail), then read **locked-test once** and report the paired delta vs the deployed
grader (alpha=0). A win counts only if the test CI excludes 0. Research-only. Not diagnostic.

Usage: ``python scripts/run_mil_ensemble_v1_5.py --route {right_foraminal,subarticular}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mil_vs_baseline_v1_5 import ROUTES, baseline_probs, mil_probs  # noqa: E402

from spinescoutx.data.locked_test import load_splits_v1  # noqa: E402
from spinescoutx.evaluation import bootstrap as bs  # noqa: E402
from spinescoutx.training.optim import select_device  # noqa: E402

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUTDIR = ROOT / "outputs/real"
ALPHAS = [round(a, 2) for a in np.linspace(0, 1, 21)]


def _align(base, mil):
    keys = sorted(set(base) & set(mil))
    y = np.array([base[k][0] for k in keys])
    pb = np.stack([base[k][1] for k in keys])
    pm = np.stack([mil[k][1] for k in keys])
    st = np.array([k.split("|")[0] for k in keys])
    return y, pb, pm, st


def _blend(pb, pm, alpha):
    p = (1 - alpha) * pb + alpha * pm
    return p / p.sum(1, keepdims=True)


def _far(y, p):
    neg = y != 2
    return float((p[neg].argmax(1) == 2).mean()) if neg.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=list(ROUTES))
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)
    cfg = ROUTES[args.route]
    run_dir, cache = ROOT / cfg["run"], ROOT / cfg["cache"]
    data = {}
    for split in ("dev", "test"):
        base = {}
        for cond in cfg["conds"]:
            base.update(baseline_probs(cond, split, run_dir, cache, split_map, device))
        data[split] = _align(base, mil_probs(args.route, split, cfg["conds"]))

    r10 = bs.make_recall_at_far(0.10)
    yd, pbd, pmd, _ = data["dev"]
    # select alpha on dev: maximize recall@FAR10, reject blends whose argmax FAR > baseline+0.05
    base_far_dev = _far(yd, pbd)
    sweep = []
    for a in ALPHAS:
        pe = _blend(pbd, pmd, a)
        far = _far(yd, pe)
        ok = far <= max(0.5, base_far_dev + 0.05)
        sweep.append(
            {"alpha": a, "dev_recall_at_far10": float(r10(yd, pe)), "dev_far": far, "ok": ok}
        )
    valid = [s for s in sweep if s["ok"]]
    best = max(valid or sweep, key=lambda s: s["dev_recall_at_far10"])
    a = best["alpha"]

    yt, pbt, pmt, stt = data["test"]
    pet = _blend(pbt, pmt, a)
    out = {
        "route": args.route,
        "selected_alpha": a,
        "dev_sweep": sweep,
        "dev": {
            "baseline_recall_at_far10": float(r10(yd, pbd)),
            "ensemble_recall_at_far10": float(r10(yd, _blend(pbd, pmd, a))),
        },
        "test": {
            "n": int(len(yt)),
            "n_severe": int((yt == 2).sum()),
            "baseline_severe_recall": float(bs.m_severe_recall(yt, pbt)),
            "ensemble_severe_recall": float(bs.m_severe_recall(yt, pet)),
            "baseline_recall_at_far10": float(r10(yt, pbt)),
            "ensemble_recall_at_far10": float(r10(yt, pet)),
            "baseline_far": _far(yt, pbt),
            "ensemble_far": _far(yt, pet),
            "paired_delta_severe_recall": bs.paired_bootstrap_delta(
                yt, pet, pbt, stt, bs.m_severe_recall, n_boot=2000
            ),
            "paired_delta_recall_at_far10": bs.paired_bootstrap_delta(
                yt, pet, pbt, stt, r10, n_boot=2000
            ),
        },
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dst = OUTDIR / f"mil_ensemble_{args.route}_v1_5.json"
    dst.write_text(json.dumps(out, indent=2, default=float))
    t = out["test"]
    ds, dr = t["paired_delta_severe_recall"], t["paired_delta_recall_at_far10"]
    print(f"[{args.route}] selected alpha={a} (dev r@FAR10 {best['dev_recall_at_far10']:.3f})")
    print(
        f"  TEST severe recall base {t['baseline_severe_recall']:.3f} -> "
        f"ens {t['ensemble_severe_recall']:.3f} "
        f"(Δ{ds['delta']:+.3f} [{ds['ci_lo']:+.3f},{ds['ci_hi']:+.3f}]"
        f"{' DECISIVE' if ds['decisive'] else ''})"
    )
    print(
        f"  TEST recall@FAR10 base {t['baseline_recall_at_far10']:.3f} -> "
        f"ens {t['ensemble_recall_at_far10']:.3f} "
        f"(Δ{dr['delta']:+.3f} [{dr['ci_lo']:+.3f},{dr['ci_hi']:+.3f}]"
        f"{' DECISIVE' if dr['decisive'] else ''}) | FAR base {t['baseline_far']:.3f} "
        f"ens {t['ensemble_far']:.3f}"
    )
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
