#!/usr/bin/env python3
"""Paired baseline (deployed single-crop grader) vs candidate-bag MIL (v1.5).

For a route, align the deployed grader and the trained MIL on the **same** held-out findings
(``study|level|condition``), then report severe recall + recall@FAR<=10% for each with cluster
(study-level) bootstrap CIs and the **paired** delta (MIL - baseline). Dev = go/no-go;
locked-test = headline (read once for the dev-selected model). Research-only. Not diagnostic.

Usage: ``python scripts/compare_mil_vs_baseline_v1_5.py --route {right_foraminal,subarticular}``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUTDIR = ROOT / "outputs/real"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_cmp.parquet"
)
ROUTES = {
    "right_foraminal": {
        "conds": ["right_neural_foraminal_narrowing"],
        "run": "runs/v1_foraminal_oracle_ctrl",
        "cache": "data/cache/rsna_auto_foraminal",
    },
    "left_foraminal": {
        "conds": ["left_neural_foraminal_narrowing"],
        "run": "runs/v1_foraminal_oracle_ctrl",
        "cache": "data/cache/rsna_auto_foraminal",
    },
    "subarticular": {
        "conds": ["left_subarticular_stenosis", "right_subarticular_stenosis"],
        "run": "runs/v1_subarticular_auto_robust",
        "cache": "data/cache/rsna_auto_subarticular",
    },
}


def baseline_probs(cond, split, run_dir, cache, split_map, device):
    """Deployed single-crop grader probs for one condition+split, keyed study|level|condition."""
    man = read_manifest(cache / "manifest.parquet")
    man = man[(man.condition == cond) & man.severity_index.isin([0, 1, 2])].copy()
    man["study_id"] = man.study_id.astype(str)
    sub = man[man.study_id.map(split_map) == split].reset_index(drop=True)
    if sub.empty:
        return {}
    sub.to_parquet(TMP)
    preds = collect_probs(run_dir, TMP, cache, device)  # {study|level: (y, p[3])}
    out = {}
    for k, (y, p) in preds.items():
        st, lv = k.split("|")[0], k.split("|")[1]
        out[f"{st}|{lv}|{cond}"] = (int(y), np.asarray(p, dtype=float))
    return out


def mil_probs(route, split, conds):
    """Trained-MIL probs from the dumped per-finding parquet, keyed study|level|condition."""
    pq = OUTDIR / f"mil_{route}_v1_5_preds.parquet"
    if not pq.exists():
        raise SystemExit(f"missing MIL preds {pq}; run train_mil_grader_v1_5.py first")
    df = pd.read_parquet(pq)
    df = df[(df.split == split) & (df.condition.isin(conds))]
    out = {}
    for r in df.itertuples():
        out[str(r.key)] = (
            int(r.y),
            np.array([r.p_normal, r.p_moderate, r.p_severe], dtype=float),
        )
    return out


def _far(y, p):
    neg = y != 2
    return float((p[neg].argmax(1) == 2).mean()) if neg.any() else float("nan")


def _metrics(y, p, st):
    r10 = bs.make_recall_at_far(0.10)
    return {
        "severe_recall": float(bs.m_severe_recall(y, p)),
        "severe_recall_ci": bs.bootstrap_ci(y, p, st, bs.m_severe_recall, n_boot=2000),
        "recall_at_far10": float(r10(y, p)),
        "recall_at_far10_ci": bs.bootstrap_ci(y, p, st, r10, n_boot=2000),
        "far": _far(y, p),
        "n": int(len(y)),
        "n_severe": int((y == 2).sum()),
    }


def _compare_split(base, mil, split):
    keys = sorted(set(base) & set(mil))
    if not keys:
        return {"split": split, "error": "no overlapping keys"}
    y = np.array([base[k][0] for k in keys])
    ym = np.array([mil[k][0] for k in keys])
    label_mismatch = int((y != ym).sum())  # labels must match (same finding)
    pb = np.stack([base[k][1] for k in keys])
    pm = np.stack([mil[k][1] for k in keys])
    st = np.array([k.split("|")[0] for k in keys])
    r10 = bs.make_recall_at_far(0.10)
    d_sr = bs.paired_bootstrap_delta(y, pm, pb, st, bs.m_severe_recall, n_boot=2000)
    d_r10 = bs.paired_bootstrap_delta(y, pm, pb, st, r10, n_boot=2000)
    mc = bs.mcnemar_severe(y, pm.argmax(1) == 2, pb.argmax(1) == 2)
    return {
        "split": split,
        "n_common": len(keys),
        "n_base_only": len(set(base) - set(mil)),
        "n_mil_only": len(set(mil) - set(base)),
        "label_mismatch": label_mismatch,
        "baseline": _metrics(y, pb, st),
        "mil": _metrics(y, pm, st),
        "paired_delta_severe_recall": d_sr,
        "paired_delta_recall_at_far10": d_r10,
        "mcnemar_severe_argmax": mc,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=list(ROUTES))
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)
    cfg = ROUTES[args.route]
    run_dir, cache = ROOT / cfg["run"], ROOT / cfg["cache"]
    out = {
        "route": args.route,
        "conditions": cfg["conds"],
        "protocol": "paired baseline(single-crop) vs MIL(K candidates) on identical findings",
        "splits": {},
    }
    for split in ("dev", "test"):
        base, mil = {}, {}
        for cond in cfg["conds"]:
            base.update(baseline_probs(cond, split, run_dir, cache, split_map, device))
        mil = mil_probs(args.route, split, cfg["conds"])
        out["splits"][split] = _compare_split(base, mil, split)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dst = OUTDIR / f"compare_mil_vs_baseline_{args.route}_v1_5.json"
    dst.write_text(json.dumps(out, indent=2, default=float))
    for split, r in out["splits"].items():
        if "error" in r:
            print(f"[{split}] {r['error']}")
            continue
        b, m = r["baseline"], r["mil"]
        ds, dr = r["paired_delta_severe_recall"], r["paired_delta_recall_at_far10"]
        print(
            f"[{split}] n={r['n_common']} (sev {b['n_severe']}) | "
            f"severe recall base {b['severe_recall']:.3f} -> MIL {m['severe_recall']:.3f} "
            f"(Δ{ds['delta']:+.3f} [{ds['ci_lo']:+.3f},{ds['ci_hi']:+.3f}]"
            f"{' DECISIVE' if ds['decisive'] else ''})"
        )
        print(
            f"        recall@FAR10 base {b['recall_at_far10']:.3f} -> "
            f"MIL {m['recall_at_far10']:.3f} "
            f"(Δ{dr['delta']:+.3f} [{dr['ci_lo']:+.3f},{dr['ci_hi']:+.3f}]"
            f"{' DECISIVE' if dr['decisive'] else ''}) | FAR base {b['far']:.3f} MIL {m['far']:.3f}"
        )
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
