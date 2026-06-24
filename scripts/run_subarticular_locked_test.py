#!/usr/bin/env python3
"""Axial subarticular auto route, locked test (splits_v1).

Generates auto subarticular crops (coordinate-supervised axial level scorer + fixed
paramedian offset; no GT at inference), trains oracle-trained control vs auto-trained
robust graders (side-aware), selects on dev auto, evaluates ONCE on locked test per side
with cluster-bootstrap CIs + paired deltas. Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from spinescoutx.data.axial_level import SUBARTICULAR, prepare_rsna_subarticular_auto_crops
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.datasets import RsnaCropDataset
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device
from spinescoutx.training.train_robust import train_robust_variant

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
ORACLE = ROOT / "data/cache/rsna"
AUTO = ROOT / "data/cache/rsna_auto_subarticular"
SCORER = ROOT / "runs/axial_level_scorer"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
E0_CFG = ROOT / "runs/e0_baseline_real/config.json"
OUT = ROOT / "outputs/real/subarticular_auto_results.json"
DOC = ROOT / "docs/run_logs/subarticular_auto_results.md"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_sub.parquet"
)


def _sel(man, split_map, split, cond=None):
    m = man[man.severity_index.isin([0, 1, 2])].copy()
    m["study_id"] = m["study_id"].astype(str)
    m = m[m["study_id"].map(split_map) == split]
    if cond is not None:
        m = m[m.condition == cond]
    elif "condition" in m.columns:
        m = m[m.condition.isin(SUBARTICULAR)]
    return m.reset_index(drop=True)


def _eval(run_dir, man, cache, device, n_boot):
    man.to_parquet(TMP)
    preds = collect_probs(run_dir, TMP, cache, device)
    keys = sorted(preds)
    y = np.array([preds[k][0] for k in keys])
    p = np.stack([preds[k][1] for k in keys])
    st = np.array([k.split("|")[0] for k in keys])
    return y, p, st, bs.ci_table(y, p, st, n_boot=n_boot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)

    if not (AUTO / "manifest.parquet").exists():
        print("[sub] generating auto subarticular crops (scorer + fixed offset)...")
        rep = prepare_rsna_subarticular_auto_crops(
            ROOT / "data/raw/rsna", SCORER, AUTO, crop_size=224
        )
        print(f"[sub] auto crops={rep['n_auto_crops']} skipped={rep['skipped']}")
    oracle_man = read_manifest(ORACLE / "manifest.parquet")
    auto_man = read_manifest(AUTO / "manifest.parquet")

    dev_loader = DataLoader(
        RsnaCropDataset(
            _sel(auto_man, split_map, "dev"), AUTO, crop_size=224, use_25d=True, guided=False
        ),
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )
    variants = {
        "subarticular_oracle_ctrl": (ORACLE, oracle_man),
        "subarticular_auto_robust": (AUTO, auto_man),
    }
    results = {
        "protocol": "splits_v1 locked-test",
        "conditions": list(SUBARTICULAR),
        "n_boot": args.n_boot,
        "variants": {},
    }
    preds = {}
    for name, (cache, man) in variants.items():
        print(f"\n========== {name} ==========")
        run_dir = ROOT / "runs" / f"v1_{name}"
        tr = _sel(man, split_map, "train")
        train_robust_variant(
            variant=f"v1_{name}",
            train_dataset=RsnaCropDataset(tr, cache, crop_size=224, use_25d=True, guided=False),
            train_targets=[int(t) for t in tr.severity_index.tolist()],
            auto_val_loader=dev_loader,
            e0_config_path=E0_CFG,
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=32,
            num_workers=4,
            device="auto",
        )
        ev = {}
        for cond in SUBARTICULAR:
            y, p, st, ci = _eval(
                run_dir, _sel(auto_man, split_map, "test", cond), AUTO, device, args.n_boot
            )
            _, _, _, cio = _eval(
                run_dir, _sel(oracle_man, split_map, "test", cond), ORACLE, device, args.n_boot
            )
            ev[cond] = {
                "test_auto": {
                    k: ci[k]
                    for k in (
                        "severe_recall",
                        "weighted_logloss",
                        "recall_at_far10",
                        "severe_auroc",
                        "ece",
                    )
                },
                "test_oracle_severe_recall": cio["severe_recall"],
                "n": int(len(y)),
                "n_severe": int((y == 2).sum()),
            }
            preds[(name, cond)] = (y, p, st)
        results["variants"][name] = ev

    results["paired_robust_vs_ctrl_auto"] = {}
    for cond in SUBARTICULAR:
        yr, pr, sr = preds[("subarticular_auto_robust", cond)]
        yc, pc, sc = preds[("subarticular_oracle_ctrl", cond)]
        if np.array_equal(yr, yc) and np.array_equal(sr, sc):
            results["paired_robust_vs_ctrl_auto"][cond] = {
                "severe_recall": bs.paired_bootstrap_delta(
                    yr, pr, pc, sr, bs.m_severe_recall, n_boot=args.n_boot
                ),
                "mcnemar": bs.mcnemar_severe(yr, np.argmax(pr, 1), np.argmax(pc, 1)),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    _print(results)
    return 0


def _print(r):
    print("\n=== SUBARTICULAR locked-test auto severe recall [95% CI] ===")
    for cond in SUBARTICULAR:
        for g in ("subarticular_oracle_ctrl", "subarticular_auto_robust"):
            e = r["variants"][g][cond]
            sr = e["test_auto"]["severe_recall"]
            o = e["test_oracle_severe_recall"]["point"]
            print(
                f"  {cond:28s} {g:24s} auto {sr['point']:.3f} "
                f"[{sr['ci_lo']:.3f},{sr['ci_hi']:.3f}] | oracle {o:.3f} (sev={e['n_severe']})"
            )


if __name__ == "__main__":
    raise SystemExit(main())
