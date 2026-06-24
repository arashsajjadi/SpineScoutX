#!/usr/bin/env python3
"""Confirm v0.9 canal robust auto-inference under the LOCKED-TEST protocol.

Retrains the oracle-trained control and the auto-trained robust grader on
splits_v1 `train`, selects on `dev` (auto distribution), and evaluates ONCE on the
locked `test`. Reports oracle vs auto per split with cluster-bootstrap CIs and the
paired robust-vs-control delta. Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

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
AUTO = ROOT / "data/cache/rsna_auto_canal_all"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
E0_CFG = ROOT / "runs/e0_baseline_real/config.json"
LOC_RUN = ROOT / "runs/l0_disc_localizer_real"
OUT = ROOT / "outputs/real/canal_locked_test.json"
DOC = ROOT / "docs/run_logs/canal_locked_test.md"
COND = "spinal_canal_stenosis"


def _canal(man, split_map, split):
    m = man[(man.condition == COND) & (man.severity_index.isin([0, 1, 2]))].copy()
    m["study_id"] = m["study_id"].astype(str)
    m = m[m["study_id"].map(split_map) == split].reset_index(drop=True)
    return m


def _targets(m):
    return [int(t) for t in m["severity_index"].tolist()]


def _ci(run_dir, manifest_df, cache, device, tmp, n_boot):
    manifest_df.to_parquet(tmp)
    preds = collect_probs(run_dir, tmp, cache, device)
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
    tmpdir = Path(
        "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad"
    )
    tmpdir.mkdir(parents=True, exist_ok=True)

    split_map = load_splits_v1(SPLITS)
    oracle_man = read_manifest(ORACLE / "manifest.parquet")

    # auto crops for ALL canal studies (resume-safe)
    if not (AUTO / "manifest.parquet").exists():
        print("[canal-lt] generating canal auto crops for all studies...")
        from spinescoutx.data.auto_localize import prepare_rsna_auto_crops

        all_canal = sorted(oracle_man[oracle_man.condition == COND].study_id.astype(str).unique())
        rep = prepare_rsna_auto_crops(
            ROOT / "data/raw/rsna",
            LOC_RUN,
            AUTO,
            split="all",
            studies=all_canal,
            crop_size=224,
            use_25d=True,
        )
        print(f"[canal-lt] auto crops: {rep['n_auto_crops']} ({rep['skipped_studies']} skipped)")
    auto_man = read_manifest(AUTO / "manifest.parquet")

    # datasets per split
    or_tr, au_tr = _canal(oracle_man, split_map, "train"), _canal(auto_man, split_map, "train")
    au_dev = _canal(auto_man, split_map, "dev")
    print(f"[canal-lt] train: oracle {len(or_tr)} / auto {len(au_tr)} | dev auto {len(au_dev)}")

    dev_loader = DataLoader(
        RsnaCropDataset(au_dev, AUTO, crop_size=224, use_25d=True, guided=False),
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    variants = {
        "canal_oracle_ctrl": (or_tr, ORACLE),
        "canal_auto_robust": (au_tr, AUTO),
    }
    results = {
        "protocol": "splits_v1 locked-test",
        "condition": COND,
        "n_boot": args.n_boot,
        "selection": "dev auto severe_aware",
        "variants": {},
    }
    preds_cache = {}
    for name, (train_man, cache) in variants.items():
        print(f"\n========== {name} ==========")
        run_dir = ROOT / "runs" / f"v1_{name}"
        train_robust_variant(
            variant=f"v1_{name}",
            train_dataset=RsnaCropDataset(
                train_man, cache, crop_size=224, use_25d=True, guided=False
            ),
            train_targets=_targets(train_man),
            auto_val_loader=dev_loader,
            e0_config_path=E0_CFG,
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=32,
            num_workers=4,
            device="auto",
        )
        ev = {}
        for split in ("dev", "test"):
            for prov, src_man, cache_p in (
                ("oracle", oracle_man, ORACLE),
                ("auto", auto_man, AUTO),
            ):
                m = _canal(src_man, split_map, split)
                y, p, st, ci = _ci(run_dir, m, cache_p, device, tmpdir / "_lt.parquet", args.n_boot)
                ev[f"{split}_{prov}"] = {
                    "severe_recall": ci["severe_recall"],
                    "weighted_logloss": ci["weighted_logloss"],
                    "recall_at_far10": ci["recall_at_far10"],
                    "severe_auroc": ci["severe_auroc"],
                    "ece": ci["ece"],
                }
                if split == "test" and prov == "auto":
                    preds_cache[name] = (y, p, st)
        results["variants"][name] = ev
        t = ev["test_auto"]["severe_recall"]
        o = ev["test_oracle"]["severe_recall"]
        print(
            f"[{name}] test auto sevR={t['point']:.3f} [{t['ci_lo']:.3f},{t['ci_hi']:.3f}] | "
            f"test oracle={o['point']:.3f} [{o['ci_lo']:.3f},{o['ci_hi']:.3f}]"
        )

    # paired robust vs control on locked TEST auto (same nodes)
    yc, pc, stc = preds_cache["canal_auto_robust"]
    yb, pb, stb = preds_cache["canal_oracle_ctrl"]
    assert np.array_equal(yc, yb) and np.array_equal(stc, stb), "test nodes misaligned"
    results["paired_test_auto_robust_vs_ctrl"] = {
        "severe_recall": bs.paired_bootstrap_delta(
            yc, pc, pb, stc, bs.m_severe_recall, n_boot=args.n_boot
        ),
        "weighted_logloss": bs.paired_bootstrap_delta(
            yc, pc, pb, stc, bs.m_weighted_logloss, n_boot=args.n_boot
        ),
        "mcnemar": bs.mcnemar_severe(yc, np.argmax(pc, 1), np.argmax(pb, 1)),
        "n": int(len(yc)),
        "n_severe": int((yc == 2).sum()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    _doc(results)
    d = results["paired_test_auto_robust_vs_ctrl"]["severe_recall"]
    print(
        f"\n[canal-lt] LOCKED-TEST paired robust-vs-control auto severe recall: "
        f"{d['delta']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] decisive={d['decisive']}"
    )
    print(f"[canal-lt] wrote {OUT}")
    return 0


def _doc(r):
    v = r["variants"]
    p = r["paired_test_auto_robust_vs_ctrl"]

    def row(name, split_prov):
        c = v[name][split_prov]["severe_recall"]
        w = v[name][split_prov]["weighted_logloss"]
        return f"{c['point']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] | {w['point']:.3f}"

    def trow(label, name):
        sp = ("dev_oracle", "dev_auto", "test_oracle", "test_auto")
        return f"| {label} | " + " | ".join(row(name, s) for s in sp) + " |"

    mc = p["mcnemar"]

    lines = [
        "# Canal robust auto-inference — LOCKED-TEST confirmation (splits_v1)",
        "",
        "> Research-only. Not diagnostic. Models retrained on splits_v1 `train`, selected on",
        "> `dev` (auto, severe-aware), evaluated ONCE on the locked `test`. Cluster-bootstrap",
        f"> 95% CIs. Locked-test n={p['n']}, severe={p['n_severe']}.",
        "",
        "## Severe recall [95% CI] | weighted log loss",
        "",
        "| model | dev oracle | dev auto | **test oracle** | **test auto (real)** |",
        "|---|---|---|---|---|",
        trow("oracle-trained control", "canal_oracle_ctrl"),
        trow("**auto-trained robust**", "canal_auto_robust"),
        "",
        "## Paired robust − control on locked test (auto, same nodes)",
        f"- severe recall Δ **{p['severe_recall']['delta']:+.3f}** "
        f"[{p['severe_recall']['ci_lo']:+.3f}, {p['severe_recall']['ci_hi']:+.3f}] "
        f"(decisive={p['severe_recall']['decisive']})",
        f"- weighted log loss Δ {p['weighted_logloss']['delta']:+.3f} "
        f"[{p['weighted_logloss']['ci_lo']:+.3f}, {p['weighted_logloss']['ci_hi']:+.3f}] "
        f"(decisive={p['weighted_logloss']['decisive']}; negative = better)",
        f"- McNemar severe (robust-catches / control-catches): {mc['b_a_catches_b_misses']} /"
        f" {mc['c_a_misses_b_catches']}, p={mc['p_value']:.4g}",
        "",
        "Artifacts: `outputs/real/canal_locked_test.json`. Reproduce:",
        "`python scripts/run_canal_locked_test.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
