#!/usr/bin/env python3
"""Phase 2+3+4 runner: localizer error profile, then robust auto-crop training.

Trains E0-architecture canal graders under several "train==inference" regimes
(localizer-aware crop jitter, auto-localized crops, mixing, consistency), selecting
and reporting on the AUTO distribution (the real inference target) with bootstrap CIs
and oracle->auto gap-recovery. Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.datasets import RsnaCropDataset
from spinescoutx.data.robust_crops import (
    CropJitterConfig,
    JitterSampler,
    RobustCanalCropDataset,
    build_canal_slice_cache,
    build_localizer_error_profile,
)
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device
from spinescoutx.training.train_robust import train_robust_variant

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
GAP = ROOT / "data/cache/rsna_gap2x2"
SLICE_CACHE = ROOT / "data/cache/rsna_canal_slices"
AUTO_TRAIN = ROOT / "data/cache/rsna_auto_train"
E0_CFG = ROOT / "runs/e0_baseline_real/config.json"
LOC_RUN = ROOT / "runs/l0_disc_localizer_real"
OUT_JSON = ROOT / "outputs/real/robust_auto_experiments.json"
OUT_PROFILE = ROOT / "outputs/real/localizer_error_profile.json"
RUNS = ROOT / "runs"

ORACLE_BASELINE = {"severe_recall": 0.828, "weighted_logloss": 0.326}  # C1 (established)


def _targets(df: pd.DataFrame) -> list[int]:
    return [int(t) for t in df["severity_index"].tolist()]


def _auto_val_loader(crop_size: int, batch_size: int) -> DataLoader:
    man = read_manifest(GAP / "c4_autoxy_midslice" / "manifest.parquet")
    ds = RsnaCropDataset(
        man, GAP / "c4_autoxy_midslice", crop_size=crop_size, use_25d=True, guided=False
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def _eval_on(run_dir: Path, manifest: Path, cache: Path, device, n_boot: int) -> dict:
    preds = collect_probs(run_dir, manifest, cache, device)
    keys = sorted(preds)
    y = np.array([preds[k][0] for k in keys])
    p = np.stack([preds[k][1] for k in keys])
    studies = np.array([k.split("|")[0] for k in keys])
    tab = bs.ci_table(y, p, studies, n_boot=n_boot)
    return {
        "y": y,
        "p": p,
        "studies": studies,
        "ci": tab,
        "severe_recall": bs.m_severe_recall(y, p),
        "weighted_logloss": bs.m_weighted_logloss(y, p),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = select_device("auto")
    crop_size, batch_size = 224, 32
    epochs = 2 if args.smoke else args.epochs

    # ---- Phase 2: localizer error profile (from the 2x2 cells) ------------- #
    profile = build_localizer_error_profile(
        GAP / "c1_gtxy_gtslice" / "manifest.parquet",
        GAP / "c2_autoxy_gtslice" / "manifest.parquet",
        GAP / "c3_gtxy_midslice" / "manifest.parquet",
    )
    OUT_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE.write_text(json.dumps(profile.summary(), indent=2))
    print(f"[robust] wrote {OUT_PROFILE}")
    print(json.dumps(profile.summary()["pooled"], indent=2))

    # ---- source slices + auto-train crops (resume-safe) ------------------- #
    if not (SLICE_CACHE / "slice_cache_summary_train.json").exists():
        print("[robust] building canal train slice cache (one-time)...")
        s = build_canal_slice_cache(
            ROOT / "data/raw/rsna", ROOT / "data/cache/rsna", SLICE_CACHE, split="train", window=3
        )
        print(f"[robust] slice cache: {s['decoded_slices']} decoded, {s['n_nodes']} nodes")
    canal_train_nodes = pd.read_parquet(SLICE_CACHE / "canal_train_nodes.parquet")

    if not (AUTO_TRAIN / "manifest.parquet").exists():
        print("[robust] generating auto-localized TRAIN crops (no GT coords)...")
        from spinescoutx.data.auto_localize import prepare_rsna_auto_crops

        rep = prepare_rsna_auto_crops(
            ROOT / "data/raw/rsna",
            LOC_RUN,
            AUTO_TRAIN,
            split="train",
            crop_size=crop_size,
            use_25d=True,
        )
        print(f"[robust] auto-train: {rep['n_auto_crops']} crops, {rep['skipped_studies']} skipped")
    auto_train_man = read_manifest(AUTO_TRAIN / "manifest.parquet")
    auto_train_man = auto_train_man[auto_train_man.severity_index.isin([0, 1, 2])].reset_index(
        drop=True
    )

    if args.smoke:
        canal_train_nodes = canal_train_nodes.head(400)
        auto_train_man = auto_train_man.head(400)

    # ---- variant -> training dataset factory ------------------------------ #
    prof_for_jit = profile

    def ds_oracle():  # control: re-crop at GT centre (no jitter)
        j = JitterSampler(CropJitterConfig(mode="none"))
        return RobustCanalCropDataset(
            canal_train_nodes, SLICE_CACHE, crop_size=crop_size, jitter=j
        ), _targets(canal_train_nodes)

    def ds_jitter_level():
        j = JitterSampler(
            CropJitterConfig(mode="level_aware", tail_prob=0.15, tail_sigma=20.0), prof_for_jit
        )
        return RobustCanalCropDataset(
            canal_train_nodes, SLICE_CACHE, crop_size=crop_size, jitter=j
        ), _targets(canal_train_nodes)

    def ds_jitter_emp():
        j = JitterSampler(CropJitterConfig(mode="empirical"), prof_for_jit)
        return RobustCanalCropDataset(
            canal_train_nodes, SLICE_CACHE, crop_size=crop_size, jitter=j
        ), _targets(canal_train_nodes)

    def ds_auto():
        return RsnaCropDataset(
            auto_train_man, AUTO_TRAIN, crop_size=crop_size, use_25d=True, guided=False
        ), _targets(auto_train_man)

    def ds_mixed():
        oracle_man = read_manifest(ROOT / "data/cache/rsna" / "manifest.parquet")
        oracle_man = oracle_man[
            (oracle_man.condition == "spinal_canal_stenosis")
            & (oracle_man.split == "train")
            & (oracle_man.severity_index.isin([0, 1, 2]))
        ].reset_index(drop=True)
        if args.smoke:
            oracle_man = oracle_man.head(400)
        d_or = RsnaCropDataset(
            oracle_man, ROOT / "data/cache/rsna", crop_size=crop_size, use_25d=True, guided=False
        )
        d_au = RsnaCropDataset(
            auto_train_man, AUTO_TRAIN, crop_size=crop_size, use_25d=True, guided=False
        )
        return ConcatDataset([d_or, d_au]), _targets(oracle_man) + _targets(auto_train_man)

    def ds_consistency():
        j = JitterSampler(
            CropJitterConfig(mode="level_aware", tail_prob=0.15, tail_sigma=20.0), prof_for_jit
        )
        return RobustCanalCropDataset(
            canal_train_nodes, SLICE_CACHE, crop_size=crop_size, jitter=j, two_views=True
        ), _targets(canal_train_nodes)

    VARIANTS = {
        "r_oracle_ctrl": (ds_oracle, 0.0, 1.0, 0),
        "r_jitter_level": (ds_jitter_level, 0.0, 1.0, 0),
        "r_jitter_empirical": (ds_jitter_emp, 0.0, 1.0, 0),
        "r_auto_train": (ds_auto, 0.0, 1.0, 4),
        "r_mixed": (ds_mixed, 0.0, 1.0, 4),
        "r_consistency": (ds_consistency, 0.10, 2.0, 0),
    }
    chosen = args.variants or list(VARIANTS)

    auto_val_loader = _auto_val_loader(crop_size, batch_size)
    c1_man = GAP / "c1_gtxy_gtslice" / "manifest.parquet"
    c4_man = GAP / "c4_autoxy_midslice" / "manifest.parquet"

    results: dict = {
        "baseline_oracle_C1": ORACLE_BASELINE,
        "n_boot": args.n_boot,
        "epochs": epochs,
        "variants": {},
    }
    for name in chosen:
        factory, cw, scm, nw = VARIANTS[name]
        print(f"\n========== {name} (consistency={cw}) ==========")
        train_ds, train_tgts = factory()
        run_dir = RUNS / name
        train_robust_variant(
            variant=name,
            train_dataset=train_ds,
            train_targets=train_tgts,
            auto_val_loader=auto_val_loader,
            e0_config_path=E0_CFG,
            run_dir=run_dir,
            epochs=epochs,
            batch_size=batch_size,
            consistency_weight=cw,
            severe_consistency_mult=scm,
            num_workers=nw,
            device="auto",
        )
        oracle_eval = _eval_on(run_dir, c1_man, ROOT / "data/cache/rsna", device, args.n_boot)
        auto_eval = _eval_on(run_dir, c4_man, GAP / "c4_autoxy_midslice", device, args.n_boot)
        base = ORACLE_BASELINE["severe_recall"]
        denom = 0.828 - 0.644
        recovered = (auto_eval["severe_recall"] - 0.644) / denom if denom else 0.0
        results["variants"][name] = {
            "oracle_C1_severe_recall": oracle_eval["ci"]["severe_recall"],
            "oracle_C1_weighted_logloss": oracle_eval["ci"]["weighted_logloss"],
            "auto_C4_severe_recall": auto_eval["ci"]["severe_recall"],
            "auto_C4_weighted_logloss": auto_eval["ci"]["weighted_logloss"],
            "auto_C4_recall_at_far10": auto_eval["ci"]["recall_at_far10"],
            "auto_C4_severe_auroc": auto_eval["ci"]["severe_auroc"],
            "auto_C4_ece": auto_eval["ci"]["ece"],
            "auto_severe_recall_point": float(auto_eval["severe_recall"]),
            "oracle_severe_recall_point": float(oracle_eval["severe_recall"]),
            "gap_recovered_frac_vs_established": float(recovered),
        }
        print(
            f"[{name}] auto severe recall={auto_eval['severe_recall']:.3f} "
            f"(baseline 0.644, oracle 0.828) gap_recovered={recovered:+.2%} | "
            f"oracle severe recall={oracle_eval['severe_recall']:.3f}"
        )
        OUT_JSON.write_text(json.dumps(results, indent=2, default=float))

    # gap recovery vs the canal-only control (cleanest causal measure of jitter)
    if "r_oracle_ctrl" in results["variants"]:
        ctrl = results["variants"]["r_oracle_ctrl"]
        base = ctrl["auto_severe_recall_point"]
        ceil = ctrl["oracle_severe_recall_point"]
        denom = ceil - base
        for v in results["variants"].values():
            v["gap_recovered_frac_vs_ctrl"] = (
                float((v["auto_severe_recall_point"] - base) / denom) if denom else None
            )
        results["control_baseline_auto_severe_recall"] = base
        results["control_oracle_ceiling_severe_recall"] = ceil
        OUT_JSON.write_text(json.dumps(results, indent=2, default=float))

    print(f"\n[robust] wrote {OUT_JSON}")
    _print_summary(results)
    return 0


def _print_summary(results: dict) -> None:
    print("\n=== ROBUST AUTO-CROP TRAINING (canal, eval on AUTO C4) ===")
    base = results.get("control_baseline_auto_severe_recall")
    ceil = results.get("control_oracle_ceiling_severe_recall")
    if base is not None:
        print(f"canal-only control: auto severe recall={base:.3f}, oracle ceiling={ceil:.3f}")
    print("established all-cond E0 canal: auto 0.644 -> oracle 0.828\n")
    print(
        f"{'variant':<20}{'auto sevR [95% CI]':<26}{'recov(ctrl)':<12}"
        f"{'auto wll':<10}{'oracle sevR':<12}"
    )
    for name, v in results["variants"].items():
        sr = v["auto_C4_severe_recall"]
        wl = v["auto_C4_weighted_logloss"]
        orc = v["oracle_C1_severe_recall"]
        rc = v.get("gap_recovered_frac_vs_ctrl")
        rc_s = f"{rc:+.0%}" if rc is not None else "n/a"
        print(
            f"{name:<20}{sr['point']:.3f} [{sr['ci_lo']:.3f},{sr['ci_hi']:.3f}]      "
            f"{rc_s:<13}{wl['point']:.3f}      {orc['point']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
