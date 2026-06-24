#!/usr/bin/env python3
"""Foraminal (sagittal-T1 side-aware) auto route, locked-test (splits_v1).

Pipeline (resume-safe): foraminal localizer data prep -> train localizer (on splits_v1
train, select on dev) -> localizer QC -> auto foraminal crops (no GT) -> robust graders
(oracle-trained control vs auto-trained) selected on dev auto -> locked-test eval per
side (oracle + auto) with cluster-bootstrap CIs + paired robust-vs-control deltas.

Research-only. Not diagnostic. Auto inference reads NO GT coordinates.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from spinescoutx.config import config_from_dict
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.datasets import RsnaCropDataset
from spinescoutx.data.foraminal_localize import (
    FORAMINAL,
    _load_foraminal_localizer,
    _localize_slice,
    prepare_foraminal_localizer_data,
    prepare_rsna_foraminal_auto_crops,
)
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.data.rsna_index import RsnaPaths
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device
from spinescoutx.training.train_localizer import train_localizer
from spinescoutx.training.train_robust import train_robust_variant

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
ORACLE = ROOT / "data/cache/rsna"
LOC_CACHE = ROOT / "data/cache/foraminal_localizer"
AUTO = ROOT / "data/cache/rsna_auto_foraminal"
LOC_RUN = ROOT / "runs/lf_foraminal_localizer"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
E0_CFG = ROOT / "runs/e0_baseline_real/config.json"
LOC_CFG = ROOT / "runs/l0_disc_localizer_real/config.json"
OUT = ROOT / "outputs/real/foraminal_auto_results.json"
DOC = ROOT / "docs/run_logs/foraminal_auto_results.md"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_for.parquet"
)


def _sel(man, split_map, split, cond=None):
    m = man[man.severity_index.isin([0, 1, 2])].copy()
    m["study_id"] = m["study_id"].astype(str)
    m = m[m["study_id"].map(split_map) == split]
    if cond is not None:
        m = m[m.condition == cond]
    elif "condition" in m.columns:
        m = m[m.condition.isin(FORAMINAL)]
    return m.reset_index(drop=True)


def _localizer_qc(device, split_map) -> dict:
    """Per-side localizer error on dev+test (median/mean px, PCK, crop-hit)."""
    man = pd.read_parquet(LOC_CACHE / "localizer_manifest.parquet")
    man["study_id"] = man.study_id.astype(str)
    model, slice_size = _load_foraminal_localizer(LOC_RUN, device)
    images_dir = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    qc: dict = {}
    for split in ("dev", "test"):
        rows = man[man.study_id.map(split_map) == split]
        for side in ("left", "right"):
            d = []
            for r in rows[rows.side == side].itertuples():
                pts, _ = _localize_slice(
                    images_dir,
                    r.study_id,
                    r.series_id,
                    int(r.instance_number),
                    model,
                    slice_size,
                    device,
                )
                if pts is None:
                    continue
                for li, lv in enumerate(["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]):
                    gx, gy = getattr(r, f"kx_{lv}"), getattr(r, f"ky_{lv}")
                    if np.isfinite(gx) and np.isfinite(gy):
                        # GT kpts are in slice_size space; scale pred back to slice space
                        px = pts[li, 0] * slice_size / r.orig_w
                        py = pts[li, 1] * slice_size / r.orig_h
                        d.append(float(np.hypot(px - gx, py - gy)))
            d = np.array(d)
            qc[f"{split}_{side}"] = {
                "n": int(d.size),
                "median_px": float(np.median(d)) if d.size else None,
                "mean_px": float(np.mean(d)) if d.size else None,
                "pck@10": float((d <= 10).mean()) if d.size else None,
                "crop_hit@224": float((d <= 112).mean()) if d.size else None,
            }
    return qc


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
    ap.add_argument("--loc-epochs", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--limit-studies", type=int, default=None)
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)

    # 1. localizer data
    if not (LOC_CACHE / "localizer_manifest.parquet").exists():
        print("[for] preparing foraminal localizer data...")
        s = prepare_foraminal_localizer_data(
            ROOT / "data/raw/rsna",
            LOC_CACHE,
            split_map,
            slice_size=256,
            limit_studies=args.limit_studies,
        )
        print(f"[for] localizer rows={s['n_rows']} split={s['split']}")

    # 2. train localizer (clone canal localizer cfg; point at foraminal cache)
    if not (LOC_RUN / "best.pt").exists():
        print("[for] training foraminal localizer...")
        d = json.loads(LOC_CFG.read_text())
        d["name"] = "lf_foraminal_localizer"
        d["data"]["rsna_cache"] = str(LOC_CACHE)
        d["train"] = {**d["train"], "epochs": int(args.loc_epochs)}
        train_localizer(config_from_dict(d), LOC_RUN)
    loc_metrics = json.loads((LOC_RUN / "metrics.json").read_text())["best"]
    print(
        f"[for] localizer best dev mean_px={loc_metrics.get('mean_px_dist'):.2f} "
        f"crop_hit@224={loc_metrics.get('crop_hit@224'):.3f}"
    )
    qc = _localizer_qc(device, split_map)

    # 3. auto crops (no GT)
    if not (AUTO / "manifest.parquet").exists():
        print("[for] generating auto foraminal crops...")
        studies = None
        if args.limit_studies:
            lm = read_manifest(ORACLE / "manifest.parquet")
            studies = sorted(lm[lm.condition.isin(FORAMINAL)].study_id.astype(str).unique())[
                : args.limit_studies
            ]
        rep = prepare_rsna_foraminal_auto_crops(
            ROOT / "data/raw/rsna", LOC_RUN, AUTO, studies=studies, crop_size=224
        )
        print(f"[for] auto crops={rep['n_auto_crops']} skipped={rep['skipped']}")
    oracle_man = read_manifest(ORACLE / "manifest.parquet")
    auto_man = read_manifest(AUTO / "manifest.parquet")

    # 4. robust graders (combined side-aware): oracle-trained control vs auto-trained
    dev_loader = DataLoader(
        RsnaCropDataset(
            _sel(auto_man, split_map, "dev"), AUTO, crop_size=224, use_25d=True, guided=False
        ),
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )
    variants = {
        "foraminal_oracle_ctrl": (ORACLE, oracle_man),
        "foraminal_auto_robust": (AUTO, auto_man),
    }
    results = {
        "protocol": "splits_v1 locked-test",
        "conditions": list(FORAMINAL),
        "n_boot": args.n_boot,
        "localizer_qc": qc,
        "localizer_dev": loc_metrics,
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
        for cond in FORAMINAL:
            y, p, st, ci = _eval(
                run_dir, _sel(auto_man, split_map, "test", cond), AUTO, device, args.n_boot
            )
            yo, po, sto, cio = _eval(
                run_dir, _sel(oracle_man, split_map, "test", cond), ORACLE, device, args.n_boot
            )
            ev[cond] = {
                "test_auto": {
                    "severe_recall": ci["severe_recall"],
                    "weighted_logloss": ci["weighted_logloss"],
                    "recall_at_far10": ci["recall_at_far10"],
                    "severe_auroc": ci["severe_auroc"],
                    "ece": ci["ece"],
                    "n": int(len(y)),
                    "n_severe": int((y == 2).sum()),
                },
                "test_oracle": {
                    "severe_recall": cio["severe_recall"],
                    "weighted_logloss": cio["weighted_logloss"],
                },
            }
            if name == "foraminal_auto_robust":
                preds[("robust", cond)] = (y, p, st)
            if name == "foraminal_oracle_ctrl":
                preds[("ctrl", cond)] = (y, p, st)
        results["variants"][name] = ev

    # paired robust vs ctrl on locked test auto, per side
    results["paired_robust_vs_ctrl_auto"] = {}
    for cond in FORAMINAL:
        yr, pr, sr = preds[("robust", cond)]
        yc, pc, sc = preds[("ctrl", cond)]
        if np.array_equal(yr, yc) and np.array_equal(sr, sc):
            results["paired_robust_vs_ctrl_auto"][cond] = {
                "severe_recall": bs.paired_bootstrap_delta(
                    yr, pr, pc, sr, bs.m_severe_recall, n_boot=args.n_boot
                ),
                "mcnemar": bs.mcnemar_severe(yr, np.argmax(pr, 1), np.argmax(pc, 1)),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    _doc(results)
    _print(results)
    return 0


def _print(r):
    print("\n=== FORAMINAL locked-test auto (severe recall [95% CI]) ===")
    for cond in FORAMINAL:
        rob = r["variants"]["foraminal_auto_robust"][cond]["test_auto"]
        ctl = r["variants"]["foraminal_oracle_ctrl"][cond]["test_auto"]
        orc = r["variants"]["foraminal_auto_robust"][cond]["test_oracle"]
        d = r["paired_robust_vs_ctrl_auto"].get(cond, {}).get("severe_recall", {})
        print(
            f"  {cond:32s} robust {rob['severe_recall']['point']:.3f} "
            f"[{rob['severe_recall']['ci_lo']:.3f},{rob['severe_recall']['ci_hi']:.3f}] "
            f"| ctrl {ctl['severe_recall']['point']:.3f} | orc {orc['severe_recall']['point']:.3f} "
            f"| Δ {d.get('delta', float('nan')):+.3f} decisive={d.get('decisive')}"
        )


def _doc(r):
    lines = [
        "# Foraminal (sagittal-T1) auto route — locked test (splits_v1)",
        "",
        "> Research-only. Not diagnostic. Auto inference reads NO GT coordinates: T1 series +",
        "> parasagittal slice chosen from DICOM `ImagePositionPatient` (laterality) + the",
        "> localizer's own confidence (best-slice). Graders retrained on splits_v1 `train`,",
        "> selected on `dev` auto, evaluated ONCE on locked `test`. Cluster-bootstrap 95% CIs.",
        "",
        "## Foraminal localizer QC (per side)",
        "| split/side | n | median px | mean px | PCK@10 | crop-hit@224 |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in r["localizer_qc"].items():
        lines.append(
            f"| {k} | {v['n']} | {v['median_px']:.2f} | {v['mean_px']:.2f} | "
            f"{v['pck@10']:.3f} | {v['crop_hit@224']:.3f} |"
            if v["median_px"] is not None
            else f"| {k} | 0 | - | - | - | - |"
        )
    lines += [
        "",
        "## Grader severe recall [95% CI] on locked test (n / severe per side)",
        "| condition | control (auto) | **auto-robust (auto)** | oracle ceiling | paired Δ |",
        "|---|---|---|---|---|",
    ]
    for cond in FORAMINAL:
        rob = r["variants"]["foraminal_auto_robust"][cond]["test_auto"]
        ctl = r["variants"]["foraminal_oracle_ctrl"][cond]["test_auto"]
        orc = r["variants"]["foraminal_auto_robust"][cond]["test_oracle"]["severe_recall"]
        d = r["paired_robust_vs_ctrl_auto"].get(cond, {}).get("severe_recall", {})
        rs, cs = rob["severe_recall"], ctl["severe_recall"]
        lines.append(
            f"| {cond} (n={rob['n']}, sev={rob['n_severe']}) | "
            f"{cs['point']:.3f} [{cs['ci_lo']:.3f}, {cs['ci_hi']:.3f}] | "
            f"**{rs['point']:.3f} [{rs['ci_lo']:.3f}, {rs['ci_hi']:.3f}]** | "
            f"{orc['point']:.3f} | {d.get('delta', float('nan')):+.3f} "
            f"(decisive={d.get('decisive')}) |"
        )
    lines += [
        "",
        "Provenance: oracle = GT-coordinate crop (upper bound); auto = localizer-predicted",
        "parasagittal-T1 crop (real inference). Artifacts: `foraminal_auto_results.json`.",
        "Reproduce: `python scripts/run_foraminal_locked_test.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
