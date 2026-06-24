#!/usr/bin/env python3
"""Multi-condition LOCKED-TEST oracle baselines + view-routing feasibility taxonomy.

Retrains an all-conditions E0 grader on splits_v1 `train`, selects on `dev`, and
evaluates ONCE on the locked `test` per condition (oracle crops) with cluster-bootstrap
CIs. Then assembles the honest view-routing taxonomy: which findings the current
sagittal-T2 auto-localizer can serve (canal) vs which need new view-specific
localization (foraminal = sagittal-T1 parasagittal; subarticular = axial-T2).

Why oracle for non-canal: the auto-localized distribution cannot be generated for
foraminal/subarticular without view-specific localizers (documented as the next
frontier), so this establishes the upper-bound multi-condition baseline on a clean
locked test, with the auto gap quantified for canal (see canal_locked_test.md).

Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from spinescoutx.constants import CONDITIONS
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
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
E0_CFG = ROOT / "runs/e0_baseline_real/config.json"
OUT = ROOT / "outputs/real/multicondition_robust_results.json"
DOC = ROOT / "docs/run_logs/multicondition_robust_results.md"
RUN = ROOT / "runs/v1_allcond_e0"

# GT-coordinate view per condition (measured: 100% each) -> auto-localization status.
VIEW_ROUTING = {
    "spinal_canal_stenosis": (
        "sagittal_t2",
        "available (sagittal-T2 disc localizer; v0.9 auto recipe)",
    ),
    "left_neural_foraminal_narrowing": (
        "sagittal_t1",
        "BLOCKER: needs parasagittal-T1 side-aware localizer",
    ),
    "right_neural_foraminal_narrowing": (
        "sagittal_t1",
        "BLOCKER: needs parasagittal-T1 side-aware localizer",
    ),
    "left_subarticular_stenosis": (
        "axial_t2",
        "BLOCKER: needs axial-T2 localizer + level matching",
    ),
    "right_subarticular_stenosis": (
        "axial_t2",
        "BLOCKER: needs axial-T2 localizer + level matching",
    ),
}


def _man(man, split_map, split, conditions=None):
    m = man[man.severity_index.isin([0, 1, 2])].copy()
    m["study_id"] = m["study_id"].astype(str)
    m = m[m["study_id"].map(split_map) == split]
    if conditions is not None:
        m = m[m.condition.isin(conditions)]
    return m.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    device = select_device("auto")
    tmp = Path(
        "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
        "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_mc.parquet"
    )

    split_map = load_splits_v1(SPLITS)
    man = read_manifest(ORACLE / "manifest.parquet")
    tr = _man(man, split_map, "train")
    dev = _man(man, split_map, "dev")
    print(f"[mc] all-cond train {len(tr)} / dev {len(dev)}")

    dev_loader = DataLoader(
        RsnaCropDataset(dev, ORACLE, crop_size=224, use_25d=True, guided=False),
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )
    print("[mc] retraining all-condition E0 on splits_v1 train (oracle)...")
    train_robust_variant(
        variant="v1_allcond_e0",
        train_dataset=RsnaCropDataset(tr, ORACLE, crop_size=224, use_25d=True, guided=False),
        train_targets=[int(t) for t in tr.severity_index.tolist()],
        auto_val_loader=dev_loader,
        e0_config_path=E0_CFG,
        run_dir=RUN,
        epochs=args.epochs,
        batch_size=32,
        num_workers=4,
        device="auto",
    )

    # locked-test per-condition (oracle) with CIs
    results = {
        "protocol": "splits_v1 locked-test",
        "provenance": "oracle (upper bound)",
        "model": "all-condition E0 retrained on splits_v1 train",
        "n_boot": args.n_boot,
        "per_condition": {},
        "view_routing": {},
    }
    for cond in CONDITIONS:
        mc = _man(man, split_map, "test", conditions=[cond])
        mc.to_parquet(tmp)
        preds = collect_probs(RUN, tmp, ORACLE, device)
        keys = sorted(preds)
        y = np.array([preds[k][0] for k in keys])
        p = np.stack([preds[k][1] for k in keys])
        st = np.array([k.split("|")[0] for k in keys])
        ci = bs.ci_table(y, p, st, n_boot=args.n_boot)
        results["per_condition"][cond] = {
            "n": int(len(y)),
            "n_severe": int((y == 2).sum()),
            "severe_recall": ci["severe_recall"],
            "weighted_logloss": ci["weighted_logloss"],
            "severe_auroc": ci["severe_auroc"],
            "macro_f1_argmax": float(bs.balanced_accuracy(y, np.argmax(p, 1))),
            "recall_at_far10": ci["recall_at_far10"],
            "ece": ci["ece"],
        }
        view, status = VIEW_ROUTING[cond]
        results["view_routing"][cond] = {"gt_view": view, "auto_localization": status}
        sr = ci["severe_recall"]
        print(
            f"[mc] {cond:32s} test-oracle severe recall {sr['point']:.3f} "
            f"[{sr['ci_lo']:.3f},{sr['ci_hi']:.3f}] n_sev={int((y == 2).sum())} | {view}"
        )

    # attach canal auto result if available
    canal_lt = ROOT / "outputs/real/canal_locked_test.json"
    if canal_lt.exists():
        cj = json.loads(canal_lt.read_text())
        results["canal_auto_locked_test"] = cj["variants"]["canal_auto_robust"]["test_auto"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    _doc(results)
    print(f"[mc] wrote {OUT}")
    return 0


def _doc(r):
    lines = [
        "# Multi-condition locked-test baselines + view-routing taxonomy",
        "",
        "> Research-only. Not diagnostic. All-condition E0 retrained on splits_v1 `train`,",
        "> selected on `dev`, evaluated ONCE on the locked `test`. Cluster-bootstrap 95% CIs.",
        "> **Non-canal numbers are oracle (GT-coordinate) UPPER BOUNDS** — the auto-localized",
        "> distribution cannot yet be generated for foraminal/subarticular (see taxonomy).",
        "",
        "## Locked-test per-condition (oracle crops)",
        "",
        "| condition | n / severe | severe recall [95% CI] | wll | severe AUROC | GT view |",
        "|---|---|---|---|---|---|",
    ]
    for cond, c in r["per_condition"].items():
        sr, wl, au = c["severe_recall"], c["weighted_logloss"], c["severe_auroc"]
        lines.append(
            f"| {cond} | {c['n']} / {c['n_severe']} | "
            f"{sr['point']:.3f} [{sr['ci_lo']:.3f}, {sr['ci_hi']:.3f}] | {wl['point']:.3f} | "
            f"{au['point']:.3f} | {r['view_routing'][cond]['gt_view']} |"
        )
    if "canal_auto_locked_test" in r:
        a = r["canal_auto_locked_test"]["severe_recall"]
        lines += [
            "",
            f"**Canal AUTO (real inference) locked-test severe recall: {a['point']:.3f} "
            f"[{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]** (the only condition with a working "
            "auto-localizer; see `canal_locked_test.md`).",
        ]
    lines += [
        "",
        "## View-routing feasibility taxonomy (the answer to 'does v0.9 generalize to all 5?')",
        "",
        "| condition | GT view | auto-localization status |",
        "|---|---|---|",
    ]
    for cond, vr in r["view_routing"].items():
        lines.append(f"| {cond} | {vr['gt_view']} | {vr['auto_localization']} |")
    lines += [
        "",
        "## Honest conclusion",
        "- **1/5 conditions (canal)** has a working auto-localizer; the v0.9 robust recipe applies",
        "  and is confirmed on the locked test (`canal_locked_test.md`).",
        "- **2/5 (foraminal L/R)** are graded on **sagittal-T1** parasagittal side-specific",
        "  slices. v0.9's 'slice doesn't matter' finding is canal-specific; foraminal needs",
        "  correct parasagittal-T1 slice selection — a side-aware T1 localizer is the next step.",
        "- **2/5 (subarticular L/R)** are graded on **axial-T2**. SPIDER has no axial anatomy and",
        "  no axial localizer exists; an axial localizer + level matching is needed.",
        "- So 'generalize v0.9 to all five' is **gated by view-specific localization**, not by the",
        "  grading recipe. This is a routing/localization frontier, documented (not faked).",
        "  The oracle baselines above bound what each condition could reach once auto-",
        "  localization for its view exists.",
        "",
        "Artifacts: `outputs/real/multicondition_robust_results.json`. Reproduce:",
        "`python scripts/run_multicondition_v1.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
