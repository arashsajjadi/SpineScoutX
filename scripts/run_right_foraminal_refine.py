#!/usr/bin/env python3
"""Right-foraminal refinement: right-specialist grader vs the combined side-aware grader.

Right-foraminal auto severe recall (0.660) trails left (0.788) on the locked test, but the
CIs overlap (n_severe ~53). This trains a RIGHT-ONLY oracle-trained specialist and tests,
paired on the same locked-test right-foraminal nodes, whether it decisively beats the
existing combined grader (runs/v1_foraminal_oracle_ctrl). Honest outcome either way.
Research-only. Not diagnostic.
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
AUTO = ROOT / "data/cache/rsna_auto_foraminal"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
E0_CFG = ROOT / "runs/e0_baseline_real/config.json"
COMBINED = ROOT / "runs/v1_foraminal_oracle_ctrl"
SPEC = ROOT / "runs/v1_right_foraminal_specialist"
COND = "right_neural_foraminal_narrowing"
OUT = ROOT / "outputs/real/right_foraminal_refinement.json"
DOC = ROOT / "docs/run_logs/right_foraminal_refinement.md"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_rf.parquet"
)


def _sel(man, split_map, split, cond=COND):
    m = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
    m["study_id"] = m["study_id"].astype(str)
    return m[m["study_id"].map(split_map) == split].reset_index(drop=True)


def _probs(run_dir, man, cache, device):
    man.to_parquet(TMP)
    preds = collect_probs(run_dir, TMP, cache, device)
    keys = sorted(preds)
    y = np.array([preds[k][0] for k in keys])
    p = np.stack([preds[k][1] for k in keys])
    st = np.array([k.split("|")[0] for k in keys])
    return y, p, st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)
    oracle_man = read_manifest(ORACLE / "manifest.parquet")
    auto_man = read_manifest(AUTO / "manifest.parquet")

    # right-only oracle-trained specialist, selected on right-foraminal dev auto
    if not (SPEC / "best.pt").exists():
        dev_loader = DataLoader(
            RsnaCropDataset(
                _sel(auto_man, split_map, "dev"), AUTO, crop_size=224, use_25d=True, guided=False
            ),
            batch_size=64,
            shuffle=False,
            num_workers=4,
        )
        tr = _sel(oracle_man, split_map, "train")
        print(f"[rf] training right-foraminal specialist on {len(tr)} oracle crops...")
        train_robust_variant(
            variant="v1_right_foraminal_specialist",
            train_dataset=RsnaCropDataset(tr, ORACLE, crop_size=224, use_25d=True, guided=False),
            train_targets=[int(t) for t in tr.severity_index.tolist()],
            auto_val_loader=dev_loader,
            e0_config_path=E0_CFG,
            run_dir=SPEC,
            epochs=args.epochs,
            batch_size=32,
            num_workers=4,
            device="auto",
        )

    te_auto = _sel(auto_man, split_map, "test")
    yc, pc, sc = _probs(COMBINED, te_auto, AUTO, device)  # combined grader
    ys, ps, ss = _probs(SPEC, te_auto, AUTO, device)  # specialist
    assert np.array_equal(yc, ys) and np.array_equal(sc, ss)
    d = bs.paired_bootstrap_delta(ys, ps, pc, ss, bs.m_severe_recall, n_boot=args.n_boot)
    out = {
        "condition": COND,
        "n": int(len(yc)),
        "n_severe": int((yc == 2).sum()),
        "combined_auto_severe_recall": bs.bootstrap_ci(
            yc, pc, sc, bs.m_severe_recall, n_boot=args.n_boot
        ),
        "specialist_auto_severe_recall": bs.bootstrap_ci(
            ys, ps, ss, bs.m_severe_recall, n_boot=args.n_boot
        ),
        "paired_specialist_minus_combined": d,
        "mcnemar": bs.mcnemar_severe(yc, np.argmax(ps, 1), np.argmax(pc, 1)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _doc(out)
    cmb, spc = out["combined_auto_severe_recall"], out["specialist_auto_severe_recall"]
    print(
        f"[rf] combined {cmb['point']:.3f} [{cmb['ci_lo']:.3f},{cmb['ci_hi']:.3f}] | "
        f"specialist {spc['point']:.3f} [{spc['ci_lo']:.3f},{spc['ci_hi']:.3f}] | "
        f"paired Δ {d['delta']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] decisive={d['decisive']}"
    )
    return 0


def _doc(o):
    cmb, spc, d = (
        o["combined_auto_severe_recall"],
        o["specialist_auto_severe_recall"],
        o["paired_specialist_minus_combined"],
    )
    verdict = (
        "The right-specialist DECISIVELY improves right-foraminal."
        if d["decisive"] and d["delta"] > 0
        else "No decisive improvement: the L/R asymmetry is within sampling noise at this "
        "severe count; the limit is sample size, not the grader."
    )
    DOC.write_text(
        "# Right-foraminal refinement — specialist vs combined (locked test)\n\n"
        "> Research-only. Not diagnostic. Right-only oracle-trained specialist vs the combined\n"
        f"> side-aware grader, paired on the same locked-test right-foraminal nodes "
        f"(n={o['n']}, severe={o['n_severe']}); cluster-bootstrap CIs.\n\n"
        "| grader | right-foraminal auto severe recall [95% CI] |\n|---|---|\n"
        f"| combined side-aware | {cmb['point']:.3f} [{cmb['ci_lo']:.3f}, {cmb['ci_hi']:.3f}] |\n"
        f"| right specialist | {spc['point']:.3f} [{spc['ci_lo']:.3f}, {spc['ci_hi']:.3f}] |\n\n"
        f"Paired specialist − combined: **{d['delta']:+.3f}** "
        f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] (decisive={d['decisive']}); "
        f"McNemar {o['mcnemar']['b_a_catches_b_misses']}/{o['mcnemar']['c_a_misses_b_catches']} "
        f"p={o['mcnemar']['p_value']:.3g}.\n\n## Verdict\n{verdict}\n\n"
        "Reproduce: `python scripts/run_right_foraminal_refine.py`.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
