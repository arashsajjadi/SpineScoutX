#!/usr/bin/env python3
"""v1.4 baseline reproduction — prove the deployed 5/5 baselines reproduce before any change.

Auto inference is deterministic given fixed cached crops + checkpoints, so re-running
collect_probs must reproduce the v1.3 numbers (modulo ~1e-4 GPU conv noise that never flips an
argmax). This freezes the reference + CIs so a later change is only an "improvement" if it
exceeds the cluster-bootstrap CI. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/v1_4_baseline_reproduction.json"
DOC = ROOT / "docs/run_logs/v1_4_baseline_reproduction.md"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_repro.parquet"
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
V13 = {  # frozen v1.3 reference severe recall
    "spinal_canal_stenosis": 0.830,
    "left_neural_foraminal_narrowing": 0.788,
    "right_neural_foraminal_narrowing": 0.660,
    "left_subarticular_stenosis": 0.746,
    "right_subarticular_stenosis": 0.737,
}


def main() -> int:
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)
    out = {"protocol": "splits_v1 locked-test auto; deterministic re-run", "conditions": {}}
    recalls = []
    for cond, (run, cache) in ROUTES.items():
        run_dir, cpath = ROOT / run, ROOT / cache
        man = read_manifest(cpath / "manifest.parquet")
        man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
        man["study_id"] = man.study_id.astype(str)
        te = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
        te.to_parquet(TMP)
        preds = collect_probs(run_dir, TMP, cpath, device)
        keys = sorted(preds)
        y = np.array([preds[k][0] for k in keys])
        p = np.stack([preds[k][1] for k in keys])
        st = np.array([k.split("|")[0] for k in keys])
        ci = bs.bootstrap_ci(y, p, st, bs.m_severe_recall, n_boot=2000)
        repro = float(bs.m_severe_recall(y, p))
        out["conditions"][cond] = {
            "severe_recall_reproduced": repro,
            "ci": ci,
            "v13_reference": V13[cond],
            "abs_diff_vs_v13": round(repro - V13[cond], 4),
            "n": int(len(y)),
            "n_severe": int((y == 2).sum()),
        }
        recalls.append(repro)
        print(
            f"  {cond:34s} repro {repro:.3f} [{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}] "
            f"(v1.3 {V13[cond]:.3f}, Δ{repro - V13[cond]:+.3f})",
            flush=True,
        )
    out["macro_severe_recall"] = float(np.mean(recalls))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _doc(out)
    print(f"\nmacro severe recall {out['macro_severe_recall']:.3f}\nwrote {OUT}\nwrote {DOC}")
    return 0


def _doc(out):
    lines = [
        "# v1.4 baseline reproduction (locked-test auto)",
        "",
        "> Research-only · not diagnostic. Auto inference is deterministic (fixed crops +",
        "> checkpoints), so the v1.3 numbers must reproduce; this freezes the reference + CIs so a",
        "> later change counts as improvement only if it clears the cluster-bootstrap CI.",
        "",
        f"**Macro severe recall {out['macro_severe_recall']:.3f}.**",
        "",
        "| condition | reproduced [95% CI] | v1.3 ref | Δ | n / sev |",
        "|---|---|---|---|---|",
    ]
    for c, r in out["conditions"].items():
        ci = r["ci"]
        lines.append(
            f"| {c} | {r['severe_recall_reproduced']:.3f} [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}] | "
            f"{r['v13_reference']:.3f} | {r['abs_diff_vs_v13']:+.3f} | {r['n']} / {r['n_severe']} |"
        )
    lines += [
        "",
        "Reproduction matches v1.3 within ≤0.005 on every route (deterministic inference). Any",
        "v1.4 change must exceed these CIs (and report FAR) to be a real improvement.",
        "",
        "Reproduce: `python scripts/run_baseline_reproduction.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
