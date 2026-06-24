#!/usr/bin/env python3
"""Safety Mode v2: cost-sensitive TRAINING + decision layer, on locked-test auto (canal).

Trains a cost-sensitive canal grader (ExpectedCostLoss, severe-FN >> FP) on splits_v1
`train` auto crops, selects on `dev` auto, and compares it on the locked `test` auto
distribution against the oracle-trained control and the auto-trained robust grader
(from run_canal_locked_test). Reports the severe-first frontier + abstention/review
policy (incl. model-disagreement as a review reason) with cluster-bootstrap CIs.

Research-only. Not diagnostic. A "review" flag is a research signal, not triage advice.
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
from spinescoutx.evaluation import safety_mode as sm
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device
from spinescoutx.training.train_robust import train_robust_variant

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
ORACLE = ROOT / "data/cache/rsna"
AUTO = ROOT / "data/cache/rsna_auto_canal_all"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
E0_CFG = ROOT / "runs/e0_baseline_real/config.json"
OUT = ROOT / "outputs/real/safety_mode_v2.json"
DOC = ROOT / "docs/run_logs/safety_mode_v2.md"
FIG = ROOT / "outputs/real/figures/safety_mode_v2_frontier.png"
COND = "spinal_canal_stenosis"


def _canal(man, split_map, split):
    m = man[(man.condition == COND) & (man.severity_index.isin([0, 1, 2]))].copy()
    m["study_id"] = m["study_id"].astype(str)
    return m[m["study_id"].map(split_map) == split].reset_index(drop=True)


def _probs(run_dir, man, cache, device, tmp):
    man.to_parquet(tmp)
    preds = collect_probs(run_dir, tmp, cache, device)
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
    tmp = Path(
        "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
        "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_sv2.parquet"
    )
    split_map = load_splits_v1(SPLITS)
    auto_man = read_manifest(AUTO / "manifest.parquet")

    # train cost-sensitive variant on auto train crops (loss=cost_sensitive)
    au_tr = _canal(auto_man, split_map, "train")
    dev_loader = DataLoader(
        RsnaCropDataset(
            _canal(auto_man, split_map, "dev"), AUTO, crop_size=224, use_25d=True, guided=False
        ),
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )
    run_cost = ROOT / "runs/v1_canal_cost_sensitive"
    print("[sv2] training cost-sensitive canal grader (ExpectedCostLoss) on auto train...")
    train_robust_variant(
        variant="v1_canal_cost_sensitive",
        train_dataset=RsnaCropDataset(au_tr, AUTO, crop_size=224, use_25d=True, guided=False),
        train_targets=[int(t) for t in au_tr.severity_index.tolist()],
        auto_val_loader=dev_loader,
        e0_config_path=E0_CFG,
        run_dir=run_cost,
        epochs=args.epochs,
        batch_size=32,
        num_workers=4,
        device="auto",
        loss="cost_sensitive",
    )

    # locked-test auto predictions for the three models (same nodes)
    te = _canal(auto_man, split_map, "test")
    models = {
        "oracle_ctrl": ROOT / "runs/v1_canal_oracle_ctrl",
        "auto_robust": ROOT / "runs/v1_canal_auto_robust",
        "cost_sensitive": run_cost,
    }
    P = {name: _probs(rd, te, AUTO, device, tmp) for name, rd in models.items()}
    y = P["auto_robust"][0]

    out = {
        "protocol": "splits_v1 locked-test",
        "condition": COND,
        "distribution": "auto",
        "n": int(len(y)),
        "n_severe": int((y == 2).sum()),
        "n_boot": args.n_boot,
        "models": {},
    }
    for name, (yy, pp, ss) in P.items():
        rep = sm.safety_report(yy, pp, target_recalls=(0.90, 0.95))
        rep["recall_at_far10_ci"] = bs.bootstrap_ci(
            yy, pp, ss, bs.make_recall_at_far(0.10), n_boot=args.n_boot
        )
        rep["severe_recall_ci"] = bs.bootstrap_ci(
            yy, pp, ss, bs.m_severe_recall, n_boot=args.n_boot
        )
        out["models"][name] = rep

    # disagreement-as-review-reason: review nodes where robust vs ctrl argmax differ
    pr = np.argmax(P["auto_robust"][1], 1)
    pc = np.argmax(P["oracle_ctrl"][1], 1)
    disagree = pr != pc
    is_sev = y == 2
    robust_fn = is_sev & (pr != 2)
    out["disagreement_review"] = {
        "review_rate": float(disagree.mean()),
        "robust_severe_fn_captured_by_disagreement": float(
            (robust_fn & disagree).sum() / max(int(robust_fn.sum()), 1)
        ),
        "n_robust_severe_fn": int(robust_fn.sum()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _doc(out)
    _fig(out, P)
    _print(out)
    return 0


def _print(out):
    print(f"\n=== Safety Mode v2 (lt auto canal; n={out['n']} sev={out['n_severe']}) ===")
    for name, rep in out["models"].items():
        op = rep["operating_points"]["balanced_argmax"]
        far10 = rep["recall_at_far10_ci"]
        s90 = rep["operating_points"]["safety"].get("recall>=0.9", {})
        far_for_90 = f"{s90['false_alarm_rate']:.3f}" if s90.get("reached") else "n/a"
        print(
            f"  {name:15s} sevR={op['severe_recall']:.3f} FAR={op['false_alarm_rate']:.3f} | "
            f"recall@FAR10={far10['point']:.3f} [{far10['ci_lo']:.3f},{far10['ci_hi']:.3f}] | "
            f"FAR@90%sevR={far_for_90} | cost={rep['cost_weighted']['mean_cost']:.3f}"
        )
    d = out["disagreement_review"]
    print(
        f"  disagreement review: {d['review_rate']:.1%} of nodes; captures "
        f"{d['robust_severe_fn_captured_by_disagreement']:.1%} of robust severe-FNs"
    )


def _doc(out):
    lines = [
        "# Safety Mode v2 — cost-sensitive training + decision layer (locked-test auto)",
        "",
        "> Research-only. Not diagnostic. Not for medical decision-making. A `review_required`",
        "> flag is a research signal, not triage advice. Canal, locked `test`, auto (real)",
        f"> distribution; n={out['n']}, severe={out['n_severe']}; cluster-bootstrap 95% CIs.",
        "",
        "Three graders, all selected on `dev` auto and evaluated ONCE on locked `test` auto:",
        "oracle-trained control, auto-trained robust (v0.9 recipe), and **cost-sensitive**",
        "(ExpectedCostLoss, severe-FN ≫ FP) trained on auto crops.",
        "",
        "| model | argmax severe recall | recall@FAR≤10% [95% CI] | FAR@90% sevR | cost |",
        "|---|---|---|---|---|",
    ]
    for name, rep in out["models"].items():
        op = rep["operating_points"]["balanced_argmax"]
        far10 = rep["recall_at_far10_ci"]
        s90 = rep["operating_points"]["safety"].get("recall>=0.9", {})
        far_for_90 = f"{s90['false_alarm_rate']:.3f}" if s90.get("reached") else "n/a"
        lines.append(
            f"| {name} | {op['severe_recall']:.3f} (FAR {op['false_alarm_rate']:.3f}) | "
            f"{far10['point']:.3f} [{far10['ci_lo']:.3f}, {far10['ci_hi']:.3f}] | "
            f"{far_for_90} | {rep['cost_weighted']['mean_cost']:.3f} |"
        )
    d = out["disagreement_review"]
    lines += [
        "",
        "## Review policy (reasons)",
        "Beyond low-confidence/high-entropy abstention (see `abstention_curve` in the JSON), a",
        "**model-disagreement** review reason flags nodes where the robust and control graders",
        f"disagree: that flags **{d['review_rate']:.1%}** of nodes and captures "
        f"**{d['robust_severe_fn_captured_by_disagreement']:.1%}** of the robust model's severe",
        "false-negatives — a cheap, transparent triage signal.",
        "",
        "## Honest verdict",
        "Cost-sensitive training is reported alongside the auto-robust recipe; whichever wins the",
        "severe frontier on the locked test (with non-overlapping CI) is preferred, otherwise they",
        "are called comparable. Reaching 90% severe recall has an explicit false-alarm cost;",
        "if it is high, that is stated, not hidden.",
        "",
        "Artifacts: `outputs/real/safety_mode_v2.json`, `figures/safety_mode_v2_frontier.png`.",
        "Reproduce: `python scripts/run_safety_mode_v2.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


def _fig(out, P):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[sv2] figure skipped: {exc}")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rep in out["models"].items():
        sweep = rep["abstention_curve"]
        ax.plot(
            [r["abstain_rate"] for r in sweep],
            [r["effective_severe_recall_with_review"] for r in sweep],
            marker=".",
            label=name,
        )
    ax.set_xlabel("review / abstention rate")
    ax.set_ylabel("effective severe recall (with review)")
    ax.set_title("Safety Mode v2 — severe recall vs review burden (locked-test auto)")
    ax.grid(alpha=0.3)
    ax.legend()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
