#!/usr/bin/env python3
"""Safety Mode v3 — severe-first dashboard across all available AUTO conditions.

Extends the canal Safety Mode to every condition that has a real auto route (canal +
left/right foraminal). For each, on the locked-test auto distribution: operating points,
severe-recall frontier (recall@FAR), abstention/review policy with reasons (low
confidence + model disagreement), and cluster-bootstrap CIs. Subarticular is reported as
oracle-only/blocked (no auto route yet). Research-only. Not diagnostic; a review flag is a
research signal, not triage advice.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation import safety_mode as sm
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/safety_mode_v3.json"
DOC = ROOT / "docs/run_logs/safety_mode_v3.md"
FIG = ROOT / "outputs/real/figures/safety_mode_v3_dashboard.png"

# condition -> (deployable grader run, comparison grader run, auto cache). The deployable
# grader is the one that wins locked-test auto per condition (Phase D model selection):
# canal -> auto-trained robust; foraminal -> oracle-trained (its localizer is clean enough
# that robust auto-training does not help). The comparison run drives the disagreement
# review signal.
ROUTES = {
    "spinal_canal_stenosis": (
        "runs/v1_canal_auto_robust",
        "runs/v1_canal_oracle_ctrl",
        "data/cache/rsna_auto_canal_all",
    ),
    "left_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "runs/v1_foraminal_auto_robust",
        "data/cache/rsna_auto_foraminal",
    ),
    "right_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "runs/v1_foraminal_auto_robust",
        "data/cache/rsna_auto_foraminal",
    ),
}
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_sv3.parquet"
)


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
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)

    out: dict = {
        "protocol": "splits_v1 locked-test",
        "distribution": "auto",
        "n_boot": args.n_boot,
        "conditions": {},
    }
    for cond, (rob_run, ctrl_run, cache) in ROUTES.items():
        rob, ctrl, cache_p = ROOT / rob_run, ROOT / ctrl_run, ROOT / cache
        if not (rob / "best.pt").exists() or not (cache_p / "manifest.parquet").exists():
            out["conditions"][cond] = {"status": "unavailable (route not built)"}
            continue
        man = read_manifest(cache_p / "manifest.parquet")
        man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
        man["study_id"] = man.study_id.astype(str)
        te = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
        if te.empty:
            out["conditions"][cond] = {"status": "no locked-test auto crops"}
            continue
        y, p, st = _probs(rob, te, cache_p, device)
        rep = sm.safety_report(y, p, target_recalls=(0.90, 0.95))
        rep["recall_at_far10_ci"] = bs.bootstrap_ci(
            y, p, st, bs.make_recall_at_far(0.10), n_boot=args.n_boot
        )
        rep["severe_recall_ci"] = bs.bootstrap_ci(y, p, st, bs.m_severe_recall, n_boot=args.n_boot)
        rep["status"] = "auto"
        rep["n"], rep["n_severe"] = int(len(y)), int((y == 2).sum())
        # review reason: disagreement vs the oracle-trained control (if available)
        if (ctrl / "best.pt").exists():
            yc, pc, sc = _probs(ctrl, te, cache_p, device)
            if np.array_equal(y, yc):
                pr, prc = np.argmax(p, 1), np.argmax(pc, 1)
                disagree = pr != prc
                fn = (y == 2) & (pr != 2)
                rep["disagreement_review"] = {
                    "review_rate": float(disagree.mean()),
                    "severe_fn_captured": float((fn & disagree).sum() / max(int(fn.sum()), 1)),
                }
        out["conditions"][cond] = rep

    out["conditions"]["left_subarticular_stenosis"] = {
        "status": "oracle-only (axial route not built; see subarticular_auto_results.md)"
    }
    out["conditions"]["right_subarticular_stenosis"] = {
        "status": "oracle-only (axial route not built; see subarticular_auto_results.md)"
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _doc(out)
    _fig(out)
    _print(out)
    return 0


def _print(out):
    print("\n=== Safety Mode v3 (locked-test auto) ===")
    for cond, r in out["conditions"].items():
        if r.get("status") != "auto":
            print(f"  {cond:32s} {r.get('status')}")
            continue
        op = r["operating_points"]["balanced_argmax"]
        f10 = r["recall_at_far10_ci"]
        s90 = r["operating_points"]["safety"].get("recall>=0.9", {})
        far90 = f"{s90['false_alarm_rate']:.3f}" if s90.get("reached") else "n/a"
        print(
            f"  {cond:32s} sevR={op['severe_recall']:.3f} | recall@FAR10={f10['point']:.3f} "
            f"[{f10['ci_lo']:.3f},{f10['ci_hi']:.3f}] | FAR@90%={far90} | n_sev={r['n_severe']}"
        )


def _doc(out):
    lines = [
        "# Safety Mode v3 — multi-condition severe-first dashboard (locked-test auto)",
        "",
        "> Research-only. Not diagnostic. Not for medical decision-making. `review_required`",
        "> is a research signal, not triage advice. Every row is the auto (real-inference)",
        "> distribution on the locked `test`, cluster-bootstrap 95% CIs.",
        "",
        "| condition | status | n / sev | argmax sevR | recall@FAR10 [CI] | FAR@90% | review→FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond, r in out["conditions"].items():
        if r.get("status") != "auto":
            lines.append(f"| {cond} | {r.get('status')} | - | - | - | - | - |")
            continue
        op = r["operating_points"]["balanced_argmax"]
        f10 = r["recall_at_far10_ci"]
        s90 = r["operating_points"]["safety"].get("recall>=0.9", {})
        far90 = f"{s90['false_alarm_rate']:.3f}" if s90.get("reached") else "n/a"
        dr = r.get("disagreement_review", {})
        rev = f"{dr.get('review_rate', 0):.1%}→{dr.get('severe_fn_captured', 0):.0%}" if dr else "-"
        lines.append(
            f"| {cond} | auto | {r['n']} / {r['n_severe']} | {op['severe_recall']:.3f} "
            f"(FAR {op['false_alarm_rate']:.3f}) | {f10['point']:.3f} "
            f"[{f10['ci_lo']:.3f}, {f10['ci_hi']:.3f}] | {far90} | {rev} |"
        )
    lines += [
        "",
        "## Review reasons (per the decision layer)",
        "low top-class confidence / high entropy (abstention curve in JSON); **model",
        "disagreement** between the auto-robust and control graders (column above: review",
        "rate → fraction of robust severe-FNs captured). Cost-sensitive *training* is NOT",
        "used (prior honest negative); severe-safety comes from robust auto-training + the",
        "threshold frontier + this review layer.",
        "",
        "Subarticular L/R are **oracle-only** (axial route not built; see",
        "`subarticular_auto_results.md`). Artifacts: `outputs/real/safety_mode_v3.json`.",
        "Reproduce: `python scripts/run_safety_mode_v3.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


def _fig(out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[sv3] figure skipped: {exc}")
        return
    autos = {c: r for c, r in out["conditions"].items() if r.get("status") == "auto"}
    if not autos:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, r in autos.items():
        c = r["abstention_curve"]
        ax.plot(
            [x["abstain_rate"] for x in c],
            [x["effective_severe_recall_with_review"] for x in c],
            marker=".",
            label=cond,
        )
    ax.set_xlabel("review / abstention rate")
    ax.set_ylabel("effective severe recall (with review)")
    ax.set_title("Safety Mode v3 — severe recall vs review burden (locked-test auto)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
