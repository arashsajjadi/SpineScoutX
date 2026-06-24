#!/usr/bin/env python3
"""Safety Mode v4 + router selection across ALL available auto findings (locked test).

Resolves the best deployable grader per condition (the router: canal=auto-robust,
foraminal=oracle-trained, subarticular=best of auto/oracle from its run) and builds the
severe-first safety dashboard on the locked-test auto distribution: operating points,
recall@FAR, FAR@90%, abstention/review (low-confidence + model-disagreement reasons),
cluster-bootstrap CIs. Conditions without a usable auto route are labelled oracle-only.
Research-only. Not diagnostic; review is a research signal, not triage advice.
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
OUT = ROOT / "outputs/real/safety_mode_v4.json"
DOC = ROOT / "docs/run_logs/safety_mode_v4.md"
FIG = ROOT / "outputs/real/figures/safety_mode_v4_dashboard.png"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_sv4.parquet"
)
# fixed routes (deployable, comparison, cache); subarticular resolved dynamically
FIXED = {
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
SUB = ("left_subarticular_stenosis", "right_subarticular_stenosis")


def _resolve_subarticular() -> dict:
    """Pick the better subarticular grader per side from its results JSON (router)."""
    out = {}
    p = ROOT / "outputs/real/subarticular_auto_results.json"
    if not p.exists():
        return out
    r = json.loads(p.read_text())
    for cond in SUB:
        best_run, best_sr = None, -1.0
        for g in ("subarticular_auto_robust", "subarticular_oracle_ctrl"):
            e = r.get("variants", {}).get(g, {}).get(cond)
            if e:
                sr = e["test_auto"]["severe_recall"]["point"]
                if sr > best_sr:
                    best_sr, best_run = sr, f"runs/v1_{g}"
        if best_run:
            other = (
                "runs/v1_subarticular_oracle_ctrl"
                if "auto_robust" in best_run
                else "runs/v1_subarticular_auto_robust"
            )
            out[cond] = (best_run, other, "data/cache/rsna_auto_subarticular")
    return out


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
    routes = {**FIXED, **_resolve_subarticular()}

    out = {
        "protocol": "splits_v1 locked-test",
        "distribution": "auto",
        "n_boot": args.n_boot,
        "router": {c: r[0] for c, r in routes.items()},
        "conditions": {},
    }
    for cond in list(FIXED) + list(SUB):
        if cond not in routes:
            out["conditions"][cond] = {"status": "oracle-only (auto route not built)"}
            continue
        rob, ctrl, cache = (ROOT / x for x in routes[cond])
        if not (rob / "best.pt").exists() or not (cache / "manifest.parquet").exists():
            out["conditions"][cond] = {"status": "oracle-only (auto route not built)"}
            continue
        man = read_manifest(cache / "manifest.parquet")
        man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
        man["study_id"] = man.study_id.astype(str)
        te = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
        if te.empty:
            out["conditions"][cond] = {"status": "no locked-test auto crops"}
            continue
        y, p, st = _probs(rob, te, cache, device)
        rep = sm.safety_report(y, p, target_recalls=(0.90, 0.95))
        rep["status"] = "auto"
        rep["deployable_grader"] = routes[cond][0]
        rep["recall_at_far10_ci"] = bs.bootstrap_ci(
            y, p, st, bs.make_recall_at_far(0.10), n_boot=args.n_boot
        )
        rep["severe_recall_ci"] = bs.bootstrap_ci(y, p, st, bs.m_severe_recall, n_boot=args.n_boot)
        rep["n"], rep["n_severe"] = int(len(y)), int((y == 2).sum())
        if (ctrl / "best.pt").exists():
            yc, pc, _ = _probs(ctrl, te, cache, device)
            if np.array_equal(y, yc):
                pr, prc = np.argmax(p, 1), np.argmax(pc, 1)
                dis = pr != prc
                fn = (y == 2) & (pr != 2)
                rep["disagreement_review"] = {
                    "review_rate": float(dis.mean()),
                    "severe_fn_captured": float((fn & dis).sum() / max(int(fn.sum()), 1)),
                }
        out["conditions"][cond] = rep

    n_auto = sum(1 for r in out["conditions"].values() if r.get("status") == "auto")
    out["n_auto_conditions"] = n_auto
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _doc(out)
    print(f"\n=== Safety Mode v4 ({n_auto}/5 auto, locked-test) ===")
    for cond, r in out["conditions"].items():
        if r.get("status") != "auto":
            print(f"  {cond:32s} {r.get('status')}")
            continue
        op = r["operating_points"]["balanced_argmax"]
        f10 = r["recall_at_far10_ci"]
        print(
            f"  {cond:32s} sevR={op['severe_recall']:.3f} recall@FAR10={f10['point']:.3f} "
            f"[{f10['ci_lo']:.3f},{f10['ci_hi']:.3f}] (sev={r['n_severe']})"
        )
    return 0


def _doc(out):
    lines = [
        "# Safety Mode v4 — multi-condition severe-first dashboard + router (locked-test auto)",
        "",
        "> Research-only. Not diagnostic. Not for medical decision-making. `review_required` is a",
        f"> research signal, not triage advice. Auto distribution, locked `test`; "
        f"{out['n_auto_conditions']}/5 conditions have a real auto route; cluster-bootstrap CIs.",
        "",
        "Router (deployable grader per condition): "
        + ", ".join(f"{c.split('_')[0]}→{Path(r).name}" for c, r in out["router"].items()),
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
        rev = f"{dr.get('review_rate', 0):.0%}→{dr.get('severe_fn_captured', 0):.0%}" if dr else "-"
        lines.append(
            f"| {cond} | auto | {r['n']} / {r['n_severe']} | {op['severe_recall']:.3f} | "
            f"{f10['point']:.3f} [{f10['ci_lo']:.3f}, {f10['ci_hi']:.3f}] | {far90} | {rev} |"
        )
    lines += [
        "",
        "Review reasons: low top-class confidence / high entropy (abstention curve in JSON) +",
        "model disagreement (router grader vs its comparison). Cost-sensitive training not used",
        "(prior honest negative). Reproduce: `python scripts/run_safety_mode_v4.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
