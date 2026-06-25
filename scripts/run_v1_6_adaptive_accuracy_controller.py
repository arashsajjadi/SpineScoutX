#!/usr/bin/env python3
"""v1.6 adaptive accuracy controller — record every plan's result + the A→B→C→D decision.

Reads each plan's locked-test result JSON (whatever exists), evaluates the v1.6 success criteria,
and writes an honest decision log (experiment id, train/dev/test usage, metrics, decision, next
action) — failed experiments are never hidden. Does not run training itself; it is the audit trail
the offensive is steered by. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
OUT = ROOT / "outputs/real"
DOC = ROOT / "docs/run_logs/v1_6_adaptive_controller.md"

# baseline (deployed) locked-test severe recall
BASE = {"canal": 0.830, "l_for": 0.788, "r_for": 0.660, "l_sub": 0.746, "r_sub": 0.737}
BASE_FOR_MACRO = (BASE["l_for"] + BASE["r_for"]) / 2
BASE_MACRO5 = sum(BASE.values()) / 5


def _read(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def _foraminal_arm(cmp_json, arm_key):
    """Extract (l_for, r_for, macro) severe recall for an arm from a transfer-compare JSON."""
    if not cmp_json:
        return None
    pc = cmp_json.get("per_condition", {})
    lc = pc.get("left_neural_foraminal_narrowing", {})
    rc = pc.get("right_neural_foraminal_narrowing", {})
    if not lc or not rc:
        return None
    lf = lc[arm_key]["severe_recall"]
    rf = rc[arm_key]["severe_recall"]
    return {"l_for": lf, "r_for": rf, "macro": (lf + rf) / 2}


def _criteria(r_for, for_macro):
    """v1.6 success thresholds vs the deployed baseline (severe recall)."""
    return {
        "r_for_+0.05": (r_for is not None) and (r_for - BASE["r_for"] >= 0.05),
        "for_macro_+0.03": (for_macro is not None) and (for_macro - BASE_FOR_MACRO >= 0.03),
    }


def main() -> int:
    log = {"baseline": BASE, "experiments": [], "decision_chain": []}

    # ---- Plan A: external LSS foraminal data ----
    cmp_a = _read("compare_foraminal_transfer_v1_6.json")
    base_arm = _foraminal_arm(cmp_a, "baseline")
    lss_arm = _foraminal_arm(cmp_a, "lss")
    joint = _read("foraminal_rsna_joint_v1_6.json")
    joint_macro = (joint or {}).get("test", {}).get("foraminal_macro_severe_recall")
    if cmp_a:
        log["experiments"].append(
            {
                "id": "A.baseline",
                "plan": "A",
                "data": "RSNA splits_v1 (train/dev/test)",
                "selection": "dev foraminal-macro recall@FAR10",
                "locked_test_reads": 1,
                "result": base_arm,
                "note": "ImageNet init; reproduces deployed (macro 0.724)",
            }
        )
        log["experiments"].append(
            {
                "id": "A.transfer_lss",
                "plan": "A",
                "data": "LSS pretrain -> RSNA fine-tune",
                "selection": "dev foraminal-macro recall@FAR10",
                "locked_test_reads": 1,
                "result": lss_arm,
                "criteria": _criteria(lss_arm["r_for"], lss_arm["macro"]) if lss_arm else None,
                "verdict": "NEGATIVE (severe recall decisively worse; ranking flat)",
            }
        )
        log["experiments"].append(
            {
                "id": "A.joint_lss",
                "plan": "A",
                "data": "RSNA + LSS lss_train pooled (+179 severe)",
                "selection": "dev foraminal-macro recall@FAR10",
                "locked_test_reads": 1,
                "result": {"macro": joint_macro},
                "verdict": "NEGATIVE (severe recall identical to baseline; Δ0.000)",
            }
        )
        log["decision_chain"].append(
            "A executed: external LSS pretrain (decisive loss) + joint (no change) -> "
            "external data does not raise the foraminal severe ceiling -> run Plan B."
        )

    # ---- Plan B: self-supervised pretraining ----
    cmp_b = _read("compare_foraminal_ssl_v1_6.json")
    ssl_arm = _foraminal_arm(cmp_b, "lss")  # 'lss' key reused as the experimental arm
    if cmp_b:
        log["experiments"].append(
            {
                "id": "B.ssl_finetune",
                "plan": "B",
                "data": "SimCLR on RSNA train+dev + LSS (locked-test excluded) -> RSNA fine-tune",
                "selection": "dev foraminal-macro recall@FAR10",
                "locked_test_reads": 1,
                "result": ssl_arm,
                "criteria": _criteria(ssl_arm["r_for"], ssl_arm["macro"]) if ssl_arm else None,
            }
        )
        won = ssl_arm and any(_criteria(ssl_arm["r_for"], ssl_arm["macro"]).values())
        nxt = "IMPROVED (stop)" if won else "no grading gain -> run Plan D / autopsy"
        log["decision_chain"].append(f"B executed: SSL representation -> {nxt}.")

    # ---- Plan D: stronger route-specific grader ----
    cmp_d = _read("compare_foraminal_strong_v1_6.json")
    strong_arm = _foraminal_arm(cmp_d, "lss")
    if cmp_d:
        log["experiments"].append(
            {
                "id": "D.stronger_backbone",
                "plan": "D",
                "data": "RSNA splits_v1; larger timm backbone (+ severe handling)",
                "selection": "dev foraminal-macro recall@FAR10",
                "locked_test_reads": 1,
                "result": strong_arm,
                "criteria": _criteria(strong_arm["r_for"], strong_arm["macro"])
                if strong_arm
                else None,
            }
        )

    # ---- overall verdict ----
    any_win = False
    for e in log["experiments"]:
        c = e.get("criteria")
        if c and any(c.values()):
            any_win = True
    log["overall"] = {
        "any_success_criterion_met": any_win,
        "verdict": "ACCURACY UPGRADE" if any_win else "ADAPTIVE EXECUTED NEGATIVE",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v1_6_adaptive_controller.json").write_text(json.dumps(log, indent=2, default=float))
    print(json.dumps(log, indent=2, default=float))
    print(f"\nOVERALL: {log['overall']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
