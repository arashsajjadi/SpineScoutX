#!/usr/bin/env python3
"""Provisional algorithmic label-cleaning (v1.7, fallback when no human labels).

Turns the train+dev signal table into **provisional soft labels + ambiguity flags + sample
weights** — explicitly NOT ground truth, a fallback for human review. Rules (never flip a label
blindly; train/dev only; locked-test never touched):
  * orig severe + every strong model confidently non-severe + low disagreement -> AMBIGUOUS
    (soften toward, but keep, severe; downweight) — not an auto-flip.
  * orig moderate + ensemble strongly severe -> soft probability mass toward severe (flag).
  * orig severe + ensemble agrees severe -> sharpen severe + upweight.
  * orig severe + poor/uncertain evidence -> ambiguous (downweight).
  * else -> keep original (mild smoothing), weight 1.
Original RSNA labels are never overwritten. Output is gitignored. Research-only. Not diagnostic.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SIG = ROOT / "outputs/real/v1_7_signal_table.parquet"
OUT = ROOT / "data/labels/v1_7_provisional_soft_labels.parquet"
DOC = ROOT / "docs/run_logs/v1_7_provisional_label_cleaning.md"


def _clean_row(r):
    y = int(r["severity_index"])
    ens, dis, ent = float(r["ens_p_severe"]), float(r["disagreement"]), float(r["dep_entropy"])
    soft = np.full(3, 0.05)
    soft[y] = 0.90  # default: original with mild smoothing
    w, amb, rule = 1.0, False, "keep_original"
    if y == 2 and ens < 0.15 and dis < 0.10:  # severe but all models confidently non-severe
        soft, w, amb, rule = (
            np.array([0.10, 0.30, 0.60]),
            0.5,
            True,
            "severe_ambiguous_models_disagree",
        )
    elif y == 1 and ens > 0.50 and r["n_models_call_severe"] >= 2:  # moderate but strong severe
        soft, w, amb, rule = (
            np.array([0.05, 0.45, 0.50]),
            1.0,
            True,
            "moderate_with_severe_evidence",
        )
    elif y == 2 and ens > 0.50:  # severe and models agree severe -> sharpen + upweight
        soft, w, rule = np.array([0.00, 0.10, 0.90]), 1.5, "severe_confirmed_upweight"
    elif y == 2 and ent > 0.80:  # severe but poor/uncertain evidence
        soft, w, amb, rule = np.array([0.05, 0.35, 0.60]), 0.7, True, "severe_poor_evidence"
    elif y == 2:  # any other true-severe: upweight modestly (severe-FN focus)
        w, rule = 1.2, "severe_baseline_upweight"
    return pd.Series({
        "soft0": soft[0], "soft1": soft[1], "soft2": soft[2],
        "sample_weight": w, "ambiguity_flag": amb, "rule": rule,
    })  # fmt: skip


def main() -> int:
    if not SIG.exists():
        raise SystemExit(f"missing {SIG}; run mine_hard_cases_v1_7.py first")
    df = pd.read_parquet(SIG)
    df = df[df.split.isin(["train", "dev"])].copy()  # locked-test never touched
    cleaned = df.join(df.apply(_clean_row, axis=1))
    keep = [
        "key", "study_id", "level", "condition", "side", "split", "severity_index",
        "soft0", "soft1", "soft2", "sample_weight", "ambiguity_flag", "rule",
    ]  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned[keep].to_parquet(OUT, index=False)

    sev = cleaned[cleaned.severity_index == 2]
    rfor = cleaned[cleaned.condition == "right_neural_foraminal_narrowing"]
    summary = {
        "n_train_dev": int(len(cleaned)),
        "rules_applied": {k: int(v) for k, v in Counter(cleaned.rule).items()},
        "n_ambiguity_flagged": int(cleaned.ambiguity_flag.sum()),
        "severe_findings": int(len(sev)),
        "severe_ambiguous_flagged": int(sev.ambiguity_flag.sum()),
        "severe_upweighted": int((sev.sample_weight > 1.0).sum()),
        "right_for_severe_ambiguous": int((rfor[rfor.severity_index == 2].ambiguity_flag).sum()),
        "mean_weight_by_class": {
            int(c): round(float(cleaned[cleaned.severity_index == c].sample_weight.mean()), 3)
            for c in (0, 1, 2)
        },
    }
    DOC.write_text(
        "# v1.7 provisional label cleaning (PROVISIONAL — not ground truth)\n\n"
        "> Research-only · not diagnostic. **Provisional soft labels + ambiguity flags + sample "
        "weights** for train+dev only; raw RSNA labels are never overwritten; locked-test is never "
        "touched. This is a fallback for human review, NOT a substitute. Reproduce: "
        "`scripts/provisional_label_cleaning_v1_7.py`.\n\n"
        f"- train+dev findings: **{summary['n_train_dev']}**\n"
        f"- ambiguity-flagged: **{summary['n_ambiguity_flagged']}** "
        f"(severe: {summary['severe_ambiguous_flagged']} of {summary['severe_findings']}; "
        f"right-foraminal severe: {summary['right_for_severe_ambiguous']})\n"
        f"- severe upweighted: **{summary['severe_upweighted']}**\n"
        f"- rules applied: {summary['rules_applied']}\n"
        f"- mean sample weight by class (0/1/2): {summary['mean_weight_by_class']}\n\n"
        "Soft-label rules never flip a label; they soften ambiguous severe cases, sharpen "
        "model-confirmed severe cases, and move probability mass for moderate-with-severe-evidence "
        "cases. Used by `train_noise_aware_foraminal_v1_7.py --mode provisional` (Phase 5).\n"
    )
    print(f"[clean] {summary['n_train_dev']} train+dev; {summary['n_ambiguity_flagged']} flagged; "
          f"rules {summary['rules_applied']}")  # fmt: skip
    print(f"wrote {OUT}; {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
