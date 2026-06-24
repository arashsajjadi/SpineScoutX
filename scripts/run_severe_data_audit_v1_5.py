#!/usr/bin/env python3
"""v1.5 severe-class data audit — how much severe signal exists to train on?

Counts severe samples by condition/side/level/split (from RSNA labels + splits_v1). This sets
the honest ceiling: a specialist can only learn from the severe examples that exist. No imaging
decoded; labels only. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.data.rsna_labels import load_labels

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
OUT = ROOT / "outputs/real/v1_5_severe_data_audit.json"
DOC = ROOT / "docs/run_logs/v1_5_severe_data_audit.md"
LEVELS = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")


def main() -> int:
    labels = load_labels(ROOT / "data/raw/rsna")
    labels["study_id"] = labels.study_id.astype(str)
    sm = load_splits_v1(ROOT / "data/cache/splits_v1/splits.json")
    labels["split"] = labels.study_id.map(sm)
    out = {"by_condition": {}, "severe_by_level": {}}
    conds = sorted(labels.condition.unique())
    for cond in conds:
        c = labels[labels.condition == cond]
        out["by_condition"][cond] = {}
        for split in ("train", "dev", "test"):
            cs = c[c.split == split]
            out["by_condition"][cond][split] = {
                "n": int(len(cs)),
                "n_severe": int((cs.severity_index == 2).sum()),
            }
        # per-level severe counts (train) — where is the signal?
        ctr = c[c.split == "train"]
        out["severe_by_level"][cond] = {
            lv: int(((ctr.level == lv) & (ctr.severity_index == 2)).sum()) for lv in LEVELS
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    _doc(out)
    print("=== severe counts (train n_severe / dev / test) ===")
    for cond, d in out["by_condition"].items():
        print(
            f"  {cond:34s} train_sev={d['train']['n_severe']:4d} "
            f"dev_sev={d['dev']['n_severe']:3d} test_sev={d['test']['n_severe']:3d} "
            f"(train n={d['train']['n']})"
        )
    print(f"wrote {OUT}\nwrote {DOC}")
    return 0


def _doc(out):
    lines = [
        "# v1.5 severe-class data audit (labels only)",
        "",
        "> Research-only · not diagnostic. The honest training ceiling: a specialist learns",
        "> from severe examples that exist. RSNA labels + splits_v1; no imaging decoded.",
        "",
        "| condition | train n / severe | dev severe | test severe |",
        "|---|---|---|---|",
    ]
    for cond, d in out["by_condition"].items():
        lines.append(
            f"| {cond} | {d['train']['n']} / **{d['train']['n_severe']}** | "
            f"{d['dev']['n_severe']} | {d['test']['n_severe']} |"
        )
    lines += [
        "",
        "## Train severe count by level (where the signal is)",
        "| condition | " + " | ".join(LEVELS) + " |",
        "|---|" + "---|" * len(LEVELS),
    ]
    for cond, lv in out["severe_by_level"].items():
        lines.append(f"| {cond} | " + " | ".join(str(lv[k]) for k in LEVELS) + " |")
    lines += [
        "",
        "## Interpretation",
        "- The **train severe count** per route bounds how much a specialist/MIL can learn. Routes",
        "  with few train-severe examples (esp. right-foraminal) are data-limited; MIL/aug",
        "  can help robustness but cannot create absent severe signal.",
        "- Severe examples concentrate at L4/L5 and L5/S1 (lower levels), so level-aware",
        "  sampling targets those.",
        "",
        "Reproduce: `python scripts/run_severe_data_audit_v1_5.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
