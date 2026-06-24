#!/usr/bin/env python3
"""Build the locked patient-level train/dev/test protocol (splits_v1).

Writes gitignored split metadata + a protocol doc with per-condition severe counts.
The locked test is for FINAL evaluation only. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import (
    assert_disjoint,
    build_splits_v1,
    save_splits_v1,
    stratified_counts,
)

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
ORACLE = ROOT / "data/cache/rsna/manifest.parquet"
OUT = ROOT / "data/cache/splits_v1/splits.json"
DOC = ROOT / "docs/run_logs/locked_test_protocol.md"
SEED = 20260623


def main() -> int:
    man = read_manifest(ORACLE)
    man["study_id"] = man["study_id"].astype(str)
    studies = sorted(man["study_id"].unique())
    split_map = build_splits_v1(studies, seed=SEED, dev_frac=0.15, test_frac=0.15)
    assert_disjoint(split_map)
    save_splits_v1(split_map, OUT, seed=SEED, timestamp="2026-06-23")
    counts = stratified_counts(man, split_map)
    (OUT.parent / "stratified_counts.json").write_text(json.dumps(counts, indent=2))

    n_by = {s: counts[s]["n_studies"] for s in ("train", "dev", "test")}
    print(f"[splits_v1] studies: {n_by} (total {sum(n_by.values())})")
    for s in ("train", "dev", "test"):
        print(f"  {s}: {counts[s]['n_crops']} crops, {counts[s]['n_severe']} severe")

    _write_doc(counts, n_by)
    print(f"[splits_v1] wrote {OUT} and {DOC}")
    return 0


def _write_doc(counts: dict, n_by: dict) -> None:
    conds = sorted(counts["train"]["by_condition"])
    lines = [
        "# Locked patient-level test protocol (splits_v1)",
        "",
        "> Research-only. Not diagnostic. The **locked test** is used for FINAL evaluation",
        "> only — never for model selection or tuning. `dev` is for selection/tuning.",
        "> Historical (seed-1337 val) results are preserved separately and are NOT v1 claims.",
        "",
        f"Patient-level (study_id) three-way split, seed {SEED}, dev=15% / test=15%.",
        f"Studies: train {n_by['train']} / dev {n_by['dev']} / test {n_by['test']}"
        f" (total {sum(n_by.values())}). Disjoint by construction (one study → one split;"
        " `assert_disjoint` + `check_no_leakage`).",
        "",
        "## Severe counts per split per condition (oracle crops)",
        "",
        "| condition | train n / severe | dev n / severe | **test n / severe** |",
        "|---|---|---|---|",
    ]
    for c in conds:
        tr = counts["train"]["by_condition"].get(c, {"n": 0, "n_severe": 0})
        dv = counts["dev"]["by_condition"].get(c, {"n": 0, "n_severe": 0})
        te = counts["test"]["by_condition"].get(c, {"n": 0, "n_severe": 0})
        lines.append(
            f"| {c} | {tr['n']} / {tr['n_severe']} | {dv['n']} / {dv['n_severe']} | "
            f"**{te['n']} / {te['n_severe']}** |"
        )
    lines += [
        "",
        "## Protocol rules",
        "- Train on `train`; select hyperparameters / checkpoints on `dev`; report final",
        "  numbers on `test` (locked) exactly once per model.",
        "- Every v1 headline table states split (`dev`/`test`) and provenance (oracle/auto),",
        "  n, n_severe, and a bootstrap CI.",
        "- Models claimed on the locked test are **retrained on `train`** (the historical",
        "  E0/E2/E3/r_* checkpoints trained on the seed-1337 split, which overlaps this test",
        "  set, so they are NOT eligible for locked-test claims).",
        "",
        "Artifacts: `data/cache/splits_v1/splits.json`, `stratified_counts.json` (gitignored).",
        "Reproduce: `python scripts/build_splits_v1.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
