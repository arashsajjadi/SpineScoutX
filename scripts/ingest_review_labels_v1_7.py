#!/usr/bin/env python3
"""Ingest + validate expert-reviewed hard-case labels (v1.7).

Reads a completed ``review_packs/v1_7_hard_cases/review_sheet_reviewed.csv`` (a ``review_label``
column added by the reviewer), validates it hard, and writes a **separately-versioned** corrected
label set (`data/labels/v1_7_reviewed_labels.parquet`, gitignored) — it **never** overwrites the raw
RSNA labels and **never** ingests locked-test cases. If no reviewed file exists, it writes the exact
human-review handoff and exits 0 so Phase 4 (provisional cleaning) can proceed. Research-only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
PACK = ROOT / "review_packs/v1_7_hard_cases"
REVIEWED = PACK / "review_sheet_reviewed.csv"
SHEET = PACK / "review_sheet.csv"
OUT = ROOT / "data/labels/v1_7_reviewed_labels.parquet"
DOC = ROOT / "docs/run_logs/v1_7_reviewed_labels_summary.md"
NEEDED = ROOT / "docs/run_logs/v1_7_review_needed.md"
VALID_LABELS = {
    "normal_mild", "moderate", "severe",
    "ambiguous_moderate_severe", "insufficient_evidence", "exclude_from_training",
}  # fmt: skip
SEV_INDEX = {"normal_mild": 0, "moderate": 1, "severe": 2}


def _write_review_needed(n_cases: int):
    NEEDED.write_text(
        "# v1.7 — expert review needed (human-review handoff)\n\n"
        "> Research-only · not diagnostic. No reviewed label file was present in this run, so the\n"
        "> label-repair path proceeds with **provisional algorithmic cleaning** (clearly marked\n"
        "> provisional, not ground truth). Human review remains the main path.\n\n"
        "## Exact handoff\n\n"
        f"1. Open `review_packs/v1_7_hard_cases/index.html` (local-only, {n_cases} cases).\n"
        "2. For each case set the true severity (or `ambiguous_moderate_severe` /\n"
        "   `insufficient_evidence` / `exclude_from_training`).\n"
        "3. Save as `review_packs/v1_7_hard_cases/review_sheet_reviewed.csv` (keep all original\n"
        "   columns; add a `review_label` column).\n"
        "4. Re-run `python scripts/ingest_review_labels_v1_7.py` then Phase 6 retraining\n"
        "   (`train_noise_aware_foraminal_v1_7.py --mode reviewed`).\n\n"
        "Priority: right-foraminal severe FN (87 in pack), then L4-L5/L5-S1, then left-foraminal.\n"
        "Reviewed labels are versioned separately (raw RSNA labels untouched); locked-test cases\n"
        "are never included.\n"
    )


def main() -> int:
    n_pack = len(pd.read_csv(SHEET)) if SHEET.exists() else 0
    if not REVIEWED.exists():
        _write_review_needed(n_pack)
        print(f"[ingest] no reviewed CSV; wrote handoff {NEEDED.name} -> provisional cleaning next")
        return 0

    sheet = pd.read_csv(SHEET)
    rev = pd.read_csv(REVIEWED)
    errors = []
    if "review_label" not in rev.columns:
        raise SystemExit("review_sheet_reviewed.csv missing 'review_label' column")
    valid_ids = set(sheet.case_id)
    unknown = set(rev.case_id) - valid_ids
    if unknown:
        errors.append(f"{len(unknown)} unknown case_id(s)")
    if rev.case_id.duplicated().any():
        errors.append("duplicate case_id(s)")
    bad = set(rev.review_label.dropna()) - VALID_LABELS
    if bad:
        errors.append(f"invalid label(s): {sorted(bad)}")
    if rev.review_label.isna().any() or (rev.review_label.astype(str).str.strip() == "").any():
        errors.append("empty review_label(s)")
    # never ingest locked-test: every reviewed case must be train/dev in the pack
    splits = sheet.set_index("case_id")["split"].to_dict()
    test_ids = [c for c in rev.case_id if splits.get(c) == "test"]
    if test_ids:
        errors.append(f"{len(test_ids)} locked-test case(s) present — refusing (no test labels)")
    if errors:
        raise SystemExit("[ingest] VALIDATION FAILED: " + "; ".join(errors))

    merged = rev.merge(sheet[["case_id", "key", "split", "current_rsna_label"]], on="case_id")
    merged["reviewed_label"] = merged["review_label"]
    merged["reviewed_severity_index"] = merged["review_label"].map(SEV_INDEX)  # NaN for flags
    merged["changed_vs_rsna"] = merged["reviewed_label"] != merged["current_rsna_label"]
    keep = [
        "case_id", "key", "split", "current_rsna_label", "reviewed_label",
        "reviewed_severity_index", "changed_vs_rsna",
    ]  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged[keep].to_parquet(OUT, index=False)  # versioned separately; raw labels untouched
    n = len(merged)
    changed = int(merged.changed_vs_rsna.sum())
    DOC.write_text(
        "# v1.7 reviewed-label ingestion summary\n\n"
        "> Research-only · not diagnostic. Corrected labels are versioned separately "
        "(`data/labels/v1_7_reviewed_labels.parquet`, gitignored); raw RSNA labels are never "
        "overwritten; locked-test cases are never ingested.\n\n"
        f"- reviewed cases: **{n}** (train/dev only)\n"
        f"- changed vs RSNA: **{changed}** ({100 * changed / max(n, 1):.1f}%)\n"
        f"- label distribution: {dict(merged.reviewed_label.value_counts())}\n\n"
        "Next: `train_noise_aware_foraminal_v1_7.py --mode reviewed` (Phase 6).\n"
    )
    print(f"[ingest] {n} reviewed cases ({changed} changed) -> {OUT}; wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
