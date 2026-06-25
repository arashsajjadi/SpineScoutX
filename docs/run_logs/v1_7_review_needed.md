# v1.7 — expert review needed (human-review handoff)

> Research-only · not diagnostic. No reviewed label file was present in this run, so the
> label-repair path proceeds with **provisional algorithmic cleaning** (clearly marked
> provisional, not ground truth). Human review remains the main path.

## Exact handoff

1. Open `review_packs/v1_7_hard_cases/index.html` (local-only, 704 cases).
2. For each case set the true severity (or `ambiguous_moderate_severe` /
   `insufficient_evidence` / `exclude_from_training`).
3. Save as `review_packs/v1_7_hard_cases/review_sheet_reviewed.csv` (keep all original
   columns; add a `review_label` column).
4. Re-run `python scripts/ingest_review_labels_v1_7.py` then Phase 6 retraining
   (`train_noise_aware_foraminal_v1_7.py --mode reviewed`).

Priority: right-foraminal severe FN (87 in pack), then L4-L5/L5-S1, then left-foraminal.
Reviewed labels are versioned separately (raw RSNA labels untouched); locked-test cases
are never included.
