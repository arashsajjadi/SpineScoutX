# SpineScoutX label-repair pipeline (v1.7)

> Research-only · not diagnostic · not clinically validated. v1.7 treats the weak-route severe
> ceiling as a **label-quality** problem (v1.4–v1.6 ruled out model / external-data / representation
> / capacity levers). This documents the active label-cleaning + re-annotation pipeline.

## Pipeline

1. **Hard-case mining** (`mine_hard_cases_v1_7.py`, `data/hard_case_mining.py`) — over `splits_v1`
   **train+dev** foraminal (locked-test never used for cleaning), with the deployed grader + v1.6
   models' p_severe. Groups: severe FN, confidently-normal severe miss, moderate/severe borderline,
   model-disagreement, high-uncertainty, + controls. Priority-ranked review set.
2. **Local-only review pack** (`build_relabel_review_pack_v1_7.py`) — HTML + CSV + JSONL + per-case
   second-read panels. **`review_packs/` is gitignored** (imaging pixels); only counts/schema are
   committed.
3. **Human-review ingestion** (`ingest_review_labels_v1_7.py`) — validates a reviewed CSV; versions
   corrected labels **separately** (`data/labels/v1_7_reviewed_labels.parquet`, gitignored); never
   overwrites raw RSNA labels; never ingests locked-test. No reviewed file → `v1_7_review_needed.md`.
4. **Provisional cleaning (fallback)** (`provisional_label_cleaning_v1_7.py`) — soft labels +
   ambiguity flags + sample weights from the model signals; **never flips a label**; train+dev only;
   PROVISIONAL, not ground truth.
5. **Noise-aware retraining** (`train_noise_aware_foraminal_v1_7.py`) — modes A/C/D/E/F/G
   (original / soft / weights / ambiguity-downweight / severe-upweight / hybrid); dev-selects on
   right-foraminal recall@FAR≤10; locked-test once.
6. **Fallbacks** — teacher distillation (`build_kaggle_teacher_distillation_v1_7.py`, ensemble of
   own models) + severe-FN triage (`train_severe_fn_triage_v1_7.py`).

## Safety invariants

- Raw RSNA labels are never modified on disk; corrected/soft labels are versioned separately and
  gitignored. Locked-test labels are never used for cleaning, training, or selection.
- The review pack contains imaging pixels and is **local-only / gitignored**; committed artifacts
  are code, docs, schema, counts, and pixel-free synthetic examples only.
- A model is deployed only if **raw locked-test severe recall** improves; a triage gain is a
  *safety* upgrade, not an accuracy upgrade.

## Human-review handoff

When a radiologist completes `review_packs/v1_7_hard_cases/review_sheet_reviewed.csv`, re-run
ingestion then `train_noise_aware_foraminal_v1_7.py --modes ...` with the reviewed labels (Phase 6).
Priority: right-foraminal severe FN, then L4-L5/L5-S1, then left-foraminal.
