# Kaggle Late Submission Plan — SpineScoutX v1.9

> Research-only · non-commercial · not diagnostic · not clinically validated.

## Purpose

Generate and submit a late Kaggle submission for the
**RSNA 2024 Lumbar Spine Degenerative Classification** competition using the
released SpineScoutX v1.9 best raw model package, then compare the resulting
score against the visible leaderboard.

## What this sprint is NOT

- **Not a training sprint.** No model is retrained.
- **Not an accuracy-improvement sprint.** Internal metrics stay unchanged.
- **Not a competition entry.** This is a post-deadline late submission for
  honest external benchmarking only.

## Kaggle metric vs. our internal metric

Our internal evaluation uses **5-route macro severe recall** (argmax).
The Kaggle competition metric is **weighted log loss** across three probability
classes (`normal_mild`, `moderate`, `severe`), with **higher weight on severe
labels**. These are different:

- Our argmax recall says nothing about probability calibration.
- Kaggle score rewards well-calibrated soft probabilities.
- A model with good recall can still have a bad log-loss if its probabilities
  are poorly calibrated.

We do **not** compare internal recall directly to Kaggle log-loss.

## Leaderboard context

This is a late submission. We are not official competition participants on the
private leaderboard. The score we receive is informational only — it shows how
the SpineScoutX research pipeline performs on the Kaggle metric under the same
conditions as competition entries.

## Safety constraints

- No Kaggle or HuggingFace token is printed, echoed, committed, or exposed.
- No competition DICOM/image data is committed to git.
- No test labels are used for tuning.
- No model weights are committed to git.
- No large files (> 50 MiB) are committed to git.
- No DICOMs/NIfTI are committed.
- No PHI is exposed.
- No clinical claims.

## Expected pipeline phases

| Phase | Description |
|---|---|
| 0 | Branch, plan doc (this file) |
| 1 | Kaggle auth |
| 2 | Download/verify competition test data |
| 3 | Locate/download v1.9 model package, verify checksums |
| 4 | Generate submission.csv |
| 5 | Dry-run local validation |
| 6 | Submit to Kaggle |
| 7 | Compare against leaderboard |
| 8 | README note (if score is meaningful) |
| 9 | Gates, push, PR, merge |

## Output files

| File | Location | Committed |
|---|---|---|
| submission.csv | `submissions/` | Only if small/safe |
| validation JSON | `submissions/` | Yes |
| Result docs | `docs/run_logs/kaggle_*.md` | Yes |
| Competition data | `data/kaggle/` (gitignored) | No |
| Model weights | `artifacts/v1_9_release/` (gitignored) | No |
