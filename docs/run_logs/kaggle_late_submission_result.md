# Kaggle Late Submission Result — SpineScoutX v1.9

> Research-only · non-commercial · not diagnostic.

## Submission status: REJECTED — competition closed

The RSNA 2024 Lumbar Spine Degenerative Classification competition
closed on **2024-10-08**. As of June 2026, the competition is not
accepting new submissions (HTTP 400 Bad Request from Kaggle API).

This is expected: RSNA 2024 was a **code competition** (Kaggle notebook),
not a file-upload competition. After the private leaderboard was revealed,
new submissions were closed.

## Submission attempt

```bash
kaggle competitions submit \
  -c rsna-2024-lumbar-spine-degenerative-classification \
  -f submissions/spinescoutx_v1_9_late_submission.csv \
  -m "SpineScoutX v1.9 best raw model late submission — research-only, no leaderboard tuning"
```

Result:
```
400 Client Error: Bad Request for url: https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateSubmission
```

## What we generated

- **Submission CSV**: `submissions/spinescoutx_v1_9_late_submission.csv`
- **25 rows** (1 study × 5 conditions × 5 levels)
- **Validation**: PASSED (all 25 rows, no NaN, prob sums = 1.0, real model probs)
- **Model**: v1.9 best raw graders (canal/foraminal/subarticular)

The generated CSV is valid and would have been a legitimate submission if
the competition were still open.

## Prior submissions

No prior Kaggle submissions for this competition from user `arash`.

## Model output summary

The generated predictions (for context only — not scored):

| Condition | Level | normal_mild | moderate | severe |
|---|---|---|---|---|
| spinal_canal_stenosis | l4_l5 | 0.015 | 0.178 | 0.808 |
| left_neural_foraminal_narrowing | l1_l2 | 0.018 | 0.192 | 0.790 |
| right_neural_foraminal_narrowing | l3_l4 | 0.188 | 0.677 | 0.135 |
| left_subarticular_stenosis | l3_l4 | 0.046 | 0.321 | 0.633 |
| right_subarticular_stenosis | l5_s1 | 0.444 | 0.455 | 0.101 |

## Leaderboard reference

Leaderboard data retrieved via `kaggle competitions leaderboard --show`
and `kaggle competitions leaderboard -d` (public LB CSV).

See `docs/run_logs/kaggle_leaderboard_comparison.md` for full comparison.

## Metadata

| Field | Value |
|---|---|
| Submission timestamp | N/A (submission rejected) |
| Public score | N/A (not submitted) |
| Private score | N/A (not submitted) |
| Submission CSV SHA-256 | see validation JSON |
| Model package SHA-256 | 452154208c346ea05529cd47f8913c54d22b150c87bd1b1ed1d612bffcedc9ba |
