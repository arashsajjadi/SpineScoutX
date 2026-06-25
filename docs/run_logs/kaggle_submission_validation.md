# Kaggle Submission Validation — SpineScoutX v1.9

> Research-only · non-commercial · not diagnostic.

## Validation result: PASSED ✅

```
python scripts/generate_kaggle_submission_v1_9.py
```

## Output file

`submissions/spinescoutx_v1_9_late_submission.csv`

- **25 rows** (1 study × 5 conditions × 5 levels)
- **4 columns**: row_id, normal_mild, moderate, severe
- No NaN values
- No duplicate row_ids
- No missing or extra row_ids vs sample_submission.csv
- Probability sums: mean=1.000000, std=3.42e-08 (within float32 tolerance)

## Inference summary

| Route | Localizer | Grader | Levels predicted |
|---|---|---|---|
| Canal (sagittal T2/STIR) | l0_disc_localizer_real | v1_canal_auto_robust | 5/5 ✅ |
| Foraminal L (sagittal T1) | lf_foraminal_localizer | v1_foraminal_oracle_ctrl | 5/5 ✅ |
| Foraminal R (sagittal T1) | lf_foraminal_localizer | v1_foraminal_oracle_ctrl | 5/5 ✅ |
| Subarticular L (axial T2) | axial_level_scorer | v1_subarticular_auto_robust | 5/5 ✅ |
| Subarticular R (axial T2) | axial_level_scorer | v1_subarticular_auto_robust | 5/5 ✅ |

## Sample probabilities (model output, not uniform)

| row_id | normal_mild | moderate | severe |
|---|---|---|---|
| 44036939_spinal_canal_stenosis_l3_l4 | 0.025 | 0.238 | 0.737 |
| 44036939_left_neural_foraminal_narrowing_l4_l5 | 0.039 | 0.443 | 0.519 |
| 44036939_right_neural_foraminal_narrowing_l3_l4 | 0.188 | 0.677 | 0.135 |
| 44036939_left_subarticular_stenosis_l3_l4 | 0.046 | 0.321 | 0.633 |

Probabilities are real model predictions, not uniform prior.

## Notes

- All 5 localizations succeeded (no fallbacks to uniform).
- Kaggle metric is weighted log loss, not severe recall.
- This is a late submission; not an official competition entry.
