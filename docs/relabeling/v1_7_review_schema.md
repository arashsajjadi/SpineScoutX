# v1.7 hard-case re-annotation schema

> Research-only · not diagnostic · the review collects expert labels, it does **not** make a
> diagnosis. The review pack itself (`review_packs/v1_7_hard_cases/`) is **local-only / gitignored**
> (it contains imaging pixels). This schema + a pixel-free synthetic example are the committed
> artifacts.

## Per-case fields (review_sheet.csv / review_items.jsonl)

`case_id` (anonymized sha1), `key` (study|level|condition, local-only), `study_id` (local-only),
`finding`, `side`, `level`, `split` (train/dev only — **no locked-test cases**),
`current_rsna_label`, `deployed_pred`, `deployed_p_severe`, `deployed_p_normal_mild`,
`deployed_entropy`, `model_disagreement`, `p_severe_<model>` (v1.6 baseline/LSS/joint/small),
`reason_selected` (group), `priority`, `panel` (local PNG path).

## Review questions

1. `target_truly_severe` — is the target finding truly severe? (yes/no)
2. `if_not_severe_grade` — if not severe: moderate or normal_mild?
3. `is_ambiguous` — moderate/severe genuinely ambiguous? (yes/no)
4. `evidence_insufficient` — is the visible evidence insufficient? (yes/no)
5. `side_level_correct` — is side/level correct? (yes/no)
6. `exclude_from_training` — should this sample be excluded from training? (yes/no)
7. `reviewer_note` — optional free text.

## Allowed review labels

`normal_mild` · `moderate` · `severe` · `ambiguous_moderate_severe` · `insufficient_evidence` ·
`exclude_from_training`

## Pixel-free synthetic example (committed)

```json
{"case_id":"case_0000example","finding":"right_neural_foraminal_narrowing","side":"right",
 "level":"l5_s1","split":"train","current_rsna_label":"severe","deployed_pred":"normal_mild",
 "deployed_p_severe":0.04,"deployed_p_normal_mild":0.71,"model_disagreement":0.06,
 "reason_selected":"B_confident_normal_severe_miss","priority":5.7,
 "review_label":"<reviewer fills: severe|moderate|normal_mild|ambiguous_moderate_severe|...>"}
```

## Ingestion

A completed `review_sheet_reviewed.csv` (with a `review_label` column) is ingested by
`scripts/ingest_review_labels_v1_7.py`, which validates IDs/labels, **versions corrected labels
separately** (never overwrites raw RSNA labels), and **never** ingests locked-test cases.
