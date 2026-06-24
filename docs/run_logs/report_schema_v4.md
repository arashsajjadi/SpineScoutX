# Study-level finding-graph schema (v4)

> Research-only · not diagnostic. The formal, auditable structure of a SpineScoutX model
> output. Built/validated/rendered by `src/spinescoutx/reporting/finding_graph_schema.py`.

## Study object
`schema_version` · `report_type` (`non_diagnostic_research_finding_graph`) · `case_id`
(`case_<sha1(study_id)[:10]>`, anonymized) · `split` · `research_only` · `disclaimer` ·
`generated_at` · `model_version` · `route_version` · `supported_findings` ·
`unsupported_or_oracle_only_findings` · `findings[]` · `blocked_findings[]` ·
`study_summary`.

## Per finding
- `condition`, `base_condition`, `level`, `side`
- `view_route` ∈ {`sagittal_t2` (canal), `sagittal_t1` (foraminal), `axial_t2` (subarticular)}
- `crop_provenance` ∈ {`auto`, `oracle`, `blocked`}
- `severity_estimate` ∈ {`normal_mild`, `moderate`, `severe`} (= argmax of the probabilities)
- `probabilities`: `P(normal_mild)`, `P(moderate)`, `P(severe)` (real softmax, sum ≈ 1)
- `calibrated_confidence` (top-class probability)
- `uncertainty_flag` ∈ {`high_confidence`, `moderate_confidence`, `review_required`}
- `review_required` (bool) + `review_reasons[]` ⊆ {`low_confidence`, `high_entropy`,
  `model_disagreement`, `localizer_uncertainty`, `axial_level_uncertainty`, `view_missing`,
  `morphology_disagreement`, `near_severe_threshold`}
- `localizer`: `route`, `confidence`, `axial_level_scorer_score` (null where not computed)
- `evidence`: `view_used`, `crop_center_source`, `notes`
- `reference_label`: held-out research target (shown for transparency; **not** a model
  input or output)

## Study summary
`max_p_severe`, `n_findings`, `n_severe_estimates`, `n_review_required`,
`n_high_confidence`, `n_low_confidence`, `n_blocked`, `findings_requiring_review`, `warnings`.

## Review-reason logic (derived from real model signals — not templated)
`low_confidence` (conf < 0.60) · `high_entropy` (normalized entropy > 0.85) ·
`model_disagreement` (router grader vs its comparison grader argmax differ) ·
`near_severe_threshold` (not severe but P(severe) ≥ 0.20) · `axial_level_uncertainty`
(subarticular axial-level-scorer score < 0.50). `review_required` = any reason present.

## Guarantees (enforced by `validate_finding_graph`, tested)
Probabilities sum ≈ 1; `severity_estimate` equals the argmax of the stated probabilities;
every finding has a provenance; review reasons are from the allowed set and consistent with
`review_required`; `case_id` is an anonymized hash; **no diagnosis/treatment wording**
anywhere (disclaimer/negated forms are explicitly allowed; positive-claim roots forbidden).
The Markdown rendering is deterministic and reflects the JSON exactly.
