# SpineScoutX gallery — model outputs, routing, and failures

> Research-only · not diagnostic · not clinically validated · not for medical
> decision-making. Every card renders a **structured model output** (a finding graph) or a
> metric plot — **no DICOM pixels, no patient identifiers** (cases are hashed `case_*`).
> Regenerate: `python scripts/make_model_output_showcase.py` and
> `python scripts/make_showcase_assets.py`.

## 1. What the model receives → what it does
![pipeline](assets/hero_pipeline.png)

An MRI study → a **view router + per-condition localizers** → auto crops placed **without
ground-truth coordinates** → robust per-condition graders → Safety Mode review layer →
a non-diagnostic finding graph.

## 2. How routing works
| finding | view route | localization |
|---|---|---|
| spinal canal stenosis | sagittal-T2 | disc localizer (median 2.5 px) |
| L/R neural foraminal narrowing | sagittal-T1 | side-aware best-slice localizer (median 2.2 px) |
| L/R subarticular stenosis | axial-T2 | coordinate-supervised axial level scorer + fixed offset |

## 3. The output: a study-level finding graph
This is what comes out — per-level/per-condition severity estimate, P(severe), calibrated
confidence, uncertainty flag, review reasons, view route, provenance:

![finding graph](assets/showcase/finding_graph_example.png)
![schema](assets/showcase/report_schema_visual.png)

## 4. Example cases (real locked-test model outputs)
| | card |
|---|---|
| canal — severe | ![](assets/showcase/case_canal_severe_card.png) |
| left foraminal | ![](assets/showcase/case_foraminal_left_card.png) |
| left subarticular | ![](assets/showcase/case_subarticular_left_card.png) |
| right subarticular | ![](assets/showcase/case_subarticular_right_card.png) |

*Interpretation:* confident findings are flagged `high_confidence`; borderline ones get
`review_required` with an explicit reason. Provenance is `auto` (no GT at inference).
`ref` is the held-out research target (transparency only).

## 5. Evidence-aware intelligence (v1.1)
Each finding is re-graded under K plausible localizer perturbations (no GT) to score how
**stable** the prediction is. Cards carry a left **stability stripe** (green=stable,
amber=mildly, red=unstable), a `route_quality` flag, and stability-driven review reasons.

| | card |
|---|---|
| stable, high-confidence (`route_quality` good) | ![](assets/showcase/case_stable_high_confidence_card.png) |
| **unstable** finding flagged by evidence stability | ![](assets/showcase/case_unstable_flagged_card.png) |

![evidence stability](assets/showcase/evidence_stability_dashboard.png)

Honest result: instability predicts errors (AUROC 0.80) but is largely redundant with
confidence; it **adds** severe-FN triage value on the 2 weakest right-side routes.
Robust-trained graders are more stable than the oracle-trained foraminal grader.

## 6. Safety Mode (evidence-aware v5)
![safety](assets/safety_mode_dashboard.png)
![safety v5](assets/showcase/safety_mode_v5_dashboard.png)

Per condition, effective severe recall vs the `review_required` burden — reaching high
severe recall has an explicit cost, shown plainly. v5 adds an `evidence_unstable` review
reason; calibration is a documented negative (already well-calibrated → raw probs kept).

## 7. Failure & uncertainty gallery (shown, not hidden)
| case | what it shows | limitation |
|---|---|---|
| ![](assets/showcase/case_review_required_card.png) | many borderline findings → `review_required` | the review layer is deliberately conservative (over-flags for safety) |
| ![](assets/showcase/case_foraminal_right_hard_card.png) | right-foraminal hard case | right-foraminal trails left (within CI overlap) |
| ![](assets/showcase/case_mostly_normal_card.png) | mostly normal/mild, **0 reviews** | shows the flag is selective, not always-on |
| ![](assets/showcase/right_foraminal_hard_cases.png) | right-foraminal failure analysis: 56% of misses are confidently-normal | a signal/sample limit, not threshold-fixable; worst at L4-L5 right |
| ![](assets/showcase/domain_shift.png) | internal domain-shift stress test | weaker at L5-S1 (0.579) vs L4-L5 (0.868); no external validation |

Coverage and oracle-vs-auto context:
![coverage](assets/coverage_dashboard.png)
![oracle vs auto](assets/oracle_vs_auto_gap.png)

## 8. What NOT to use this for
**Not** a diagnosis. **Not** clinical decision-making. **No** treatment recommendations.
A research prototype on public non-commercial data; no external/prospective/reader-study
validation. See [`trust_and_limitations.md`](trust_and_limitations.md).

## Where the numbers come from
Full pack (JSON/MD, gitignored): `outputs/real/showcase_reports/`. Schema:
[`run_logs/report_schema_v5.md`](run_logs/report_schema_v5.md). Audit:
[`run_logs/output_intelligence_audit.md`](run_logs/output_intelligence_audit.md). Results +
CIs: [`results.md`](results.md).
