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

## 5. Safety Mode
![safety](assets/safety_mode_dashboard.png)

Per condition, effective severe recall vs the `review_required` burden — reaching high
severe recall has an explicit cost, shown plainly.

## 6. Failure & uncertainty gallery (shown, not hidden)
| case | what it shows | limitation |
|---|---|---|
| ![](assets/showcase/case_review_required_card.png) | many borderline findings → `review_required` | the review layer is deliberately conservative (over-flags for safety) |
| ![](assets/showcase/case_foraminal_right_hard_card.png) | right-foraminal hard case | right-foraminal trails left (within CI overlap) |
| ![](assets/showcase/case_mostly_normal_card.png) | mostly normal/mild, **0 reviews** | shows the flag is selective, not always-on |

Coverage and oracle-vs-auto context:
![coverage](assets/coverage_dashboard.png)
![oracle vs auto](assets/oracle_vs_auto_gap.png)

## 7. What NOT to use this for
**Not** a diagnosis. **Not** clinical decision-making. **No** treatment recommendations.
A research prototype on public non-commercial data; no external/prospective/reader-study
validation. See [`trust_and_limitations.md`](trust_and_limitations.md).

## Where the numbers come from
Full pack (JSON/MD, gitignored): `outputs/real/showcase_reports/`. Schema:
[`run_logs/report_schema_v4.md`](run_logs/report_schema_v4.md). Audit:
[`run_logs/output_intelligence_audit.md`](run_logs/output_intelligence_audit.md). Results +
CIs: [`results.md`](results.md).
