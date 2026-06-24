# SpineScoutX gallery

> Research-only · not diagnostic · not clinically validated. All figures are metric plots
> or schematic diagrams generated from locked-test result JSONs — **no DICOM pixels, no
> patient identifiers**. Regenerate with `python scripts/make_showcase_assets.py`.

## Pipeline
![pipeline](assets/hero_pipeline.png)

MRI study → view router + per-condition localizers → auto crop (no GT coordinates) →
robust per-condition grader → Safety Mode review layer → non-diagnostic finding graph.

## Auto coverage (5/5 findings, locked test)
![coverage](assets/coverage_dashboard.png)

Bars are auto (real-inference) severe recall on the patient-level locked test, with 95%
CIs. All five RSNA findings have a real auto route.

## Oracle upper bound vs auto inference
![oracle vs auto](assets/oracle_vs_auto_gap.png)

The oracle (GT-coordinate) bars are an upper bound; the auto bars are deployable
inference. The canal gap is large (robust auto-training recovers it); the foraminal gap is
small (clean localizer → oracle-trained transfers).

## Safety Mode — severe recall vs review burden
![safety](assets/safety_mode_dashboard.png)

For each auto condition: effective severe recall as a function of how many low-confidence /
disagreement cases are flagged `review_required`. Reaching high severe recall has an
explicit review-burden cost, shown plainly.

## Example finding graphs
Non-diagnostic per-study research finding graphs (per-level severity estimate, P(severe),
confidence, uncertainty flag, review reasons, provenance) are written to
`outputs/real/reports_v3/*.md` / `*.json` (gitignored, regenerable via
`python scripts/run_report_v3.py`). Schema: [`run_logs/report_v3.md`](run_logs/report_v3.md).

## Where the numbers come from
- Canal locked-test confirmation: [`run_logs/canal_locked_test.md`](run_logs/canal_locked_test.md)
- Foraminal auto route: [`run_logs/foraminal_auto_results.md`](run_logs/foraminal_auto_results.md)
- Axial subarticular auto route: [`run_logs/subarticular_auto_results.md`](run_logs/subarticular_auto_results.md)
- Safety Mode v4 dashboard: [`run_logs/safety_mode_v4.md`](run_logs/safety_mode_v4.md)
- Full results + CIs: [`results.md`](results.md)
