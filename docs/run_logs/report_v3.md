# Study-level report assistant v3 — multi-condition finding graph

> Research-only. Not diagnostic. Not clinically validated. Not for medical
> decision-making. No treatment recommendations. Severity values are **research findings /
> severity estimates**, never a diagnosis. Generated reports live under
> `outputs/real/reports_v3/` (gitignored, regenerable).

`scripts/run_report_v3.py` assembles a deterministic, non-diagnostic **finding graph** per
study across every **auto-supported** condition, on the locked-test auto distribution. It
selects the deployable grader per condition (canal → auto-trained robust; foraminal →
oracle-trained, which wins on auto — see `foraminal_auto_results.md`).

## Per-study schema (JSON + Markdown)
- `study_id`, `disclaimer`, `limitations`
- `auto_supported_conditions`: canal, left/right foraminal
- `blocked_conditions`: left/right subarticular (axial route not built — see
  `subarticular_auto_results.md`); each is emitted as `oracle_only_blocked`, never given a
  fabricated auto severity.
- `findings_auto[]` — one entry per (condition, side, level):
  - `severity_estimate` ∈ {normal_mild, moderate, severe}
  - `p_severe`, `confidence` (calibrated top-class probability)
  - `uncertainty_flag` ∈ {high_confidence, moderate_confidence, review_required}
  - `review_reasons` (e.g. `low_confidence`, `near_severe_threshold`)
  - `provenance`: `auto_crop (sagittal localizer; no GT at inference)`
  - `reference_label`: the held-out research target (for evaluation context, not an input)
- `study_review_required`: true if any finding is flagged for review.

## What it demonstrates
- Output reads like a research assistant: per-level severity estimates with calibrated
  confidence, explicit uncertainty/review reasons, and clear provenance.
- It **labels coverage honestly**: auto findings (3/5) vs blocked/oracle-only (2/5).
- No diagnosis, no treatment, no out-of-scope pathology. A `review_required` flag is a
  research triage signal, not clinical advice.

Optional local Ollama rewording (existing `reporting/llm_report.py`) may only rephrase the
deterministic graph and fails closed; it never invents findings, severities, or advice.

Reproduce: `python scripts/run_report_v3.py` (writes 12 example reports incl. severe and
review-flagged cases).
