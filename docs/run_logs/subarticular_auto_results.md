# Axial subarticular route — level-matching feasibility (locked test)

> Research-only. Not diagnostic. Subarticular stenosis is graded on **axial-T2**.
> The gating sub-problem is assigning each lumbar level to the correct axial slice
> with no GT. This QC measures the z-based matcher (sagittal-disc-z → axial-slice-z)
> against the GT subarticular axial instance (GT used for evaluation only).

Studies scored: 200 (924 level matches); skipped 0.

## Matched-vs-GT axial slice distance

| metric | value |
|---|---|
| median |Δslice| | 2.0 |
| mean |Δslice| | 3.19 |
| within 0 slices | 0.084 |
| within 1 slice | 0.275 |
| within 2 slices | 0.509 |
| median |Δz| (mm) | 12.8 |

## Verdict
- **Level matching is not yet reliable enough** for a headline auto subarticular
  result (see distances above). Likely causes: cross-series geometry mismatch,
  multiple axial stacks, or oblique acquisitions. This is the precise, measured
  **blocker**: the axial route needs a more robust level matcher (e.g.,
  coordinate-supervised slice scoring) before an axial grader is worth training.

Either way: **no faked auto subarticular metric is reported.** Oracle locked-test
subarticular baselines (upper bounds) remain in `multicondition_robust_results.md`.

Artifacts: `outputs/real/axial_matching_qc.json`. Reproduce:
`python scripts/run_axial_feasibility.py`.
