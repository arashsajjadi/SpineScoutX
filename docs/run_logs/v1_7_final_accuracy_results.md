# v1.7 — final results (hard-case label repair + noise-aware + triage)

> Research-only · not diagnostic · not clinically validated. Protocol: `splits_v1`; **dev** selects;
> **RSNA locked-test read once** per final model; raw RSNA labels never modified; locked-test labels
> never used for cleaning/training/selection. **No model is deployed from v1.7** (no raw accuracy
> gain); the deployed 5/5 graders are unchanged (5-route macro 0.752).

## Raw accuracy (locked-test foraminal severe recall) — NO improvement

| arm | L-for | R-for | foraminal macro |
|---|---|---|---|
| deployed reference | 0.788 | 0.660 | 0.724 |
| v1.6 ImageNet baseline | 0.769 | 0.679 | 0.724 |
| noise-aware (dev-best = mode A, original labels) | ≈0.769 | ≈0.679 | ≈0.724 |
| teacher-distilled student | 0.538 | 0.453 | **0.496** (collapse) |

Provisional label cleaning did **not** beat original labels on **dev** (mode A 0.792 > soft modes
0.750), so dev-selection kept the original-label model (≡ baseline). The teacher student overfit dev
and collapsed on test. **No raw severe-recall improvement on any route.**

## Severe-FN triage — SAFETY UPGRADE (the v1.7 win)

Deployed grader unchanged; triage fit on dev, locked-test once (n=2950 foraminal, 29 severe-FN):

| review budget | severe-FN captured | effective severe recall |
|---|---|---|
| 5% | 10/29 (0.34) | 0.819 |
| 10% | 15/29 (0.52) | 0.867 |
| **15%** | **22/29 (0.76)** | **0.933** |
| 20% | 24/29 (0.83) | 0.952 |

**At 15% review burden the triage lifts effective foraminal severe recall 0.724 → 0.933.** This is
a deployable *safety* upgrade (selective review of severe-FN-risk findings), not a raw-accuracy
upgrade.

## Data product

A real **704-case local-only review pack** (right-foraminal 338 / left 366; 87 right-foraminal
severe FN) for expert re-annotation — the main label-quality deliverable, awaiting human labels.

## Verdict / tag

No raw locked-test severe-recall improvement (label cleaning + teacher both negative) → **no
accuracy-upgrade tag**. The triage review metrics improve materially → **`v1.7.0-triage-safety-
upgrade`**, plus a complete hard-case review pack + provisional-cleaning + noise-aware + teacher all
executed. Autopsy + exact human-review handoff: `v1_7_failure_autopsy.md`, `v1_7_review_needed.md`.
