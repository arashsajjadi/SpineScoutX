# Right neural foraminal narrowing — hard-case audit + bounded refinement (locked test)

> Research-only. Not diagnostic. No GT coordinates at inference. Grader =
> v1_foraminal_oracle_ctrl. Bounded, no-retrain analysis (prior specialist non-decisive).

## Left vs right
- left severe recall **0.788** [0.667, 0.896]
- right severe recall **0.660** [0.526, 0.782] (recall@FAR10 0.811)

## Per-level severe recall (n_severe in parens)
| level | left | right |
|---|---|---|
| l1_l2 | nan (0) | nan (0) |
| l2_l3 | nan (0) | nan (0) |
| l3_l4 | 0.500 (2) | 0.250 (4) |
| l4_l5 | 0.941 (17) | 0.667 (18) |
| l5_s1 | 0.727 (33) | 0.710 (31) |

## Why right-side severe cases are missed
- 18/53 right severe findings missed; mean P(severe) **0.204** (median 0.138) vs 0.761 for caught cases.
- **56%** of misses are *confidently normal*
  (P(severe) < 0.2) — the grader is sure they are not severe, not borderline. A hard
  (largely threshold-irreducible) error, consistent with a sample-size / signal limit.

## Bounded experiment — per-level dev-tuned severe threshold (no test tuning)
- argmax: severe recall 0.660 at FAR 0.054
- per-level dev-tuned (FAR≤10% on dev): severe recall 0.660 at test FAR 0.088

## Deployable mitigation — evidence stability
- right-side severe FNs are more unstable (instability 0.441) than caught cases (0.279),
  which is why stability-aware review (Safety v5) improves right-foraminal severe-FN
  capture at matched review budget (0.72→0.89 @30%).

## Diagnosis (honest)
Right-foraminal trails left, but the gap is **sample-size / signal limited**, not tuning
artifact: most misses are confidently-normal severe cases that per-level thresholding can
not recover at an acceptable FAR, and the L/R CIs overlap (n_severe≈52–53). We do **not**
claim a decisive improvement. The **deployable gain** is evidence-stability-aware review,
which preferentially flags the unstable right-side misses for human research review.

Reproduce: `python scripts/run_right_foraminal_audit.py`.
