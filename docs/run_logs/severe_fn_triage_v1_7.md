# v1.7 severe-FN triage / review-required fallback (Phase 8) — SAFETY UPGRADE

> Research-only · not diagnostic. The deployed grader is **unchanged**. A triage risk model is fit
> on **dev** (cross-model disagreement + ensemble-minus-deployed p_severe + entropy + p_normal_mild)
> and evaluated on **locked-test once**, routing the riskiest foraminal findings to human review.
> This is a **safety/triage** upgrade (effective recall under a review budget), **not** a raw
> accuracy upgrade. Reproduce: `train_severe_fn_triage_v1_7.py`.

## Locked-test (n=2950 foraminal; 105 severe; 29 deployed severe-FN, 6 high-confidence)

Deployed argmax foraminal severe recall (no triage): **0.724**.

| review budget | review burden | severe-FN captured | high-conf FN captured | **effective severe recall** |
|---|---|---|---|---|
| 5% | 0.05 | 10/29 (0.34) | 0/6 | 0.819 |
| 10% | 0.10 | 15/29 (0.52) | 0/6 | 0.867 |
| **15%** | 0.15 | **22/29 (0.76)** | 3/6 | **0.933** |
| 20% | 0.20 | 24/29 (0.83) | 3/6 | **0.952** |

## Verdict — SAFETY UPGRADE

At a **15% review budget the triage captures 76% of foraminal severe false negatives**, raising
*effective* severe recall from 0.724 to **0.933** (0.952 at 20%) while the deployed grader stays
fixed. This is a real, deployable safety improvement for severe-FN reduction via selective review.

**Honest limit:** the **6 high-confidence severe-FN** (true severe predicted *confidently*
normal_mild) are largely **not** captured (0/6 at 5–10%, 3/6 at 15–20%) — every model agrees they
look normal. These are exactly the label-ambiguity core from the v1.6 autopsy: probable mislabels /
genuinely subtle cases that triage cannot surface and only **human re-annotation** (the review pack)
can resolve. Triage reduces *recoverable* severe-FN; it does not fix label noise.
