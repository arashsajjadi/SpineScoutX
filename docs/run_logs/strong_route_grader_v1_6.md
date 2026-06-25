# v1.6 Plan D — stronger route-specific foraminal grader: EXECUTED NEGATIVE

> Research-only · not diagnostic. A capacity test: **convnext_small** (vs the convnext_tiny
> baseline) + severe oversampling, RSNA splits_v1, dev-selected (foraminal-macro recall@FAR≤10%),
> locked-test once. Reproduce: `train_foraminal_grader_v1_6.py --backbone convnext_small
> --severe-over`; `compare_foraminal_transfer_v1_6.py --exp-tag rsna_strong`.

## Locked-test (paired vs baseline)

| side | baseline (convnext_tiny) | convnext_small + severe-over | paired Δ [95% CI] |
|---|---|---|---|
| L-for | 0.769 | 0.596 | **−0.173 [−0.302,−0.051] DECISIVE** |
| R-for | 0.679 | 0.472 | **−0.208 [−0.349,−0.085] DECISIVE** |
| macro | 0.724 | 0.534 | −0.190 |

## Verdict — NEGATIVE

More capacity does **not** help — it *hurts*. convnext_small + severe oversampling converges (like
the LSS-pretrained arm) to a **conservative argmax operating point** (low FAR, fewer argmax-severe),
collapsing argmax severe recall on the thin severe data; dev best (0.746) was already below the
convnext_tiny baseline (0.794). This reproduces the v1.5 finding that **capacity is not the
bottleneck** on the weak routes (candidate-bag MIL likewise overfit the thin severe data). The
binding constraint is severe-label quantity/quality, not grader size (see
`v1_6_failure_autopsy.md`).
