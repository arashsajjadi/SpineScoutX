# Safety Mode v2 — cost-sensitive training + decision layer (locked-test auto)

> Research-only. Not diagnostic. Not for medical decision-making. A `review_required`
> flag is a research signal, not triage advice. Canal, locked `test`, auto (real)
> distribution; n=1480, severe=53; cluster-bootstrap 95% CIs.

Three graders, all selected on `dev` auto and evaluated ONCE on locked `test` auto:
oracle-trained control, auto-trained robust (v0.9 recipe), and **cost-sensitive**
(ExpectedCostLoss, severe-FN ≫ FP) trained on auto crops.

| model | argmax severe recall | recall@FAR≤10% [95% CI] | FAR@90% sevR | cost |
|---|---|---|---|---|
| oracle_ctrl | 0.434 (FAR 0.011) | 0.868 [0.776, 0.959] | 0.155 | 0.185 |
| auto_robust | 0.830 (FAR 0.078) | 0.943 [0.833, 1.000] | 0.085 | 0.220 |
| cost_sensitive | 0.019 (FAR 0.007) | 0.264 [0.129, 0.386] | 0.538 | 0.966 |

## Review policy (reasons)
Beyond low-confidence/high-entropy abstention (see `abstention_curve` in the JSON), a
**model-disagreement** review reason flags nodes where the robust and control graders
disagree: that flags **13.4%** of nodes and captures **22.2%** of the robust model's severe
false-negatives — a cheap, transparent triage signal.

## Honest verdict
Cost-sensitive training is reported alongside the auto-robust recipe; whichever wins the
severe frontier on the locked test (with non-overlapping CI) is preferred, otherwise they
are called comparable. Reaching 90% severe recall has an explicit false-alarm cost;
if it is high, that is stated, not hidden.

Artifacts: `outputs/real/safety_mode_v2.json`, `figures/safety_mode_v2_frontier.png`.
Reproduce: `python scripts/run_safety_mode_v2.py`.
