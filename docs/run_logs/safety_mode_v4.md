# Safety Mode v4 — multi-condition severe-first dashboard + router (locked-test auto)

> Research-only. Not diagnostic. Not for medical decision-making. `review_required` is a
> research signal, not triage advice. Auto distribution, locked `test`; 5/5 conditions have a real auto route; cluster-bootstrap CIs.

Router (deployable grader per condition): spinal→v1_canal_auto_robust, left→v1_foraminal_oracle_ctrl, right→v1_foraminal_oracle_ctrl, left→v1_subarticular_auto_robust, right→v1_subarticular_auto_robust

| condition | status | n / sev | argmax sevR | recall@FAR10 [CI] | FAR@90% | review→FN |
|---|---|---|---|---|---|---|
| spinal_canal_stenosis | auto | 1480 / 53 | 0.830 | 0.943 [0.833, 1.000] | 0.085 | 13%→22% |
| left_neural_foraminal_narrowing | auto | 1480 / 52 | 0.788 | 0.885 [0.795, 0.962] | 0.123 | 16%→45% |
| right_neural_foraminal_narrowing | auto | 1470 / 53 | 0.660 | 0.811 [0.681, 0.898] | 0.255 | 15%→28% |
| left_subarticular_stenosis | auto | 1434 / 138 | 0.746 | 0.732 [0.601, 0.808] | 0.252 | 30%→46% |
| right_subarticular_stenosis | auto | 1434 / 137 | 0.737 | 0.708 [0.602, 0.779] | 0.258 | 29%→36% |

Review reasons: low top-class confidence / high entropy (abstention curve in JSON) +
model disagreement (router grader vs its comparison). Cost-sensitive training not used
(prior honest negative). Reproduce: `python scripts/run_safety_mode_v4.py`.
