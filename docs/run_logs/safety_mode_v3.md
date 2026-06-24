# Safety Mode v3 — multi-condition severe-first dashboard (locked-test auto)

> Research-only. Not diagnostic. Not for medical decision-making. `review_required`
> is a research signal, not triage advice. Every row is the auto (real-inference)
> distribution on the locked `test`, cluster-bootstrap 95% CIs.

| condition | status | n / sev | argmax sevR | recall@FAR10 [CI] | FAR@90% | review→FN |
|---|---|---|---|---|---|---|
| spinal_canal_stenosis | auto | 1480 / 53 | 0.830 (FAR 0.078) | 0.943 [0.833, 1.000] | 0.085 | 13.4%→22% |
| left_neural_foraminal_narrowing | auto | 1480 / 52 | 0.788 (FAR 0.060) | 0.885 [0.795, 0.962] | 0.123 | 16.5%→45% |
| right_neural_foraminal_narrowing | auto | 1470 / 53 | 0.660 (FAR 0.054) | 0.811 [0.681, 0.898] | 0.255 | 14.8%→28% |
| left_subarticular_stenosis | oracle-only (axial route not built; see subarticular_auto_results.md) | - | - | - | - | - |
| right_subarticular_stenosis | oracle-only (axial route not built; see subarticular_auto_results.md) | - | - | - | - | - |

## Review reasons (per the decision layer)
low top-class confidence / high entropy (abstention curve in JSON); **model
disagreement** between the auto-robust and control graders (column above: review
rate → fraction of robust severe-FNs captured). Cost-sensitive *training* is NOT
used (prior honest negative); severe-safety comes from robust auto-training + the
threshold frontier + this review layer.

Subarticular L/R are **oracle-only** (axial route not built; see
`subarticular_auto_results.md`). Artifacts: `outputs/real/safety_mode_v3.json`.
Reproduce: `python scripts/run_safety_mode_v3.py`.
