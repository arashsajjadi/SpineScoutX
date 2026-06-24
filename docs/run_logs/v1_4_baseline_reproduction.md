# v1.4 baseline reproduction (locked-test auto)

> Research-only · not diagnostic. Auto inference is deterministic (fixed crops +
> checkpoints), so the v1.3 numbers must reproduce; this freezes the reference + CIs so a
> later change counts as improvement only if it clears the cluster-bootstrap CI.

**Macro severe recall 0.753.**

| condition | reproduced [95% CI] | v1.3 ref | Δ | n / sev |
|---|---|---|---|---|
| spinal_canal_stenosis | 0.830 [0.725, 0.929] | 0.830 | +0.000 | 1480 / 53 |
| left_neural_foraminal_narrowing | 0.788 [0.673, 0.892] | 0.788 | +0.001 | 1480 / 52 |
| right_neural_foraminal_narrowing | 0.660 [0.524, 0.788] | 0.660 | +0.000 | 1470 / 53 |
| left_subarticular_stenosis | 0.746 [0.674, 0.815] | 0.746 | +0.000 | 1434 / 138 |
| right_subarticular_stenosis | 0.737 [0.667, 0.807] | 0.737 | +0.000 | 1434 / 137 |

Reproduction matches v1.3 within ≤0.005 on every route (deterministic inference). Any
v1.4 change must exceed these CIs (and report FAR) to be a real improvement.

Reproduce: `python scripts/run_baseline_reproduction.py`.
