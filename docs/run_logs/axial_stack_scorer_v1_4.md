# Axial stack scorer v1.4 — does better localization raise subarticular severe recall?

> Research-only · not diagnostic. Paired test: score each axial stack once, decode with
> current decoder vs the v1.3 positional-prior decoder (`assign_levels_monotonic_prior`,
> β=1.0 dev-selected), re-crop subarticular evidence at each, re-grade with the DEPLOYED
> grader (fixed). No CNN retrain, no GT coordinates, no locked-test tuning. Severe recall
> via GT severity only.

| split | decoder | severe recall [95% CI] | n / sev |
|---|---|---|---|
| dev | current | 0.689 [0.624, 0.749] | 2857 / 273 |
| dev | prior | 0.692 [0.621, 0.757] | 2857 / 273 |
| test | current | 0.742 [0.681, 0.797] | 2868 / 275 |
| test | prior | 0.702 [0.628, 0.766] | 2868 / 275 |

## Verdict (honest)
- dev Δ(prior−current) = +0.004; test Δ = -0.040.
- The deployed grader is **robust to the leveling change** — re-cropping at the better decoder's slice does **not** materially move subarticular severe recall (Δ / within CI). Honest negative: localization improved (v1.3) but the grading payoff is bounded (the robust grader already tolerates leveling noise). Raw subarticular recall grader/data-limited, not decode-limited.

Reproduce: `python scripts/run_subarticular_recrop_v1_4.py`.
