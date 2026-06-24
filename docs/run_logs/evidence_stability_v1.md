# Evidence Stability v1 — prediction stability under localizer perturbation

> Research-only. Not diagnostic. Instability is a reliability signal, not triage advice.
> Perturbations (in-plane jitter + slice shift) are drawn from the localizer's own
> measured error scale and the AUTO centre/slice — **no GT coords**. Each finding is
> graded by re-running the SAME deployed grader on K plausible crops.

## Fidelity (baseline reproduces the deployed predictions)
Argmax agreement = fraction of sampled findings whose baseline severity argmax matches
`collect_probs` (the deployed path). Residual sub-1e-3 prob diffs are GPU conv
non-determinism, not a pipeline error.

| condition | n compared | argmax agreement | max |Δp| | reproduces deployed |
|---|---|---|---|---|
| spinal_canal_stenosis | 96 | 1.000 | 2.41e-04 | YES |
| left_neural_foraminal_narrowing | 96 | 1.000 | 2.87e-04 | YES |
| right_neural_foraminal_narrowing | 96 | 1.000 | 2.42e-04 | YES |
| left_subarticular_stenosis | 96 | 1.000 | 2.35e-04 | YES |
| right_subarticular_stenosis | 96 | 1.000 | 2.28e-04 | YES |

## Stability grade mix + does instability predict errors?
AUROC = probability instability ranks a wrong finding above a correct one (0.5 = no signal). 'combined' = mean of normalised (1-confidence) and instability.

| condition | n / sev | stable/mild/unstable | AUROC err·conf | AUROC err·instab | AUROC err·comb |
|---|---|---|---|---|---|
| spinal_canal_stenosis | 1480 / 53 | 1103 / 199 / 178 | 0.919 [0.899,0.935] | 0.843 [0.816,0.868] | 0.902 [0.882,0.920] |
| left_neural_foraminal_narrowing | 1480 / 52 | 644 / 298 / 538 | 0.822 [0.800,0.843] | 0.790 [0.766,0.815] | 0.826 [0.805,0.847] |
| right_neural_foraminal_narrowing | 1470 / 53 | 654 / 286 / 530 | 0.818 [0.793,0.841] | 0.781 [0.754,0.808] | 0.819 [0.794,0.841] |
| left_subarticular_stenosis | 1434 / 138 | 668 / 396 / 370 | 0.822 [0.801,0.843] | 0.803 [0.778,0.826] | 0.829 [0.806,0.850] |
| right_subarticular_stenosis | 1434 / 137 | 664 / 408 / 362 | 0.803 [0.775,0.829] | 0.777 [0.752,0.803] | 0.804 [0.779,0.830] |

**Pooled (all routes, n=7298):** AUROC error from confidence 0.845 [0.832,0.858], from instability 0.797 [0.783,0.812], **combined 0.841 [0.828,0.854]**.

## Triage uplift — severe-FN capture at matched review budget (per condition)
| condition | budget | confidence only | instability only | combined |
|---|---|---|---|---|
| spinal_canal_stenosis | 10% | 0.778 | 0.111 | 0.111 |
| spinal_canal_stenosis | 20% | 1.000 | 0.333 | 1.000 |
| spinal_canal_stenosis | 30% | 1.000 | 0.667 | 1.000 |
| left_neural_foraminal_narrowing | 10% | 0.545 | 0.273 | 0.273 |
| left_neural_foraminal_narrowing | 20% | 0.909 | 0.455 | 0.818 |
| left_neural_foraminal_narrowing | 30% | 0.909 | 0.727 | 0.818 |
| right_neural_foraminal_narrowing | 10% | 0.389 | 0.444 | 0.333 |
| right_neural_foraminal_narrowing | 20% | 0.611 | 0.500 | 0.722 |
| right_neural_foraminal_narrowing | 30% | 0.722 | 0.722 | 0.889 |
| left_subarticular_stenosis | 10% | 0.257 | 0.143 | 0.171 |
| left_subarticular_stenosis | 20% | 0.514 | 0.343 | 0.429 |
| left_subarticular_stenosis | 30% | 0.686 | 0.571 | 0.629 |
| right_subarticular_stenosis | 10% | 0.167 | 0.389 | 0.306 |
| right_subarticular_stenosis | 20% | 0.389 | 0.500 | 0.556 |
| right_subarticular_stenosis | 30% | 0.667 | 0.667 | 0.750 |

## Interpretation (honest, no overclaim)
- **Fidelity:** baseline argmax reproduces the deployed predictions exactly (agreement
  1.000 per condition); residual sub-1e-3 prob diffs are GPU conv non-determinism.
- **Stability is a real signal:** instability predicts a baseline error at pooled AUROC 0.797 and severe-FNs at 0.713 — both well above chance (0.5).
- **But it is largely redundant with confidence:** pooled `combined` AUROC 0.841 ≈ confidence 0.845; confidence dominates on the strong
  routes. We do **not** claim stability beats confidence in general.
- **Where it adds triage value:** at a matched 20% review budget, `combined` severe-FN capture exceeds confidence-only on **2/5** routes: right_neural_foraminal_narrowing, right_subarticular_stenosis — notably the weakest **right-side** routes.
- **Robust-training validation:** robust-trained graders are more stable (canal 75%) than the oracle-trained foraminal grader (44% stable, most unstable) — even though the
  foraminal localizer is cleaner. Stability buys robustness; oracle-trained graders are
  perturbation-sensitive.
- **Use:** stability feeds `route_quality` + the `evidence_unstable` /
  `axial_candidate_disagreement` / `foraminal_slice_disagreement` review reasons (Safety
  v5) and the finding-graph schema — an explanatory reliability signal with a measured
  triage benefit on right-side routes.

Reproduce: `python scripts/run_evidence_stability.py` (smoke: `--max-studies 30`).
