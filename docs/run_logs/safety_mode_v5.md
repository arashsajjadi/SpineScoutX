# Safety Mode v5 — evidence-aware review + condition-specific calibration

> Research-only. Not diagnostic. `review_required` is a research signal, not triage.
> Temperature is fit on `dev` only; `test` is eval-only. Locked-test auto distribution.

Review score: v4 = calibrated confidence; v5 = confidence − 0.5·instability.

## Calibration (test ECE / Brier, before → after dev-fit temperature)
| condition | T | ECE | Brier | n / sev |
|---|---|---|---|---|
| spinal_canal_stenosis | 1.10 | 0.029→0.034 | 0.192→0.192 | 1480 / 53 |
| left_neural_foraminal_narrowing | 1.20 | 0.038→0.043 | 0.320→0.320 | 1480 / 52 |
| right_neural_foraminal_narrowing | 1.30 | 0.034→0.054 | 0.313→0.315 | 1470 / 53 |
| left_subarticular_stenosis | 1.10 | 0.079→0.083 | 0.413→0.414 | 1434 / 138 |
| right_subarticular_stenosis | 1.10 | 0.070→0.075 | 0.418→0.419 | 1434 / 137 |

## Evidence-aware review — severe-FN capture at matched review burden (v4 vs v5)
| condition | unstable | sevFN | budget | v4 conf-only | v5 conf+stability |
|---|---|---|---|---|---|
| spinal_canal_stenosis | 178 | 9 | 10% | 0.778 | 0.111 |
| spinal_canal_stenosis | 178 | 9 | 20% | 1.000 | 1.000 |
| spinal_canal_stenosis | 178 | 9 | 30% | 1.000 | 1.000 |
| left_neural_foraminal_narrowing | 538 | 11 | 10% | 0.636 | 0.455 |
| left_neural_foraminal_narrowing | 538 | 11 | 20% | 0.909 | 0.818 |
| left_neural_foraminal_narrowing | 538 | 11 | 30% | 1.000 | 1.000 |
| right_neural_foraminal_narrowing | 530 | 18 | 10% | 0.444 | 0.333 |
| right_neural_foraminal_narrowing | 530 | 18 | 20% | 0.667 | 0.722 |
| right_neural_foraminal_narrowing | 530 | 18 | 30% | 0.722 | 0.889 |
| left_subarticular_stenosis | 370 | 35 | 10% | 0.257 | 0.200 |
| left_subarticular_stenosis | 370 | 35 | 20% | 0.514 | 0.429 |
| left_subarticular_stenosis | 370 | 35 | 30% | 0.686 | 0.657 |
| right_subarticular_stenosis | 362 | 36 | 10% | 0.167 | 0.333 |
| right_subarticular_stenosis | 362 | 36 | 20% | 0.417 | 0.556 |
| right_subarticular_stenosis | 362 | 36 | 30% | 0.667 | 0.750 |

## Interpretation (honest, no overclaim)
- **Calibration negative.** Graders are already well-calibrated (test ECE 0.03–0.08);
  dev-fit temperature (T=1.1–1.3) does NOT transfer — calibrated test ECE ≤ uncalibrated
  on only **0/5** conditions (worsens the rest). **Deployed path keeps raw
  probabilities** (no temperature applied); reported, not hidden.
- **Evidence-aware review is MIXED (uniform λ).** v5 (confidence + stability) beats
  v4 (confidence only) severe-FN capture @20% review on **2/5** routes: right_neural_foraminal_narrowing, right_subarticular_stenosis
  — the weakest **right-side** routes (right-foraminal @30% 0.72→0.89; right-subarticular
  @20% 0.42→0.56). On the 3 strong routes confidence alone is as good or better, so a
  uniform stability penalty is NOT deployed globally.
- **Deployed v5 policy:** stability is an **inference-time** review reason
  (`evidence_unstable` / `axial_candidate_disagreement` / `foraminal_slice_disagreement`)
  and a `route_quality` flag; a measured severe-FN benefit on weak right-side routes.

Reproduce: `python scripts/run_safety_mode_v5.py` (after run_evidence_stability.py).
