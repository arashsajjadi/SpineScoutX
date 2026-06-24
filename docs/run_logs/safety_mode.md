# Severe-first Safety Mode (Phase 6) — on the AUTO distribution

> Research-only. Not diagnostic. Not clinically validated. Not for medical
> decision-making. No treatment recommendations. A "review" flag is a research signal,
> not triage advice. Canal val, auto (real-inference) crops; n=1955, severe=87.

Safety Mode is the **decision layer** on top of a trained grader (no retraining): it
turns calibrated per-node probabilities into severe-first behaviour and reports the
honest trade-offs. Compared here: the oracle-trained control vs the robust winner
(`r_auto_train`). `outputs/real/safety_mode_frontier.json`.

## Operating points (auto)

| metric | baseline (oracle-trained control) | **robust (`r_auto_train`)** |
|---|---|---|
| balanced (argmax) severe recall | 0.667 (FAR 0.053) | **0.793 (FAR 0.064)** |
| severe recall ≥ 0.90 reachable at FAR | 0.192 | **0.153** |
| recall @ FAR ≤ 10% [95% CI] | 0.816 [0.714, 0.893] | **0.851 [0.766, 0.924]** |

## Abstention / review policy (auto)

A node is sent to review when its top-class confidence is low; a severe finding is
"safe" if the model calls it severe **or** it is sent to review.

| target | baseline | robust (`r_auto_train`) |
|---|---|---|
| review burden for **effective** severe recall ≥ 0.90 | abstain 20.9% | abstain 19.9% |
| of those, share of model severe-FNs captured by review | 75.9% | 61.1% |

(The robust model needs review to capture a *smaller* share of severe-FNs because it
makes fewer severe-FNs to begin with.)

## Honest reading

- The robust model gives a **better severe-first frontier on the real auto
  distribution**: higher balanced severe recall at comparable false-alarm rate, and it
  reaches 90% severe recall at a lower false-alarm cost (15.3% vs 19.2%).
- **Reaching 90% severe recall is not free.** It costs either ~15% false alarms (pure
  thresholding) or ~20% human-review burden (abstention). We report the cost, not just
  the recall. At argmax the robust model already gives 0.793 severe recall at ~6% FAR.
- A **cost-sensitive training loss** (`ExpectedCostLoss`, severe-FN ≫ FP) is implemented
  and unit-tested as an additional lever (`loss='cost_sensitive'`); the headline Safety
  Mode result above is the inference-time decision layer, which needs no retraining.

## What would count as a real improvement (met)

Higher severe recall **at fixed false-alarm budget on the auto distribution**, with CIs
and an explicit false-positive / review tradeoff. `r_auto_train` delivers recall@FAR≤10%
0.851 [0.766, 0.924] vs the control 0.816, and a lower false-alarm cost to reach 90%
severe recall — reported with the abstention burden, nothing hidden.

Artifacts: `outputs/real/safety_mode_frontier.json`,
`outputs/real/figures/safety_mode_frontier.png`. Reproduce:
`python scripts/run_safety_mode.py`.
