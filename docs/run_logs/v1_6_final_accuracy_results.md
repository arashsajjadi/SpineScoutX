# v1.6 — final accuracy results (adaptive offensive)

> Research-only · not diagnostic · not clinically validated. Protocol: `splits_v1`; **dev** selects;
> **RSNA locked-test read once** per final model. Cluster bootstrap CIs, n_boot=2000. **No model is
> deployed from v1.6** — every intervention is executed-negative; the deployed 5/5 graders are
> unchanged.

## Headline (locked-test severe recall) — foraminal focus

| arm | L-for | R-for | foraminal macro | verdict |
|---|---|---|---|---|
| deployed reference | 0.788 | 0.660 | 0.724 | — |
| **v1.6 ImageNet baseline** | 0.769 [0.654,0.873] | **0.679** [0.559,0.800] | **0.724** | reproduces deployed |
| A — LSS pretrain → fine-tune | 0.577 | 0.528 | 0.553 | **decisive loss** |
| A — joint LSS+RSNA (+179 sev) | 0.769 | 0.679 | 0.724 | Δ0.000 (no change) |
| B — SSL → fine-tune | — | — | — | **non-convergent** (no usable encoder) |
| C — anatomy prior | — | — | — | executed (v0.4/v0.5); no grading gain |
| D — convnext_small + severe-over | 0.596 | 0.472 | 0.534 | **decisive loss** |

Paired locked-test deltas vs the v1.6 baseline (severe recall): transfer L −0.192 / R −0.151
(both decisive); joint ±0.000; stronger-grader L −0.173 / R −0.208 (both decisive).
recall@FAR≤10% (R-for): baseline 0.774; transfer 0.792 (+0.019 n.s.); joint 0.679; none material.

## Success criteria (vs deployed baseline)

| criterion | target | result |
|---|---|---|
| R-for severe recall | +≥0.05 | ✗ (best = baseline 0.679; no arm ≥ 0.710) |
| foraminal macro severe recall | +≥0.03 | ✗ (best = 0.724; no arm ≥ 0.754) |
| overall macro (5-route) | +≥0.03 | ✗ (unchanged 0.752; no deployed change) |
| high-confidence severe FN −≥20% | — | ✗ (no grader improved) |
| recall@FAR≤10% R-for/macro material | — | ✗ (R-for +0.019 n.s.) |
| **executed/blocked w/ exact evidence** | fallback | **✓ all four levers executed; autopsy + next-data spec** |

## Verdict

**ADAPTIVE EXECUTED NEGATIVE.** All four adaptive levers (external data, SSL, anatomy prior,
stronger grader) were genuinely executed; **none improved raw severe grading.** The clean baseline
reproduces the deployed foraminal performance, so the comparisons are sound. The deployed 5/5
graders (macro 0.752) ship **unchanged**. Root cause + the exact data needed to move the metric are
in `v1_6_failure_autopsy.md`; the per-experiment audit trail is in `v1_6_adaptive_controller.json`.

Tag: **`v1.6.0-adaptive-accuracy-negative-result`** (no raw metric improved → no accuracy-upgrade
tag, per policy).
