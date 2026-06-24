# Evidence Intelligence v2 — instability typing (locked-test auto, subsample)

> Research-only · not diagnostic. Each finding's instability is attributed to a cause
> by isolating slice-only vs in-plane-only perturbation (no GT to perturb). Subsample of
> 70 studies/condition; 1727 findings.

## Instability type distribution (all routes)
| type | count | severe-FN rate (n_FN/n) |
|---|---|---|
| stable | 1044 | 0.008 (8/1044) |
| crop_sensitive | 145 | 0.041 (6/145) |
| slice_sensitive | 201 | 0.020 (4/201) |
| axial_candidate_sensitive | 189 | 0.032 (6/189) |
| route_sensitive | 148 | 0.054 (8/148) |

## Instability type by condition
| condition | dominant types |
|---|---|
| spinal_canal_stenosis | slice_sensitive:48, route_sensitive:12, crop_sensitive:11 |
| left_neural_foraminal_narrowing | slice_sensitive:76, crop_sensitive:57, route_sensitive:36 |
| right_neural_foraminal_narrowing | slice_sensitive:77, crop_sensitive:45, route_sensitive:37 |
| left_subarticular_stenosis | axial_candidate_sensitive:91, route_sensitive:39, crop_sensitive:11 |
| right_subarticular_stenosis | axial_candidate_sensitive:98, route_sensitive:24, crop_sensitive:21 |

## Interpretation (honest)
- Instability typing localizes the **cause** of an unstable finding (crop vs
  slice/level vs mixed), which v1's scalar score could not. It feeds the case viewer's
  `instability_type` and route-specific review reasons.
- Highest severe-FN rate is among **route_sensitive** findings — i.e. that perturbation
  cause concentrates the missed-severe cases, the most informative review trigger.
- Explanatory/triage enrichment; it does not change any prediction. Subsample sizes mean
  per-type rates are indicative, not decisive (reported, not over-interpreted).

Reproduce: `python scripts/run_evidence_intel_v2.py --max-studies 70`.
