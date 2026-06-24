# Locked patient-level test protocol (splits_v1)

> Research-only. Not diagnostic. The **locked test** is used for FINAL evaluation
> only — never for model selection or tuning. `dev` is for selection/tuning.
> Historical (seed-1337 val) results are preserved separately and are NOT v1 claims.

Patient-level (study_id) three-way split, seed 20260623, dev=15% / test=15%.
Studies: train 1382 / dev 296 / test 296 (total 1974). Disjoint by construction (one study → one split; `assert_disjoint` + `check_no_leakage`).

## Severe counts per split per condition (oracle crops)

| condition | train n / severe | dev n / severe | **test n / severe** |
|---|---|---|---|
| left_neural_foraminal_narrowing | 6910 / 286 | 1470 / 59 | **1480 / 52** |
| left_subarticular_stenosis | 6743 / 643 | 1431 / 131 | **1434 / 138** |
| right_neural_foraminal_narrowing | 6909 / 277 | 1470 / 48 | **1480 / 53** |
| right_subarticular_stenosis | 6747 / 646 | 1431 / 142 | **1434 / 137** |
| spinal_canal_stenosis | 6835 / 348 | 1455 / 68 | **1463 / 53** |

## Protocol rules
- Train on `train`; select hyperparameters / checkpoints on `dev`; report final
  numbers on `test` (locked) exactly once per model.
- Every v1 headline table states split (`dev`/`test`) and provenance (oracle/auto),
  n, n_severe, and a bootstrap CI.
- Models claimed on the locked test are **retrained on `train`** (the historical
  E0/E2/E3/r_* checkpoints trained on the seed-1337 split, which overlaps this test
  set, so they are NOT eligible for locked-test claims).

Artifacts: `data/cache/splits_v1/splits.json`, `stratified_counts.json` (gitignored).
Reproduce: `python scripts/build_splits_v1.py`.
