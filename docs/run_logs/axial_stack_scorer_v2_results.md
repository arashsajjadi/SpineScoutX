# Axial decode v2 — positional-prior monotonic decoding (localization)

> Research-only · not diagnostic. A TRAIN-derived per-level normalized-z prior is added
> to the monotonic decode (`assign_levels_monotonic_prior`) — no CNN retrain, no test
> data in the prior. Beta selected on DEV, locked-test evaluated once. GT used to score
> slice-hit only (localizer evaluation, not auto inference).

Selected beta (dev) = **1.0**. Geometry baseline ±1-hit ≈ 0.275.

## Locked-test slice-hit: current decoder vs v2 prior decoder
| metric | current | v2 prior |
|---|---|---|
| ±0 slice | 0.134 | 0.162 |
| ±1 slice | 0.432 | 0.487 |
| ±2 slice | 0.652 | 0.714 |
| median abs err | 2.00 | 2.00 |
(n = 1369 level-instances over 296 test studies)

## Verdict (honest)
- The v2 prior decoder **improves** ±1 slice-hit (0.432 → 0.487) on the locked test — a real, no-retrain localization gain. Downstream subarticular grading is robust to leveling noise (v1.1/v1.2), so the main value is route trust/quality, not a large severe-recall change.

Reproduce: `python scripts/run_axial_decode_v2.py`.
