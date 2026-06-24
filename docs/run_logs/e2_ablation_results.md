# E2 anatomy-forced ablation (REAL RSNA val) — does the model use anatomy?

> Research-only, not diagnostic. The decisive test of the v0.5 design: perturb the
> anatomy prior at inference and measure the impact. Run artifacts gitignored;
> reproduce: `spinescoutx ablate --config configs/ablation_e2_forced.yaml`.

| anatomy mode | weighted log loss | macro F1 | severe recall | severe FNR | mean AEC | Δwll vs correct |
|---|---|---|---|---|---|---|
| correct | 0.4730 | 0.716 | 0.735 | 0.265 | 0.107 | — |
| shuffled | 0.4741 | 0.721 | 0.737 | 0.263 | 0.106 | +0.001 |
| zero | 0.4928 | 0.700 | 0.800 | 0.200 | 0.102 | **+0.020** |
| noise | 0.4873 | 0.712 | 0.733 | 0.267 | 0.104 | **+0.014** |
| target_region_only | 0.4654 | 0.713 | 0.754 | 0.246 | 0.107 | **−0.008** |
| wrong_region_only | 0.4887 | 0.707 | 0.786 | 0.214 | 0.102 | **+0.016** |

## Interpretation (honest)
1. **Anatomy is genuinely used** (unlike v0.4): zero/noise/wrong-region degrade
   weighted log loss by 0.014–0.020 vs **< 0.001** for the v0.4 concat model — a
   ~**20×** increase in anatomy sensitivity from the region-pooling forcing.
2. **The target region is the useful signal:** `target_region_only` (keep only the
   condition's dominant region) is the *best* mode (Δwll −0.008).
3. **Used as a regional gate, not localization:** `shuffled ≈ correct`, so the model
   keys on *plausible anatomy being present* rather than the *correct sample's*
   precise mask.
4. **Evidence localization (AEC) did not improve** (~0.10, flat): forcing feature
   reliance ≠ forcing saliency. Future work: an explicit AEC/region-saliency loss
   (Phase 7 hook) and the multi-view model (Phase 6, designed; future work).

## Verdict
v0.5 achieved its scientific aim — anatomy became operationally used and exposes a
higher severe-recall frontier (sevR 0.855 reachable vs E0's 0.751). It is not a free
aggregate-accuracy win and does not yet improve AEC. No claim beyond what is measured.
