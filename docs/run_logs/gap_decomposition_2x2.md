# 2×2 oracle→auto gap decomposition (Phase 1)

> Research-only. Not diagnostic. Not clinically validated. Measurement instrumentation
> only — **no retraining**. The two `hybrid_debug` cells use GT information solely to
> attribute the gap and are never reported as a headline auto number.

## Question

The oracle→auto severe-recall collapse (E0 canal 0.828 → 0.644) conflates two
changes at once: the in-plane crop **centre** (GT coord → localizer prediction) and
the **slice** (GT-marked instance → geometric-mid instance). Which one causes the
collapse?

## Design

A controlled 2×2, **holding the series fixed to the GT sagittal-T2 series** so the
only varying factors are xy-source and slice-source. Nodes, GT series/instance/x,y
are taken from the established cached oracle manifest, so **C1 is exactly the
project's oracle crop** (pixel-identical) and C4 is the within-series analogue of the
real auto path. Same E0 image-only model on all cells; identical 1955 canal val
nodes (87 severe); cluster-bootstrap 95% CIs (resampled by `study_id`, 2000 reps).

| cell | crop_xy | slice | coordinate_source | isolates |
|---|---|---|---|---|
| C1 | gt | gt | oracle | upper bound |
| C2 | auto | gt | hybrid_debug | **in-plane error** |
| C3 | gt | geometric_mid | hybrid_debug | **slice error** |
| C4 | auto | geometric_mid | auto | combined |

## Result (E0, canal val, n=1955, severe=87)

| cell | severe recall [95% CI] | weighted log loss [95% CI] |
|---|---|---|
| C1 gt-xy / gt-slice | **0.828** [0.742, 0.904] | 0.326 [0.283, 0.371] |
| C2 auto-xy / gt-slice | **0.644** [0.528, 0.754] | 0.512 [0.420, 0.613] |
| C3 gt-xy / mid-slice | **0.839** [0.750, 0.918] | 0.372 [0.324, 0.422] |
| C4 auto-xy / mid-slice | **0.644** [0.536, 0.746] | 0.554 [0.461, 0.658] |

**Validation:** C1 reproduces the established oracle exactly (0.828 / wll 0.326) and
C4 reproduces the established real-auto exactly (0.644 / wll 0.554) — even though C4
holds the series fixed while the real auto path re-selects the series. So the
**series-selection effect is negligible**; the within-series auto IS the real auto.

### Attribution (paired cluster-bootstrap deltas vs oracle C1)

| effect | Δ severe recall [95% CI] | decisive? | Δ weighted log loss [95% CI] | decisive? |
|---|---|---|---|---|
| in-plane only (C2−C1) | **−0.184** [−0.291, −0.089] | **yes** | +0.187 [+0.115, +0.267] | yes |
| slice only (C3−C1) | +0.011 [−0.039, +0.065] | **no** | +0.046 [+0.026, +0.066] | yes (small) |
| combined (C4−C1) | −0.184 [−0.277, −0.095] | yes | +0.228 [+0.155, +0.308] | yes |
| interaction | −0.011 | — | — | — |

McNemar on severe hits, C1 vs C4: **18** caught→missed vs **2** missed→caught,
p = 4.0e-4.

### Per-level severe recall

| level | n | severe | C1 oracle | C2 in-plane | C3 slice | C4 combined |
|---|---|---|---|---|---|---|
| l1_l2 | 382 | 8 | 0.875 | **0.250** | 0.750 | 0.250 |
| l2_l3 | 388 | 11 | 0.636 | 0.455 | 0.636 | 0.364 |
| l3_l4 | 395 | 23 | 0.739 | 0.652 | 0.826 | 0.652 |
| l4_l5 | 395 | 40 | 0.950 | 0.825 | 0.950 | 0.825 |
| l5_s1 | 395 | 5 | 0.600 | 0.200 | 0.600 | 0.400 |

## Conclusion

**The oracle→auto severe-recall collapse is entirely in-plane (crop-centre) driven.**

1. Replacing GT xy with the localizer prediction at the *same slice* drops severe
   recall by the full **−0.184** (decisive). Replacing the GT slice with the
   geometric-mid slice at the *same xy* changes severe recall by **+0.011**
   (not decisive — geometric-mid is, if anything, marginally better).
2. The damage tracks the localizer's in-plane error tail: worst at the upper levels
   (L1/L2, L5/S1 — few severe but large mean px error) and mildest at L4/L5 (most
   severe, smallest localizer error). The localizer is good on the median
   (~2.5 px) but heavy-tailed (mean ~17 px in the real auto path, p99 > 100 px), and
   the grader — trained only on perfectly centred crops — is brittle to those offsets.
3. Slice has a small, decisive effect on calibration/log-loss (+0.046 wll) but not on
   the safety-critical severe recall.

## Strategic implication (drives Phases 3–6)

- **Pursue:** localizer-aware **crop-centre jitter** + **consistency regularization**
  (train the grader on the offsets it meets at inference); optionally reduce the
  localizer's in-plane *tail* (esp. upper levels).
- **Deprioritize:** a learned **slice selector** (Phase 5) — the evidence shows slice
  selection is not the bottleneck for severe recall. Recorded as not-warranted by
  data rather than silently dropped.

Artifacts: `outputs/real/gap_decomposition_2x2.json`,
`outputs/real/figures/gap_decomposition_2x2.png` (gitignored, regenerable via
`python scripts/run_gap_decomposition.py`).
