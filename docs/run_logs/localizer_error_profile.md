# Localizer error profile (Phase 2) — the jitter distribution to train against

> Research-only. Not diagnostic. Measured on the canal val set (the localizer's honest
> held-out error), from the 2×2 cells (`c1` GT xy, `c2` localizer xy at GT slice).

The in-plane residual is `auto − gt` (px); it is the distribution the grader must be
robust to. `outputs/real/localizer_error_profile.json`.

## In-plane residual (auto − GT), per level

| level | n | σ_x (px) | σ_y (px) | median ‖·‖ | mean ‖·‖ | p90 ‖·‖ |
|---|---|---|---|---|---|---|
| l1_l2 | 382 | 8.2 | **33.8** | 5.7 | 21.0 | 62.9 |
| l2_l3 | 388 | 11.9 | 31.4 | 5.5 | 20.1 | 59.5 |
| l3_l4 | 395 | 15.4 | 32.0 | 4.3 | 16.7 | 57.4 |
| l4_l5 | 395 | 15.9 | 27.2 | 3.5 | 11.4 | 39.4 |
| l5_s1 | 395 | 17.4 | 27.2 | 3.3 | 8.8 | 11.3 |
| **pooled** | 1955 | — | — | **4.3** | **15.6** | **54.1** (p99 **102.5**) |

## Slice offset (geometric-mid − GT instance)

mean 0.12, std 0.67, **62% are 0**, |offset| p90 = 1. The geometric-mid slice is at or
within ±1 of the GT-marked instance for the vast majority of nodes.

## Reading

1. **The error is anisotropic and dominated by the superior–inferior (y) axis**
   (σ_y ≈ 27–34 px vs σ_x ≈ 8–17 px). The localizer occasionally snaps to an adjacent
   disc level vertically — that is the heavy tail (median 4.3 px but mean 15.6 px,
   p99 103 px), and it is worst at the upper levels (L1/L2).
2. **Slice selection is essentially a non-issue** (62% exact, p90 ±1 instance) —
   independent confirmation of the Phase 1 result that slice-source does not drive the
   severe-recall gap.

## Use

These per-level σ seed the `level_aware` `CropJitterConfig` (with a heavy-tail mixture)
and the `empirical` sampler draws the raw residuals directly. In the Phase 3
experiments, training on the *actual* auto-localized crops (`r_auto_train`) — which
carry this exact error structure — beat synthetic jitter, confirming the profile is the
right target but that matching it exactly is better than approximating it.

> Note: this profile is the localizer's **val** error. It is used only to shape training
> augmentation magnitude (a known device property), never as a label — no val severity
> target informs training. The leakage-free `r_auto_train` variant (localizer errors
> generated organically on the train split) agrees with and exceeds the jitter variants.
