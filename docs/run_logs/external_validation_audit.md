# External-validation feasibility + internal domain-shift stress test

> Research-only · not diagnostic · not clinically validated. Privacy-safe acquisition
> parameters only; no scanner identifiers (RSNA strips them). Deployed locked-test
> predictions reused (no new inference, no GT coordinates).

## External validation: NOT performed (feasibility audit)
RSNA LumbarDISC de-identifies Manufacturer / model / field strength (absent in headers); SPIDER has segmentation masks, not the 5 graded findings. No independent labeled lumbar-MRI source with the same five severity labels is legally available, so external/prospective validation is not performed. Internal acquisition-shift stress test is reported instead.

| metadata | coverage |
|---|---|
| slice thickness present | 100% |
| pixel spacing present | 100% |
| scanner / vendor / field strength | 0% (stripped) |

## Internal domain-shift (pooled severe recall, all 5 findings; overall 0.748 [0.703, 0.791], n_severe=433)

| axis | stratum | severe recall [95% CI] | n_severe | n |
|---|---|---|---|---|
| slice_thickness | thin (<=3.5mm) | 0.832 [0.744, 0.915] | 95 | 1565 |
| slice_thickness | standard (4.0mm) | 0.728 [0.676, 0.777] | 309 | 5200 |
| slice_thickness | thick (>4.0mm) | 0.690 [0.474, 0.844] | 29 | 533 |
| pixel_spacing | fine (<=median 0.56mm) | 0.747 [0.687, 0.804] | 261 | 3688 |
| pixel_spacing | coarse (>median 0.56mm) | 0.750 [0.682, 0.816] | 172 | 3610 |
| matrix_rows | small (<=384) | 0.740 [0.672, 0.806] | 204 | 3213 |
| matrix_rows | large (>384) | 0.755 [0.692, 0.816] | 229 | 4085 |
| level | l1_l2 | 0.429 [0.143, 0.857] | 7 | 1418 |
| level | l2_l3 | 0.750 [0.522, 0.919] | 28 | 1448 |
| level | l3_l4 | 0.738 [0.630, 0.837] | 80 | 1478 |
| level | l4_l5 | 0.868 [0.810, 0.915] | 197 | 1478 |
| level | l5_s1 | 0.579 [0.488, 0.667] | 121 | 1476 |

## Interpretation
- Severe recall is reported across acquisition protocol (slice thickness, in-plane
  resolution, matrix size) and anatomy (level). Strata whose CI excludes the overall
  point estimate indicate a real acquisition-shift sensitivity; wide CIs (small
  n_severe) are reported as such and not over-interpreted.
- This is **internal** robustness only. It does **not** establish generalization to new
  institutions, scanners, populations, or prospective use — that needs external and
  prospective studies, which have **not** been done (no legal labeled source available).

Reproduce: `python scripts/run_domain_shift_audit.py` (after `run_evidence_stability.py`).
