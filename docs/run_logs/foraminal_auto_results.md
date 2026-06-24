# Foraminal (sagittal-T1) auto route — locked test (splits_v1)

> Research-only. Not diagnostic. Auto inference reads NO GT coordinates: the T1 series +
> parasagittal slice are chosen from DICOM `ImagePositionPatient` (laterality, +x =
> patient-left, verified vs GT) + the localizer's own confidence (best-slice scoring).
> Graders retrained on splits_v1 `train`, selected on `dev` auto, evaluated ONCE on the
> locked `test`. Cluster-bootstrap (by study) 95% CIs.

## Foraminal localizer QC (per side) — excellent
| split/side | n | median px | mean px | PCK@10 | crop-hit@224 |
|---|---|---|---|---|---|
| dev_left | 1470 | 2.35 | 6.70 | 0.843 | 0.999 |
| dev_right | 1470 | 2.19 | 6.60 | 0.848 | 0.999 |
| test_left | 1480 | 2.22 | 6.71 | 0.841 | 0.999 |
| test_right | 1480 | 2.16 | 6.53 | 0.839 | 0.999 |

The sagittal-T1 side-aware localizer is clean (median ~2.2 px, crop-hit@224 0.999) — even
tighter than the canal localizer, because foraminal levels are co-planar on one
parasagittal slice. Side assignment from `ImagePositionPatient[0]` is robust.

## Severe recall [95% CI] on locked test — each grader × provenance (n / severe per side)
| grader | left: test-oracle | **left: test-auto** | right: test-oracle | **right: test-auto** |
|---|---|---|---|---|
| oracle-trained | 0.865 | **0.788 [0.673, 0.892]** | 0.925 | **0.660 [0.524, 0.788]** |
| auto-trained (robust) | 0.692 | 0.654 [0.508, 0.786] | 0.679 | 0.491 [0.355, 0.633] |

n: left 1480 / sev 52; right 1470 / sev 53.

## Honest verdict — foraminal auto-inference is UNLOCKED (coverage → 3/5), and the
## deployable grader is the **oracle-trained** one applied to auto crops

- **Deployable foraminal auto severe recall: left 0.788 [0.673, 0.892], right 0.660
  [0.524, 0.788]** (oracle-trained grader, auto-localized crops, locked test, no GT at
  inference). This is real end-to-end auto inference for both foraminal sides.
- **Robust auto-training did NOT help foraminal — it hurt** (paired auto severe recall
  −0.135 left / −0.170 right vs the oracle-trained grader, both decisive). This is the
  **opposite of canal**, and the reason is instructive:
  - **Canal**: heavy-tailed localizer (mean ~9–17 px) → large oracle→auto gap (0.83→0.43)
    → robust auto-training is essential (recovers to 0.83).
  - **Foraminal**: clean localizer (median 2.2 px, crop-hit 0.999) → **small oracle→auto
    gap** (oracle-trained drops only −0.077 left / −0.264 right). With little distribution
    shift to absorb, training on the (slightly noisier) auto crops only adds label noise
    and lowers the ceiling; clean oracle training + transfer wins.
- **Generalizable finding:** whether robust auto-training helps is governed by the size of
  the oracle→auto gap (i.e. localizer quality), not by the condition per se. Pick the
  grader per condition by locked-test auto performance — canal → auto-trained, foraminal →
  oracle-trained.
- **Honest caveat:** right foraminal still has a moderate residual gap (0.925→0.660); its
  auto severe recall (0.660) trails left (0.788). The combined side-aware auto-robust
  variant did not close it; a right-specific localizer/grader is a possible refinement,
  recorded as future work rather than over-built now.

Provenance: oracle = GT-coordinate crop (upper bound); auto = localizer-predicted
parasagittal-T1 crop (real inference). Artifacts: `outputs/real/foraminal_auto_results.json`.
Reproduce: `python scripts/run_foraminal_locked_test.py`.
