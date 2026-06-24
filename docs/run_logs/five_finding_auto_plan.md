# Five-finding auto SpineScoutX — plan + decision audit

> Research-only. Not diagnostic. Not clinically validated. Not for medical decision-making.
> Goal: move real **auto-localized** inference from 1/5 findings (canal) toward 5/5,
> improving real locked-test severe-safety, not task-completion optics.

## Current state (verified)
- **Canal (1/5): SOLVED on locked test.** sagittal-T2 disc localizer + robust auto-training;
  auto severe recall 0.830 [0.725, 0.929] vs oracle-trained control 0.434 (paired +0.396,
  McNemar p<1e-6). `canal_locked_test.md`.
- **Foraminal L/R, Subarticular L/R (4/5): oracle baselines only** — no auto route yet.

## Decision audit (evidence-grounded)
GT-coordinate views (measured, 100%): canal=sagittal-T2; **foraminal=sagittal-T1**;
**subarticular=axial-T2**.

- **Foraminal is the highest-confidence next win (Priority 1).** Probes show: (a) laterality
  is robust from DICOM `ImagePositionPatient[0]` (+x = patient LEFT, verified vs GT:
  GT-left mean x +18.2 vs GT-right −15.1); instance order alone is unreliable (left<right
  only 36%). (b) The 5 foraminal levels of a side are **co-planar** on one parasagittal
  slice (per-side instance std 0.55, p90 1.0) → the canal-style single-slice 5-keypoint
  heatmap localizer transfers directly. → **Build a sagittal-T1 side-aware foraminal route.**
- **Axial subarticular feasibility gate: PASSES (Priority 2, stretch).** All 1975 studies
  have axial-T2 (median 1 series, ~30 slices, ~203 mm z-extent spanning all levels),
  `ImagePositionPatient` present 100% → **sagittal-disc-z → axial-slice-z level matching is
  possible**. Levels lie on different axial slices (instance std 8.1). Higher complexity
  (cross-series geometry); attempt after foraminal; ship the auto result or an
  evidence-backed blocker.

## Chosen strategy (and skips)
1. **Foraminal first** (1/5 → 3/5): T1 series selector → metadata laterality → side-aware
   foraminal localizer (reuse `DiscLevelLocalizer`) → best-slice selection → auto crops →
   robust auto-trained graders → locked-test CI + 2×2 gap decomposition. **Side-aware
   specialists** (left/right) — likely beat a shared model given mirror anatomy.
2. **Axial subarticular** (stretch → 5/5): z-based level matching + side-aware lateral-recess
   crops + robust grader on locked test; else a precise blocker.
3. **Safety Mode v3** across every unlocked condition (frontier + review reasons).
4. **Multi-condition study report** labelling auto vs oracle-only/blocked findings.
5. Docs + honest tags. **v1.0 only if 5/5 real auto locked-test results exist.**

### Skipped / not reused (with reason)
- **Cost-sensitive training**: prior honest negative (brittle on imbalance); not reused —
  severe-safety via robust auto-training + threshold frontier + review layer.
- **Big multi-view graph / transformer**: not needed; per-condition specialists + simple
  best-slice/top-k pooling preferred unless evidence says otherwise.
- **Morphology as classifier input**: kept as report/safety signal only (canal-specific masks).

## Provenance (every metric)
`oracle_crop` (GT coords; upper bound), `auto_crop` (predicted; real inference),
`hybrid_debug` (GT used only for gap decomposition, never a headline auto number).
Every headline carries split (dev/test), condition, side, n, n_severe, 95% CI.
