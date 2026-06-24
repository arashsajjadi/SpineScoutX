# Robust Auto-Inference — gap recovery plan (v0.9 line)

> Research-only. Not diagnostic. Not clinically validated. Not for medical
> decision-making. No treatment recommendations. Outputs are non-diagnostic
> **finding severity estimates**, never a diagnosis.

## Why this work exists

Through v0.8 the project measured its single most important honesty result: the
**oracle→auto collapse**. When the grader is fed crops centred on RSNA
ground-truth localizer coordinates (`oracle`) it looks strong; when it is fed
crops centred on the *predicted* disc-level localizer + a geometric-mid slice
(`auto`, the real inference path) severe recall drops sharply, despite a
near-perfect crop-hit@224 (0.998):

| model | severe recall oracle → auto | weighted-logloss oracle → auto |
|---|---|---|
| E0 image-only | 0.828 → 0.644 (−0.184) | 0.326 → 0.554 (+0.228) |
| E2 anatomy-forced | 0.759 → 0.621 (−0.138) | 0.351 → 0.651 (+0.300) |

(canal stenosis, 1955 matched val pairs; `outputs/real/oracle_auto_gap.json`.)

**The oracle numbers are therefore an upper bound, not deployable performance.**
The auto-crop distribution is the real target.

## Root-cause hypothesis (to be tested, not assumed)

The grader was trained *only* on perfectly GT-centred oracle crops with **no
geometric augmentation** (`RsnaCropDataset.__getitem__` loads a cached crop and
applies no jitter/affine/translation), and is evaluated on auto crops whose centre
is offset (localizer residual: median 2.5 px, mean 9.2 px, heavy-tailed at L1/L2)
**and** whose slice is the geometric-mid instance rather than the GT-marked
instance. This is a train/inference covariate shift, not (primarily) a
localizer-accuracy problem.

## Objective

**Gap recovery, not architecture showmanship.** Concretely:

1. **Decompose** the gap into in-plane (xy) vs slice-selection vs combined
   (Phase 1, 2×2 — no retraining).
2. **Model** the localizer error as a training distribution (Phase 2).
3. **Train** graders on the distribution they meet at inference — localizer-aware
   crop/slice jitter, oracle+auto mixing (Phase 3).
4. **Regularize** for consistency across valid crop/slice perturbations (Phase 4).
5. **Learn slice selection** to replace the geometric-mid heuristic (Phase 5).
6. **Severe-first safety mode** on the auto distribution (Phase 6).
7. **Report** every headline number on the **auto** distribution with **bootstrap
   95% CIs** (Phase 7); no oracle-only overclaiming.

## Primary research question

Can localizer-aware robust training recover the severe-recall loss caused by
replacing GT oracle crops with real auto-localized crops — or is the residual gap
dominated by slice selection / irreducible crop brittleness?

## Success definition

A run is successful if it **either**
- (A) recovers a meaningful fraction of the auto severe-recall loss (target:
  auto severe recall +≥0.06 absolute over the oracle-trained-on-auto baseline, with
  non-trivially-overlapping bootstrap CIs and an honest false-positive report);
- **or** (B) proves rigorously that the remaining gap is dominated by slice
  selection or irreducible crop brittleness, with exact evidence and next steps.

## Non-negotiables

- No GT coordinates in any `auto` metric (the 2×2 `hybrid_debug` cells use GT only
  for *diagnosis*, never as a headline auto number, and are labelled as such).
- No fabricated metrics; negative results are reported.
- No data / DICOMs / masks / caches / runs / checkpoints / weights committed.
- Repo stays private; meaningful commits; tag only when gates pass.
- Prefer simple E0-like robust training first; escalate to E2/architecture only if
  the simple recipe proves the idea.

## Provenance vocabulary (every metric row carries these)

- `crop_xy_source` ∈ {`gt`, `auto`}
- `slice_source` ∈ {`gt`, `geometric_mid`, `learned`}
- `coordinate_source` ∈ {`oracle`, `hybrid_debug`, `auto`}
