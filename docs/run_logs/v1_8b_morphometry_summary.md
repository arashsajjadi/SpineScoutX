# v1.8b — morphometry feature signal (Phases 8-9)

> Research-only · not diagnostic. Segmentation-derived foraminal morphometry (13 features:
> opening area/extent/compactness, intensity contrast, segmentation confidence, min-opening) from
> the SAM2.1 masks. Reproduce: `train_morphometry_only_models_v1_8b.py`.

## Morphometry-only severity signal (dev, train-fit)

| route | logistic dev AUROC | GBM dev AUROC | GBM severe recall@FAR≤10 | top feature |
|---|---|---|---|---|
| right-foraminal | 0.636 | **0.687** | 0.208 | `m_contrast` |
| left-foraminal  | (≈0.6) | (≈0.65) | — | `m_contrast` (\|AUROC−0.5\|=0.22) |

**Morphometry contains real severity signal** (dev AUROC 0.687 for right-foraminal, well above
chance) — but it comes from **intensity contrast** inside vs around the segmented opening, **not**
from the opening area/geometry (area is flat severe-vs-non-severe). It is **weaker than the deployed
image grader** (whose right-foraminal severe recall is 0.660 at a much higher AUROC).

## Interpretation

The signal is genuine but **derived from the same pixels the image grader already sees** — a
coarse, redundant view of the image. Whether it adds anything is tested by fusion (Phase 10) and
triage (Phase 11); both say no (`v1_8b_final_morphometry_accuracy_results.md`).
