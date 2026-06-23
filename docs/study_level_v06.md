# SpineScoutX v0.6/0.7 — study-level multi-view anatomy-graph reasoner

> Research-only. Not diagnostic. Not clinically validated. Not for medical
> decision-making. This document tracks the v0.6/0.7 build on top of the v0.5
> anatomy-forced crop classifier.

The driving question: **can study-level reasoning plus anatomy/morphology graphs
improve severe-case reliability, per-condition performance, and evidence usefulness
beyond the E0/E1/E2 crop systems — under *honest* (localizer-driven) inference?**

## What shipped

| phase | component | status |
|---|---|---|
| 1 | coordinate-dependency audit + crop provenance (`oracle` vs `auto`) | done — `coordinate_dependency_audit.md` |
| 2 | disc-level localizer + auto-crop path + **measured oracle→auto gap** | done — tag `v0.6.0-study-level-localizer` |
| 3 | study-level series/view registry + view selector | done |
| 5 | morphology feature engine (structured geometry from anatomy masks) | done |

The headline **honesty result** lives in `coordinate_dependency_audit.md`: removing GT
coordinates (oracle→auto) costs E0 ≈ +0.23 weighted-logloss and **−18 pp severe
recall**, despite a near-perfect crop-hit. The strong v0.4–0.5 numbers are an oracle
upper bound; a deployable system is materially weaker on the severe class.

## Study registry (Phase 3)
`build-study-index` records, per study, view availability (sagittal T1 / sagittal
T2-STIR / axial T2), the best (most-populated) series per view, slice counts, a 3-bit
view mask, and a usability flag. Real RSNA: **1975 studies, all usable**; 1972 have
all three views (mask `111`), 2 miss sagittal-T1, 1 misses sagittal-T2, 0 miss axial.
This is the substrate for multi-view evidence bundles and the per-view attention mask.

## Morphology feature engine (Phase 5)
`features/morphology.py` derives **14 deterministic structured features** per crop
from the SPIDER-trained anatomy masks `(disc, spinal_canal, vertebra)` — areas,
canal/disc and canal/vertebra ratios, min/mean canal width, width irregularity (CV),
AP extent, compactness, centroid, left/right asymmetry. No model, no randomness, no GT
coordinates. These are anatomy geometry, **not pathology measurements**.

**Does it carry a real severity signal?** On 9,753 oracle canal crops, Spearman
correlation of feature vs severity (normal<moderate<severe), strongest first:

| feature | ρ vs severity | direction |
|---|---|---|
| canal_vert_ratio | **−0.323** | smaller canal-to-vertebra when severe |
| canal_disc_ratio | **−0.223** | canal/disc ratio collapses when severe |
| canal_width_cv | +0.212 | canal calibre more irregular when severe |
| mean_canal_width | −0.153 | narrower when severe |
| min_canal_width | −0.125 | narrowest point shrinks when severe |
| canal_area | −0.122 | smaller canal when severe |
| canal_compactness | −0.107 | less compact when severe |

Mean-by-severity (clinically expected monotone trends):

| feature | normal | moderate | severe |
|---|---|---|---|
| canal_disc_ratio | 5.92 | 5.15 | **2.34** |
| min_canal_width (frac) | 0.043 | 0.031 | **0.023** |
| canal_area (frac) | 0.084 | 0.073 | **0.066** |

The signal is **modest but real and in the clinically expected direction** (a stenotic
canal is narrower / smaller / lower-ratio). It is an interpretable, structured evidence
stream the multi-view graph reasoner (E3) can fuse with image tokens — and a
transparent feature an evidence report can cite.

## Honest scope note (multi-view / Phase 4)
The localizer and anatomy masks are honest only for the **spinal canal on sagittal
T2** (SPIDER provides canal/disc/vertebra ground truth; foraminal/subarticular regions
are flagged *approximate*). Genuine tri-view evidence for foraminal/subarticular
findings needs an **axial localizer** and would carry approximate regions — so E3 is
built and evaluated first on the canal condition where the anatomy is real, and the
multi-view extension to axial conditions is staged explicitly rather than faked.
