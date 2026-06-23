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
| 6 | E3 study-level multi-view anatomy-graph reasoner | done |
| 7 | severe-first operating frontier (E0/E2/E3) | done |
| 9 | E3 stream / graph ablation matrix | done |

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

## E3 — study-level multi-view anatomy-graph reasoner (Phase 6)
`MultiViewAnatomyGraphClassifier`: each lumbar level is a graph node fusing up to
three evidence streams — the 2.5D image crop (convnext_tiny token), the SPIDER
anatomy masks (small-CNN token), and the 14 morphology features (MLP token). A
Transformer does **cross-level attention** under a level-availability mask; a shared
head predicts per-level 3-class severity. Trained on canal stenosis, oracle crops,
**same split_seed=1337** as E0/E1/E2 (so val nodes are identical). Best val
(`runs/e3_multiview_canal_real`, early-stop epoch 12):

| metric | E0 image | E2 anatomy-forced | **E3 graph** |
|---|---|---|---|
| weighted-logloss ↓ | **0.326** | 0.351 | 0.330 |
| balanced accuracy ↑ | 0.701 | **0.777** | 0.730 |
| ECE ↓ (calibration) | 0.034 | 0.036 | **0.023** |
| severe AUROC ↑ | 0.978 | **0.979** | 0.972 |

E3 ties E0 on weighted-logloss and is **the best-calibrated** of the three, but does
not beat E2 on balanced accuracy or severe ranking. (The convnext backbone overfits
1.6k studies — train loss → 0.06 while val logloss rises — so E3's best checkpoint is
early.)

## Severe-first operating frontier (Phase 7)
The clinically relevant question is *how much severe recall at what false-alarm cost*.
`severe-frontier` sweeps the severe threshold on the **same 1955 canal val nodes** (87
severe) and reports recall at fixed false-alarm-rate (FAR) budgets. `outputs/real/severe_frontier.json`.

| model | severe AUROC | severe AP | recall@FAR≤5% | recall@FAR≤10% | recall@FAR≤20% | FAR to reach 90% / 95% recall |
|---|---|---|---|---|---|---|
| E0 image | 0.978 | 0.676 | 0.897 | 0.954 | **0.989** | 0.054 / 0.096 |
| **E2 anatomy-forced** | **0.979** | **0.735** | **0.931** | 0.954 | 0.966 | **0.041 / 0.075** |
| E3 graph | 0.972 | 0.681 | 0.828 | **0.966** | 0.966 | 0.087 / 0.100 |

**Honest verdict: E2 (anatomy-forced) wins the severe-first frontier on canal.** It has
the best severe AP, the best recall at the tight 5% false-alarm budget (0.931), and
reaches 90/95% severe recall at the lowest false-alarm cost. E0 is a strong simple
baseline (best at a loose 20% budget). E3 is competitive and best-calibrated, and has
the single highest recall at exactly a 10% budget, but it does **not** justify its extra
complexity on this one condition / one sagittal view. Per the project's honesty policy,
**E2 remains the operating winner and E3 is reported as competitive-but-not-superior,
not spun as a win.**

## E3 stream / graph ablation matrix (Phase 9)
Toggling each evidence stream and the cross-level attention attributes where E3's
behaviour comes from. All variants: canal, oracle crops, same split, retrained.
`outputs/real/e3_ablation.json`.

| variant | streams | graph | val wll ↓ | ECE ↓ | severe AUROC ↑ | severe AP ↑ | recall@FAR≤10% |
|---|---|---|---|---|---|---|---|
| full | image+anatomy+morph | ✓ | 0.330 | **0.023** | 0.972 | 0.681 | **0.966** |
| no_graph | image+anatomy+morph | ✗ | 0.374 | 0.044 | 0.977 | **0.701** | 0.943 |
| **image_only** | image | ✓ | **0.324** | 0.028 | **0.978** | 0.678 | **0.966** |
| no_image | anatomy+morph | ✓ | 0.568 | 0.081 | 0.884 | 0.307 | 0.621 |
| morph_only | morph | ✓ | 0.681 | 0.092 | 0.782 | 0.226 | 0.414 |

**What the ablation honestly shows:**
1. **The image stream carries essentially all the discriminative signal.**
   `image_only` (AUROC 0.978, wll 0.324) is the **best** variant — as good as or
   better than the full model. E3 effectively collapses to a strong per-level image
   model on this task.
2. **Cross-level graph attention does not help and slightly hurts** (image_only 0.978
   ≥ no_graph 0.977 > full 0.972). There is little cross-level signal to exploit when
   each canal level is already well localized.
3. **The anatomy + morphology streams are real but weak on their own**: structured
   features with **no pixels at all** still reach AUROC 0.884 (anatomy+morph) and 0.782
   (morphology alone, vs 0.5 chance) — a genuinely interpretable signal — but far below
   the image stream. Their main contribution to the full model is **calibration**
   (best ECE) rather than discrimination.

So the structured anatomy/morphology graph is valuable as **transparent, citable
evidence** and for calibration, but it does not raise the discriminative ceiling on
the canal condition. This is reported as-is.

## Why E3 did not beat E2 here (and where it should help)
1. **Single condition, single view.** E3's design value is multi-*view* / multi-condition
   joint reasoning; on canal-only sagittal-T2 there is little cross-level signal to
   exploit beyond what a well-localized per-crop model already sees.
2. **Overfitting.** A pretrained convnext backbone on ~1.6k studies overfits quickly;
   the structured (anatomy+morphology) streams mostly help **calibration** (best ECE),
   not peak discrimination.
3. **Where it should pay off** is the staged extension: an **axial localizer** + true
   tri-view evidence bundles for the foraminal/subarticular conditions, where per-level
   and per-view context is genuinely complementary. That needs honest axial anatomy
   (currently *approximate*) and is left as explicit future work rather than faked.

## Honest scope note (multi-view / Phase 4)
The localizer and anatomy masks are honest only for the **spinal canal on sagittal
T2** (SPIDER provides canal/disc/vertebra ground truth; foraminal/subarticular regions
are flagged *approximate*). Genuine tri-view evidence for foraminal/subarticular
findings needs an **axial localizer** and would carry approximate regions — so E3 is
built and evaluated first on the canal condition where the anatomy is real, and the
multi-view extension to axial conditions is staged explicitly rather than faked.
