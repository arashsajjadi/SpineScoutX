# Results — SpineScoutX (real data)

> Research-only, not diagnostic, not clinically validated, not for medical
> decision-making. All numbers below are **real**, measured on held-out data; run
> artifacts are gitignored and regenerable. Nothing is fabricated, nothing
> cherry-picked.

## Research question
Can explicit anatomical priors from a real SPIDER-trained segmenter improve
disc-level lumbar degenerative finding grading on RSNA/LumbarDISC?

**Short answer (this implementation): no meaningful improvement, and the
anatomy-guided model largely ignores the priors — shown by the ablation.** Details
below; we report this honestly rather than overclaim the tiny aggregate edge.

## Data
RSNA: 1,974 studies → 48,692 localizer crops (2.5D, 224²), patient/study-level
split (38,947 train / 9,745 val), **0 leakage**, severe ≈ 6.3%. SPIDER: 218
patients / 10,338 slices (official split).

## E4 — SPIDER anatomy segmentation (real)
mean Dice **0.884** · canal **0.902** · vertebra **0.903** · disc 0.846 (official
SPIDER val). Anatomy masks only (not pathology). Used to generate RSNA anatomy
priors (48,692; vertebra present 99.8%, disc 85%, canal 47%).

## E0 vs E1 — disc-level grading (real RSNA val, 9,745 crops)
| metric | E0 image-only | E1 anatomy-guided | better |
|---|---|---|---|
| weighted log loss ↓ | 0.4621 | **0.4579** | E1 (−0.0042) |
| macro F1 ↑ | 0.706 | **0.717** | E1 |
| balanced accuracy ↑ | **0.763** | 0.752 | E0 |
| **severe recall** ↑ | **0.751** | 0.711 | E0 |
| severe FNR ↓ | **0.249** | 0.289 | E0 |
| severe AUROC ↑ | 0.971 | 0.972 | ~tie |
| ECE ↓ | **0.027** | 0.034 | E0 |

**Mixed and marginal.** E1 edges E0 on aggregate log loss / macro-F1; E0 is better
on severe recall and calibration. No clear win for anatomy priors.

## Counterfactual anatomy ablation (E1 model, real val) — the decisive test
| anatomy mode | weighted log loss | severe recall | mean AEC |
|---|---|---|---|
| correct | 0.4579 | 0.711 | 0.099 |
| shuffled (other sample) | 0.4576 | 0.711 | 0.100 |
| zero | 0.4584 | 0.723 | 0.101 |
| noise | 0.4580 | 0.711 | 0.100 |

Δ vs correct: |Δ weighted log loss| < 0.001 for every perturbation; severe recall
unchanged (zero even +0.012). **Zeroing, shuffling, or noising the anatomy prior
barely changes anything ⇒ E1 does not meaningfully use the anatomy branch.** By the
project's own interpretation rules (`zero ≈ correct ⇒ branch ignored`,
`correct ≈ shuffled ⇒ anatomy not semantically used`), the small E1>E0 edge is
**not attributable to anatomy** — more plausibly extra head capacity / training
variance.

## Evidence consistency (AEC)
Mean AEC ≈ **0.10** and **flat across all anatomy perturbations** — consistent with
the model ignoring anatomy. AEC is also region-size sensitive (the canal/foraminal
target regions are a small fraction of the crop), so a low absolute AEC is partly
expected; the *flatness across ablation modes* is the informative signal. Foraminal
/ subarticular AEC regions are **approximate** (SPIDER lacks those labels).

## Calibration
Both models are well-calibrated out of the box (E0 ECE 0.027, E1 ECE 0.034);
temperature scaling (fit on val — a research-demo simplification) leaves ECE
essentially unchanged (E1 0.034→0.031, T≈1.13; E0 T≈0.98). Uncertainty flags
(`high`/`moderate`/`review_required`) are attached to every finding.

## Why v0.5 — "anatomy-forced" (motivation)

The v0.4 ablation is the crux: concat-fusion (`[image crop ⊕ anatomy channels] → classifier`)
let the model **ignore** anatomy (zero ≈ shuffle ≈ correct). Concatenated mask channels are an
*optional* input the network can down-weight to zero. v0.5 changes the **structure** so anatomy
is not bypassable: the anatomy masks define **regions**, and features are **region-pooled** from
those masks (canal/disc/vertebra), with global-feature dropout forcing the head to rely on the
region features. If the mask is zeroed or shuffled, the region features change by construction, so
the prediction must change too — making the ablation a genuine test of anatomy usage.

## v0.5 — E2 anatomy-forced (REAL, real RSNA val)

E2 = `AnatomyForcedRegionClassifier`: image→feature map; anatomy masks define
regions; masked region pooling (disc/canal/vertebra); condition→target-region
attention; global-feature dropout (0.5) forces region reliance; severe-aware model
selection (`wll + 0.08·severe_fnr`).

### Grading metrics (selected checkpoint)
| metric | E0 image-only | E1 concat | **E2 anatomy-forced** |
|---|---|---|---|
| weighted log loss ↓ | 0.4621 | 0.4579 | 0.4730 |
| macro F1 ↑ | 0.706 | 0.717 | 0.716 |
| severe recall ↑ | 0.751 | 0.711 | 0.735 |
| severe AUROC ↑ | 0.971 | 0.972 | **0.973** |
| ECE ↓ | 0.027 | 0.034 | **0.0275** |

At the wll/severe-aware-optimal checkpoint, E2 is roughly comparable to E0/E1.
**But the severe-recall frontier differs:** during training E2 reached **severe
recall 0.855** (epoch 2) at weighted log loss 0.516 — a **+0.10 severe-recall**
operating point above E0's best (0.751). The severe-aware λ=0.08 was too small to
*select* that point; a larger λ selects it. So E2 can trade ~0.05 log loss for ~+0.10
severe recall — a capability E0/E1 did not show. We report both, hiding no tradeoff.

### Decisive ablation — does E2 use anatomy? (YES, unlike v0.4)
| anatomy mode | weighted log loss | severe recall | Δwll vs correct |
|---|---|---|---|
| correct | 0.4730 | 0.735 | — |
| shuffled | 0.4741 | 0.737 | +0.001 |
| zero | 0.4928 | 0.800 | **+0.020** |
| noise | 0.4873 | 0.733 | **+0.014** |
| target_region_only | 0.4654 | 0.754 | **−0.008** |
| wrong_region_only | 0.4887 | 0.786 | **+0.016** |

- **E2 genuinely uses anatomy.** Zeroing / noising / wrong-region degrade weighted
  log loss by **0.014–0.020** — versus **< 0.001** for the v0.4 concat model (E1).
  The structural region-pooling forcing increased anatomy sensitivity ~**20×**.
- **`target_region_only` is the best mode** (Δwll −0.008): keeping only the
  condition's dominant region (canal for canal-stenosis, disc for foraminal) and
  zeroing the rest *improves* log loss — direct evidence the target region carries
  the useful signal.
- **Honest nuance:** `shuffled ≈ correct` (Δwll +0.001). The model is sensitive to
  whether *plausible* anatomy is present (zero/noise hurt) but not to whether it is
  the *correct sample's* anatomy — so E2 uses anatomy as a **regional
  presence/plausibility gate**, not precise per-sample localization.
- **AEC ≈ 0.10**, still low and roughly flat — Grad-CAM evidence does not strongly
  concentrate in the (small) target regions (AEC is region-size limited). Forcing
  *feature* reliance did not by itself force *saliency* localization (future work:
  an explicit AEC/region-saliency loss).

### v0.4 → v0.5 verdict
The anatomy-forced design **succeeded at its scientific goal**: it converted an
"anatomy is ignored" model into one that **measurably uses anatomy** (20× larger
ablation deltas) with a **higher severe-recall frontier**. It is not a free accuracy
win at the selected checkpoint (E2 ≈ E0 on aggregate log loss) and does not yet
improve evidence localization (AEC). Reported exactly as measured.

## Why v0.6 — study-level multi-view reasoning (motivation)

v0.5 proved anatomy *sensitivity* is achievable (region pooling) but did not produce
a decisive aggregate improvement, and the pipeline is still **crop-centric and
dependent on RSNA ground-truth localizer coordinates** — i.e. it is not yet a
study-level inference system. v0.6 attacks this directly:

1. **Coordinate provenance** — every metric is tagged `oracle_crop` (uses GT
   coordinates; research upper bound) vs `auto_crop` (predicted localization; real
   inference). No GT coordinates are read at auto-mode inference.
2. **Automatic disc-level localization** — a heatmap localizer replaces GT
   coordinates, so we can measure the honest oracle→auto grading gap.
3. **Study-level multi-view** — a series/view registry + sagittal+axial evidence
   bundles + anatomy morphology features feed a multi-view model (E3), instead of one
   crop per GT localizer.

The goal is a real study-level reasoner; results are reported with provenance and
without spin (if auto-localization bottlenecks grading, we say so).

## v0.6 / v0.7 — study-level results (REAL; full writeup in `study_level_v06.md`)

**Headline honesty result — the oracle→auto gap.** A disc-level heatmap localizer
(median 2.5 px, crop-hit@224 0.998) replaces GT coordinates. On 1955 matched canal
val (study, level) pairs, removing GT coordinates costs:

| model | weighted-logloss | severe recall |
|---|---|---|
| E0 | 0.326 → 0.554 (**+0.228**) | 0.828 → 0.644 (**−0.184**) |
| E2 | 0.351 → 0.651 (**+0.300**) | 0.759 → 0.621 (**−0.138**) |

So the strong v0.4–0.5 numbers are an **oracle upper bound**; a deployable
localizer-driven system is **14–18 pp weaker on severe recall**. (Full provenance in
`coordinate_dependency_audit.md`.)

**E3 study-level multi-view anatomy-graph reasoner** (per-level image + anatomy +
morphology tokens, cross-level attention; canal, oracle crops, same split). Severe-
first frontier on 1955 shared canal val nodes:

| model | severe AUROC | severe AP | recall@FAR≤5% | recall@FAR≤10% | ECE |
|---|---|---|---|---|---|
| E0 image | 0.978 | 0.676 | 0.897 | 0.954 | 0.034 |
| **E2 anatomy-forced** | **0.979** | **0.735** | **0.931** | 0.954 | 0.036 |
| E3 graph | 0.972 | 0.681 | 0.828 | **0.966** | **0.023** |

**Honest verdict: E2 wins the severe-first frontier on canal.** E3 is competitive and
best-calibrated but does not beat E2. The ablation shows the **image stream carries
nearly all discriminative signal** (image-only AUROC 0.978 = best), **cross-level
attention does not help** (full 0.972 < no-graph 0.977 < image-only 0.978), and the
anatomy/morphology streams are a real-but-weak interpretable signal (anatomy+morph
with no pixels → AUROC 0.884; morphology-only → 0.782) whose main contribution is
calibration. The multi-view graph's genuine promise (axial views, foraminal/
subarticular) needs an axial localizer and is staged as explicit future work, not
faked. External-literature context: `external_validation.md` (E3 canal AUROC 0.972 ≈
M-SCAN's published 0.971; localize-then-classify and severe-recall@fixed-FAR are both
field-standard; the SPIDER→RSNA prior and anatomy-mask graph appear novel).

## Honest interpretation
- **Did anatomy priors help classification?** Not meaningfully — a <1% log-loss /
  macro-F1 edge that the ablation shows is **not** due to anatomy; severe recall is
  worse for E1.
- **Did they improve evidence consistency (AEC)?** No — AEC is low and flat.
- **Did calibration help?** Both are already well-calibrated; temperature scaling
  adds little.
- **What worked:** a strong real E0 baseline (severe AUROC 0.971), real E4
  segmentation (Dice 0.884), and a **rigorous ablation that prevents a false
  "anatomy helps" claim**.
- **Why the negative result is useful:** it demonstrates that naive
  channel-concatenation of anatomy priors is insufficient; future work should force
  reliance on anatomy (e.g., anatomy-conditioned attention, region-pooled features,
  auxiliary AEC loss) rather than concluding anatomy is useless.

## Failure cases (shown, not hidden)
E0: 310 flagged (severe false negatives + high-confidence wrong); spinal-canal
stenosis is the hardest condition (F1 0.625). E1: 474 flagged. E4: disc Dice (0.846)
lags canal/vertebra. See `outputs/real/*_failure_cases.csv`, `*_error_analysis.md`,
and the figures under `outputs/real/figures/` (all gitignored; regenerable).

## v0.9 — Robust auto-inference: decomposing AND closing the oracle→auto gap (REAL)

> Full writeups: `run_logs/gap_decomposition_2x2.md`, `run_logs/robust_auto_experiments.md`,
> `run_logs/safety_mode.md`, `run_logs/localizer_error_profile.md`. **All v0.4–v0.7
> oracle-crop numbers above are an upper bound; the auto-localized distribution is the
> real inference target.**

**1. The gap is in-plane, not slice (2×2 decomposition, no retraining).** Holding the
series fixed and varying only the crop centre (GT vs localizer) and slice (GT vs
geometric-mid), on the same 1955 canal val nodes (87 severe), cluster-bootstrap CIs:

| cell | severe recall [95% CI] | isolates |
|---|---|---|
| GT-xy / GT-slice (oracle) | 0.828 [0.742, 0.904] | upper bound |
| auto-xy / GT-slice | 0.644 [0.528, 0.754] | **in-plane −0.184 [−0.291, −0.089] (decisive)** |
| GT-xy / mid-slice | 0.839 [0.750, 0.918] | slice +0.011 [−0.039, +0.065] (**not** decisive) |
| auto-xy / mid-slice (auto) | 0.644 [0.536, 0.746] | combined |

So the −18 pp collapse is **entirely in-plane crop-centre error**; slice selection is
not the bottleneck (so no slice selector was built). The localizer error is
superior-inferior dominated (σ_y up to 34 px at L1/L2) and heavy-tailed (p99 ~103 px).

**2. Training on the auto distribution recovers it.** E0-architecture canal graders,
each selected/reported on the auto distribution with paired cluster-bootstrap CIs:

| variant | auto severe recall [95% CI] | paired Δ vs deployed E0 |
|---|---|---|
| deployed E0 (all-cond) | 0.644 | — |
| oracle-trained control | 0.667 [0.570, 0.754] | +0.023 (n.s.) |
| level-aware crop jitter | 0.736 [0.634, 0.833] | +0.092 (n.s., p=0.10) |
| **auto-localized crops** | **0.793 [0.696, 0.881]** | **+0.149 [+0.050, +0.243] (decisive; McNemar p=0.007)** |

Training the grader on auto-localized crops recovers **~81%** of the 0.644→0.828 gap,
also decisively improves auto weighted log loss (−0.100) and is best on the oracle
distribution too (0.874). Honest nuance: synthetic jitter helps log-loss and trends on
severe recall but is **not** decisive; matching the real localizer error (auto-train) is.

**3. Safety Mode (auto).** Robust model recall@FAR≤10% **0.851 [0.766, 0.924]** (vs
control 0.816); reaches 90% severe recall at FAR 0.153 (vs 0.192) or ~20% review
burden — tradeoffs reported, not hidden.

## v0.10–v0.12 — LOCKED-TEST, multi-condition, Safety Mode v2 (REAL, v1-track)

> Full writeups: `run_logs/locked_test_protocol.md`, `canal_locked_test.md`,
> `multicondition_robust_results.md`, `safety_mode_v2.md`. These are evaluated on a
> **never-tuned locked test** (splits_v1, patient-level train 1382 / dev 296 / test 296;
> dev = selection, test = final eval only; leakage-tested). The historical seed-1337 val
> numbers above are kept for history and are **not** v1 claims. Models claimed on the
> locked test are retrained on splits_v1 `train`.

**Canal robust auto-inference — CONFIRMED on the locked test** (n=1480, severe=53;
retrained on `train`, selected on `dev` auto, evaluated once on `test`):

| model | test auto severe recall [95% CI] | test oracle |
|---|---|---|
| oracle-trained control | 0.434 [0.306, 0.562] | 0.566 |
| **auto-trained robust** | **0.830 [0.725, 0.929]** | 0.868 |

Paired robust−control (auto, same nodes): **+0.396 [+0.268, +0.529]** (decisive;
McNemar 21 severe recovered / 0 lost, p<1e-6). Robust auto-training reaches **~96% of
the oracle ceiling** on a clean locked test — v0.9 confirmed and amplified out-of-val.

**Multi-condition locked-test oracle baselines** (all-condition E0 retrained on `train`;
oracle = GT-coordinate **upper bound**; auto-localization for non-canal is the documented
frontier, not yet built):

| condition | GT view | locked-test oracle severe recall [95% CI] |
|---|---|---|
| spinal_canal_stenosis | sagittal_t2 | 0.925 [0.849, 0.985] |
| left_neural_foraminal_narrowing | sagittal_t1 | 0.769 [0.638, 0.891] |
| right_neural_foraminal_narrowing | sagittal_t1 | 0.811 [0.712, 0.906] |
| left_subarticular_stenosis | axial_t2 | 0.790 [0.718, 0.857] |
| right_subarticular_stenosis | axial_t2 | 0.854 [0.778, 0.920] |

**View-routing taxonomy (the honest answer to "does v0.9 generalize to all five?").**
Only **canal** has a working auto-localizer (sagittal-T2), so only canal has a confirmed
auto result. **Foraminal** is graded on sagittal-**T1** parasagittal side-specific slices
(needs a side-aware T1 localizer; v0.9's "slice doesn't matter" is canal-specific).
**Subarticular** is graded on **axial-T2** (needs an axial localizer + level matching;
SPIDER has no axial anatomy). Generalization is **gated by view-specific localization,
not by the grading recipe** — documented with evidence, not faked.

**Safety Mode v2 (locked-test auto, canal).** The auto-robust grader gives the strongest
severe-first frontier: recall@FAR≤10% **0.943 [0.833, 1.000]**, reaching 90% severe
recall at **8.5% FAR** (vs the control's 15.5%). A model-disagreement review flag (robust
vs control) flags 13.4% of nodes and captures 22% of the robust model's severe
false-negatives. **Honest negative:** cost-sensitive (expected-cost) *training* is brittle
on this imbalanced 3-class task — it collapses to a moderate-class hedge without class
weighting and over-predicts severe with it; the dev-selected checkpoint is dominated on
the locked test (recall@FAR≤10% 0.264 vs the auto-robust 0.943). The effective severe-aware
recipe is **class-weighted CE + auto-training (auto-robust) + the inference-time decision
layer**, not a cost-sensitive loss. Detail: `safety_mode_v2.md`.

## Runtime (RTX 5080)
RSNA prep 48,692 crops ≈ 5 min; anatomy priors ≈ 3 min; E0/E1 train ≈ 30–45 min
each (AMP, early-stopped); E4 inference 0.85 ms/slice. Canal robust variants ≈ 5–10 min
each (canal-only, AMP, early-stopped); all-condition E0 retrain ≈ 15–20 min. Crops/priors/
runs are cached, gitignored, and resumable.
