# SpineScoutX — Technical Report

> **Research-only.** Not diagnostic, not clinically validated, not for medical
> decision-making. Uses public non-commercial research data; no data is committed.

## 0. Status of this report

**All experiments below were run on REAL data** (RSNA/LumbarDISC via Kaggle; SPIDER
via Zenodo 10159290, CC BY 4.0). Headline real results — full detail and honest
interpretation in [`results.md`](results.md):

- **E4 SPIDER segmentation:** mean Dice **0.884** (canal 0.902), official split.
- **E0 image-only baseline (RSNA val, 9,745 crops):** weighted log loss **0.462**,
  macro F1 0.706, severe recall **0.751**, severe AUROC **0.971**, ECE 0.027.
- **E1 anatomy-guided:** weighted log loss 0.458, macro F1 0.717, severe recall
  0.711, ECE 0.034 — a marginal, mixed change vs E0.
- **Counterfactual ablation (decisive):** zeroing / shuffling / noising the anatomy
  prior changes weighted log loss by **< 0.001** ⇒ **E1 largely ignores the anatomy
  branch**; the small E1>E0 aggregate edge is **not** attributable to anatomy.

**Answer to the research question:** in this implementation, explicit anatomy priors
do **not** meaningfully improve disc-level grading, and the model does not rely on
them (mean AEC ≈ 0.10, flat across perturbations). We report this **negative/nuanced
result honestly** — the ablation is precisely what prevents a false "anatomy helps"
claim from the tiny aggregate edge. The synthetic smoke (§9.2) is retained only as a
no-data CI path and is clearly separated from these real results.

## 1. Abstract

SpineScoutX studies whether explicit anatomical priors (vertebra/disc/spinal-canal
masks) improve disc-level lumbar degeneration *finding grading*. It pairs an
image-only baseline (E0) with an anatomy-guided model (E1) that consumes anatomy
priors transferred from a SPIDER-trained segmenter (E4), and adds anatomical
evidence-consistency (AEC) scoring, calibration, counterfactual anatomy ablations,
and deterministic, non-diagnostic structured finding graphs. It is a research
prototype, not a clinical product, and does not claim to be the first AI lumbar
spine system.

## 2. Background and related work

Automated lumbar MRI grading and stenosis assessment is an active area;
DeepSPINE and the RSNA 2024 Lumbar Spine Degenerative Classification challenge are
representative prior work. SpineScoutX does **not** claim novelty of lumbar AI
grading itself. Its contribution is scoped to: (i) anatomy-grounded finding-graph
generation, (ii) cross-dataset anatomy transfer (SPIDER → RSNA), (iii) anatomical
evidence-consistency scoring, (iv) counterfactual anatomy ablation, (v)
calibration-aware non-diagnostic reporting, and (vi) minimal, license-aware,
reproducible engineering.

## 3. Research question

Can explicit anatomical priors improve reliability, severe-case recall,
calibration, and visual evidence localization of disc-level lumbar degeneration
grading — or at least yield more anatomically meaningful evidence even when raw
classification metrics do not improve?

## 4. Datasets

| Dataset | Role | License |
|---|---|---|
| RSNA 2024 Lumbar Spine Degenerative Classification | grading (primary) | non-commercial research |
| SPIDER Lumbar Spine Segmentation | anatomy priors | CC BY 4.0 |

Conditions: spinal canal stenosis; L/R neural foraminal narrowing; L/R
subarticular stenosis. Levels L1/L2…L5/S1. Severity: `normal_mild`/`moderate`/
`severe`. SPIDER labels are **anatomy**, not pathology. See `data_setup.md`.

## 5. Methods

- **Ingestion:** lazy `pydicom` decode → robust percentile-clip normalization →
  disc-level localizer matching → 2.5D crops (prev/center/next slice; missing
  neighbors duplicated/zero-padded with a recorded `pad_note`). Crops cached as
  `.npy`; manifests as Parquet/CSV with full provenance (`CropRecord`).
- **Splits:** patient/study-level only, deterministic, leakage-checked; seed +
  timestamp + counts saved.
- **Anatomy priors:** SPIDER segmenter (E4) predicts 4-class masks
  (background/vertebra/disc/canal), remapped via a documented, *approximate*
  label collapse. For RSNA, priors come only from a *valid* segmenter cache; if
  absent, guided runs fail clearly rather than fabricate priors.
- **Evidence:** Grad-CAM heatmaps → AEC = on-target heatmap mass / total mass.
- **Calibration:** ECE + reliability diagram + optional temperature scaling
  (clamped to a sane range); uncertainty flags from calibrated confidence.
- **Reporting:** deterministic finding-graph JSON → Markdown report → panels.

## 6. Model architecture

- **E0 image-only:** timm backbone (default ConvNeXt-Tiny; `small_cnn` offline) on
  2.5D crops + level/condition embeddings → 3-class head. Frozen-backbone warmup
  then partial fine-tune; AMP on CUDA; class-weighted loss or weighted sampler.
- **E4 segmenter:** lightweight 2D U-Net (`monai` UNet optional), Dice+CE.
- **E1 anatomy-guided:** image encoder + anatomy encoder (disc/canal/vertebra
  channels) + level/condition embeddings → concat fusion → 3-class head. No
  transformer/GNN/LLM; the finding graph is deterministic.

## 7. Experiments

E0 image-only · E1 anatomy-guided · E2 shuffled-anatomy ablation · E3
zero/noise-anatomy ablation · E4 SPIDER segmenter · E5 evidence (Grad-CAM/AEC) ·
E6 calibration · E7 runtime. Each is a CLI command; configs in `configs/`.

## 8. Metrics

Classification: RSNA-weighted log loss, macro/per-condition/per-level F1, severe
recall, severe FNR, severe one-vs-rest AUROC, confusion, balanced accuracy, ECE.
Segmentation: per-class Dice/IoU, mean Dice, canal Dice, latency. Evidence: AEC,
leakage, peak-to-localizer distance, completeness, uncertainty coverage. Runtime:
prep/train/infer time, GPU peak memory, CPU fallback.

## 9. Results

### 9.1 E4 — SPIDER anatomy segmentation (REAL)

Real data: SPIDER (Zenodo 10159290, CC BY 4.0). Preprocessing cached **10,338
sagittal 2D slices** from **447 volumes / 218 patients** (both T1 and T2),
foreground-filtered, normalized, resized to 256². Split is SPIDER's **official**
subset split (360 train / 87 val volumes → 8,040 / 2,298 slices). Model: built-in
2D U-Net, Dice+CE loss, AMP on an RTX 5080, 25 epochs (best by val mean Dice at
epoch 16, early-stopped). These are **real research metrics** on the official
validation split.

| Class | Dice | IoU |
|---|---|---|
| Vertebra | 0.903 | 0.823 |
| Disc | 0.846 | 0.733 |
| **Spinal canal** | **0.902** | 0.822 |
| **Mean (foreground)** | **0.884** | 0.793 |

Disc Dice (0.846) is the weakest class — expected, since intervertebral discs are
thin and adjacent, so boundary slices cost the most overlap. Canal and vertebra
both exceed 0.90. Qualitative best/worst overlays:
`outputs/real/figures/e4_segmentation_examples.png` /
`e4_segmentation_failures.png` (failure cases shown, not hidden). These are
**anatomy** masks, not pathology/stenosis masks.

### 9.2 E0 / E1 / ablation — REAL RSNA (research results)

Real RSNA val (9,745 crops, study-level split). Full detail + interpretation in
[`results.md`](results.md).

| metric | E0 image-only | E1 anatomy-guided |
|---|---|---|
| weighted log loss ↓ | **0.4621** | **0.4579** |
| macro F1 ↑ | 0.706 | 0.717 |
| severe recall ↑ | **0.751** | 0.711 |
| severe AUROC ↑ | 0.971 | 0.972 |
| ECE ↓ | 0.027 | 0.034 |

Ablation (E1 model, real val): correct/shuffled/zero/noise all give weighted log
loss ≈ 0.458 (|Δ| < 0.001) and severe recall ≈ 0.71 ⇒ **the anatomy branch is
largely ignored**; the marginal E1>E0 edge on aggregate metrics is **not** due to
anatomy. Mean AEC ≈ 0.10, flat across modes. This is an honest negative/nuanced
result, not an improvement claim.

### 9.3 Synthetic smoke (CI only — not research results)

The same stages also run on tiny synthetic fixtures (`n≈48`, `small_cnn`, CPU) as a
no-data CI path. **Interpret nothing here as a finding.**

| Experiment (synthetic) | weighted log loss | macro F1 | severe recall | severe FNR | ECE | ECE (post-temp) |
|---|---|---|---|---|---|---|
| E0 image-only | 1.032 | 0.417 | 0.333 | 0.667 | 0.273 | 0.260 |
| E1 anatomy-guided | 1.199 | 0.222 | 0.000 | 1.000 | 0.148 | 0.167 |

> The E1<E0 ordering above is a **small-synthetic-data / random-init artifact**,
> not evidence about anatomy priors. With ~10 val samples and 6 training steps,
> neither model has learned; the comparison is meaningless by construction and is
> shown only to demonstrate the comparison harness works.

## 10. Ablations — SYNTHETIC SMOKE ONLY

Counterfactual anatomy perturbations on the (untrained, synthetic) E1 model,
Δ vs the `correct` prior:

| Mode | Δ severe recall | Δ weighted log loss |
|---|---|---|
| shuffled (other study) | 0.000 | −0.001 |
| zero | 0.000 | −0.006 |
| noise | 0.000 | +0.008 |

Near-zero deltas are expected: an untrained synthetic model barely uses the
anatomy channel, so perturbing it changes little. On real data this ablation is
the key test of whether the model *relies* on anatomy; here it only proves the
ablation machinery (perturb → re-evaluate → delta) is correct.

## 11. Calibration

ECE and reliability diagrams are computed for every classification run; optional
temperature scaling is fit on the held-out split (a research-demo simplification —
ideally fit on a separate calibration split) and clamped to `[1e-2, 100]` to avoid
degenerate fits on tiny data. Uncertainty flags: `high_confidence` (≥0.85),
`moderate_confidence` (≥0.60), else `review_required`. See
`outputs/figures/reliability_diagram.png`.

## 12. Evidence consistency

AEC = (heatmap mass inside the target anatomy region) / (total heatmap mass).
Target regions: spinal-canal stenosis → the real `spinal_canal` mask
(`source=anatomy`); foraminal/subarticular → side-aware approximations
(`source=approximate`, surfaced in every report). Leakage = 1 − AEC; we also
report peak-to-localizer distance. Evidence overlays are saved and failure cases
(anatomically inconsistent heatmaps) are shown, not hidden — see
`outputs/figures/failure_cases.png` and the FAILURE cell in the hero panel.

## 13. Runtime benchmarks

### 13.1 E4 — SPIDER segmentation (REAL, RTX 5080)

- Preprocessing (decode 447 .mha volumes → 10,338 cached slices): **96 s** (~0.2 s/volume).
- Training: 25 epochs (early-stopped), ~30 s/epoch with AMP; GPU util ~94%.
- Inference: **0.85 ms/slice** (p90 0.89 ms), GPU peak **≈ 83 MB** (batch 1, 256²).
- `outputs/real/e4_segmentation_metrics.json` holds the machine-readable record.

### 13.2 Classifier path (synthetic)

`spinescoutx benchmark --run <run>` times a warm forward pass and reports
per-batch / per-sample latency and (on CUDA) peak memory. Synthetic CPU numbers in
§9.2. Real RSNA classifier timings are pending RSNA availability.

## 14. Failure cases

Reported honestly: (a) **E4 (real):** disc Dice (0.846) lags vertebra/canal
(~0.90); the worst-case overlays in `outputs/real/figures/e4_segmentation_failures.png`
show disc-boundary and end-plate confusions — shown, not hidden. (b) the synthetic
E1<E0 ordering (§9.2), a small-data artifact, not a finding. (c) anatomically
inconsistent Grad-CAM heatmaps in the synthetic evidence panel. None cherry-picked.

## 15. Limitations

- **No real-data results** in this environment (datasets absent).
- **Anatomy ≠ pathology:** SPIDER priors are anatomy masks, not stenosis masks.
- **Approximate regions** for foraminal/subarticular AEC (flagged everywhere).
- **No clinical or external validation.**
- Temperature scaling fit on the eval split (demo simplification).

## 16. Ethics and safety

Research/education only; not a medical device; not diagnostic; no PHI assumptions;
no telemetry; no hidden network calls. Datasets are non-commercial / CC BY 4.0 and
are never redistributed. Reports carry standing non-diagnostic disclaimers and
limitation lists. Forbidden marketing/clinical claims are enumerated in the README
and never asserted.

## 17. Future work

Run E0/E4/E1 + ablations on real RSNA/SPIDER; calibrate on a dedicated split;
quantify AEC on real anatomy masks; add per-condition/per-level breakdowns at
scale; optional qualitative external sanity check (no generalization claims
without verified splits/labels/licenses).
