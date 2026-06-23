# SpineScoutX — Technical Report

> **Research-only.** Not diagnostic, not clinically validated, not for medical
> decision-making. Uses public non-commercial research data; no data is committed.

## 0. Status of this report

This repository ships the **complete, tested pipeline** plus a **synthetic
smoke** that exercises every stage end-to-end. **The real RSNA and SPIDER
datasets were not present in this environment, so no real-data training was run
and no real metrics are reported.** Every number in §9–§13 below comes from the
*synthetic* smoke and exists only to prove the code executes and the metrics are
wired correctly. **These synthetic numbers are not research results and must not
be read as evidence for or against the hypothesis.** To produce real results,
follow [`data_setup.md`](data_setup.md) and run E0/E4/E1/ablation on the datasets.

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

## 9. Results — SYNTHETIC SMOKE ONLY (pipeline validation, not research results)

Synthetic config: `n≈48`, `crop=48`, `backbone=small_cnn`, CPU, 2 epochs,
`max_steps=6`. Held-out synthetic val ≈ 9–10 samples. **Interpret nothing here as
a finding** — these confirm the metric plumbing and that every stage runs.

| Experiment (synthetic) | weighted log loss | macro F1 | severe recall | severe FNR | ECE | ECE (post-temp) |
|---|---|---|---|---|---|---|
| E0 image-only | 1.032 | 0.417 | 0.333 | 0.667 | 0.273 | 0.260 |
| E1 anatomy-guided | 1.199 | 0.222 | 0.000 | 1.000 | 0.148 | 0.167 |

E4 segmenter (synthetic): val mean Dice ≈ 0.051 (2 epochs, 6 steps — untrained).

E7 runtime (synthetic, `small_cnn`, batch 8, crop 48, CPU): ≈ 0.76 ms/batch
(≈ 0.095 ms/sample) forward. On the available RTX 5080, real configs would use
AMP + GPU.

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

`spinescoutx benchmark --run <run>` times a warm forward pass and reports
per-batch / per-sample latency and (on CUDA) peak memory. Synthetic CPU numbers in
§9. Real-data preprocessing/training timings depend on dataset size and are
produced by the same command once a real run exists.

## 14. Failure cases

Reported honestly: (a) the synthetic E1<E0 ordering above; (b) the untrained
synthetic segmenter (Dice ≈ 0.05); (c) anatomically inconsistent Grad-CAM
heatmaps captured in the failure-case panel. None are cherry-picked away.

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
