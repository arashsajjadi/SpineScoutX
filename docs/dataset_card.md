# Dataset card — SpineScoutX

> Research-only, non-commercial. **No dataset files are committed or redistributed.**
> Each dataset is used under its own license; obtain them yourself.

## RSNA 2024 Lumbar Spine Degenerative Classification (LumbarDISC)
- **Role:** primary disc-level degenerative **finding grading** dataset.
- **Source / license:** Kaggle competition + RSNA MIRA; **non-commercial research
  use** under the competition/RSNA terms. Not redistributed here.
- **Size (this run):** 1,975 train studies, 147,218 DICOMs; 1,974 studies carry
  localizers → 6,291 series → **48,692 localizer findings**.
- **Conditions (5):** spinal canal stenosis; left/right neural foraminal narrowing;
  left/right subarticular stenosis. **Levels (5):** L1/L2…L5/S1.
- **Severity (3, ordinal):** `normal_mild`, `moderate`, `severe`. Class balance is
  skewed: normal_mild 77% · moderate 16% · **severe 6.3%**. RSNA-style sample
  weights (1/2/4) and class-weighting are used.
- **Localizer sequences:** sagittal_t1 19,724 · axial_t2 19,220 · sagittal_t2 9,748.
- **Splits:** patient/study-level only (38,947 train / 9,745 val crops);
  **0 study leakage** verified. The 35 unlabeled findings are dropped before the loss.
- **Preprocessing:** decode-once-per-series → robust percentile-normalized →
  2.5D (prev/center/next) 224² crops around each localizer; cached `.npy`.

## SPIDER Lumbar Spine Segmentation
- **Role:** **anatomy** segmentation → anatomy priors (vertebra / disc / spinal canal).
- **Source / license:** Zenodo record **10159290**, **CC BY 4.0** (attribution
  required; van der Graaf et al.). Not redistributed here.
- **Size:** 218 patients, 447 volumes (T1+T2) → 10,338 cached 256² sagittal slices,
  SPIDER's **official** split (8,040 train / 2,298 val).
- **Label remap (approximate, documented):** raw ids → 4 classes
  (background / vertebra 1–49 / spinal_canal=100 / disc≥200), verified against real
  masks (vertebra 1–7, canal 100, discs 201–207).
- **Important:** these are **anatomy** masks, **not** pathology/stenosis masks.

## Cross-dataset transfer
The SPIDER-trained segmenter (E4) predicts anatomy masks on RSNA source slices; the
disc/canal/vertebra channels are cropped to each RSNA localizer to form per-crop
anatomy priors. **Region validity:** spinal-canal-stenosis evidence uses the real
canal mask (`anatomy`); foraminal/subarticular evidence regions are **approximate**
(SPIDER has no foraminal/lateral-recess labels) and are flagged everywhere.

## What is never committed
DICOMs, NIfTI/MHA volumes, masks, crops, caches, model weights, checkpoints, runs,
or generated patient-like outputs. Enforced by `.gitignore` +
`tests/test_no_data_committed.py`.
