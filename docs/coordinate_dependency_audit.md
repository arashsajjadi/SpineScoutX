# Coordinate-dependency audit (oracle-crop vs auto-crop)

> Research-only. This audit makes the **provenance** of every grading metric explicit.
> Independently verified by a 3-way fan-out audit of the data / training / reporting
> subsystems.

## Headline finding
**Every crop in the current pipeline is centred on the RSNA ground-truth localizer
coordinate `(x, y)` from `train_label_coordinates.csv`, for *both* the train and the
val split.** Therefore **all previously reported E0 / E1 / E2 grading metrics are
`oracle_crop`** — a research upper bound that assumes a perfect disc-level localizer.
They are *not* real end-to-end inference numbers. This was honest at the crop-classifier
stage but must be labelled; v0.6 adds an `auto_crop` path to measure the real gap.

## Where GT coordinates / labels are used
| step | file | uses | classification | inference-safe |
|---|---|---|---|---|
| crop placement (`extract_25d` at GT x,y) | `data/rsna_prepare.py` | coordinates | **inference_forbidden** | ❌ |
| `load_coordinates` (train_label_coordinates.csv) | `data/rsna_labels.py` | coordinates | **inference_forbidden** | ❌ |
| `crop_bounds`/`extract_crop` (fed GT x,y here) | `data/crops.py` | coordinates | inference_forbidden (in this pipeline) | ❌ |
| `CropRecord.x/y` + `crop_path` (GT-centred pixels) | `data/crops.py` | both | inference_forbidden | ❌ |
| `RsnaCropDataset` image = GT-centred crop | `data/datasets.py` | crop_paths | **inference_forbidden** | ❌ |
| anatomy prior keyed by GT crop_path | `data/{datasets,anatomy_priors}.py` | crop_paths | inference_forbidden | ❌ |
| `load_labels` severity (supervision / val target) | `data/rsna_labels.py` | labels | training_only / **val_target** | ✅ (not a localizer leak) |
| series index / `classify_sequence` | `data/rsna_index.py` | none | neutral | ✅ |
| training loaders / metrics (read crop_path + severity only) | `training/`, `evaluation/` | crop_paths/labels | inherits crop provenance | ❌ for images, ✅ for the label target |

**Severity labels are NOT a leak**: on held-out studies the severity is the
evaluation answer key (`val_target`); on train studies it is supervision. The leak
risk is entirely the **localizer coordinates**, which place the crop.

## Provenance modes (v0.6)
- **`oracle_crop`** — crops centred on GT coordinates. Research upper bound. All
  v0.4–v0.5 numbers. Cache: `data/cache/rsna`.
- **`auto_crop`** — crops centred on the *predicted* disc-level localizer (Phase 2).
  Real inference. Cache: `data/cache/rsna_auto`. **Must not read
  `train_label_coordinates.csv` at inference.**
- **`hybrid_debug`** — GT used only for QA / localization-error measurement, never
  for a grading-metric claim.

`CropRecord.coordinate_source` ∈ {`oracle`, `auto`} now tags every crop, and the
auto pipeline asserts it never opens the coordinates CSV. Every metric table states
its provenance.

## Consequence for v0.6
The honest research question becomes: **how much does grading degrade from
`oracle_crop` → `auto_crop`?** That gap measures the project's real dependence on GT
coordinates and is the headline v0.6 result.

## Disc-level localizer (Phase 2)
A heatmap-regression UNet (`disc_localizer`, 5 disc-level channels) predicts the
L1/L2…L5/S1 keypoints on the **geometric mid sagittal-T2 slice** — chosen from the
series index, never from GT. Trained on the RSNA canal localizer coordinates
(supervision only). Held-out val (`runs/l0_disc_localizer_real`):

| metric | value |
|---|---|
| median keypoint error | **2.53 px** |
| mean keypoint error | 9.21 px |
| PCK@10 / @20 / @32 | 0.761 / 0.768 / 0.941 |
| crop-hit@224 (GT point inside an auto crop) | **0.998** |
| per-level mean px (L1/L2 → L5/S1) | 12.9 / 12.8 / 9.9 / 6.0 / 4.7 |

Upper levels (L1/L2) are hardest (near the slice edge); L5/S1 easiest.

## Measured oracle → auto gap (Phase 2 headline)
Same trained models, same **1955 matched (study, level) canal-stenosis val pairs**
(severity-target agreement **1.000**); the *only* change is crop centre — GT
coordinate (`oracle`) vs localizer prediction + geometric mid slice (`auto`).
`outputs/real/oracle_auto_gap.json`.

| model | metric | oracle_crop | auto_crop | Δ (auto − oracle) |
|---|---|---|---|---|
| **E0** image-only | weighted-logloss ↓ | 0.326 | 0.554 | **+0.228** |
| | severe recall ↑ | 0.828 | 0.644 | **−0.184** |
| | severe AUROC ↑ | 0.978 | 0.919 | −0.059 |
| | balanced acc ↑ | 0.701 | 0.646 | −0.055 |
| | ECE ↓ | 0.034 | 0.036 | +0.003 |
| **E2** anatomy-forced | weighted-logloss ↓ | 0.351 | 0.651 | **+0.300** |
| | severe recall ↑ | 0.759 | 0.621 | **−0.138** |
| | severe AUROC ↑ | 0.979 | 0.920 | −0.060 |
| | balanced acc ↑ | 0.777 | 0.706 | −0.071 |
| | ECE ↓ | 0.036 | 0.053 | +0.017 |

**Interpretation.** Despite a near-perfect crop-hit@224 (0.998), removing GT
coordinates costs ~0.23–0.30 weighted-logloss and **14–18 pp of severe recall**.
crop-hit@224 only asks whether the GT point lies *somewhere* inside the crop; the
classifier is far more sensitive — it was trained on crops centred exactly on the GT
point and slice, so the residual in-plane offset (median 2.5 px) **plus** the
auto-path's geometric-mid-slice choice (vs the GT-marked instance) together produce a
real distribution shift. This is the honest cost of real end-to-end inference and the
quantitative answer to "how much did the oracle crop flatter us": **the strong v0.4–0.5
numbers are an upper bound; a deployable localizer-driven system is materially weaker,
especially on the safety-critical severe class.** Closing this gap (better slice
selection, localizer-aware crop augmentation, multi-view evidence) is the work of
v0.6/0.7 Phases 4–8.
