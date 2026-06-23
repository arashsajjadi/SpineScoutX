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
