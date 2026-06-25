# Reproducibility — SpineScoutX

> Research-only. Not diagnostic. No data, weights, or caches are committed; everything
> below regenerates from the public datasets and this code.

## Environment

- Python 3.10+, PyTorch 2.x (CUDA recommended; CPU works for inference)
- NVIDIA GPU with ≥ 8 GB VRAM for training; AMP enabled on CUDA
- Install: `pip install -e ".[dicom,parquet]"` (add `SimpleITK` for SPIDER `.mha` files)
- Verify setup: `spinescoutx doctor --data`

## Datasets

| Dataset | Source | Licence | Download |
|---|---|---|---|
| RSNA 2024 Lumbar Spine | Kaggle | CC BY-NC-SA 4.0 | `kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification` |
| SPIDER | Zenodo 10159290 | CC BY 4.0 | `zenodo_get 10159290` or manual download |

Place datasets at `data/raw/rsna/` and `data/raw/spider/` (both gitignored).

## Splits

```bash
python scripts/build_splits_v1.py
```

Creates `data/cache/splits_v1/splits.json`: 1382 train / 296 dev / 296 test studies
(patient-level, seed 1337). **The locked test was never used for model selection or tuning.**

## Reproduce best locked-test metrics (v1.9)

```bash
python scripts/reproduce_best_metrics_v1_9.py
```

Reads stored predictions from `runs/v1_*` and evaluates on `splits_v1` test split.
Reproduces: canal 0.830, L-for 0.788, R-for 0.660, L-sub 0.746, R-sub 0.737, macro 0.752.

## Reproduce charts

```bash
python scripts/create_v1_9_charts.py
```

Writes 6 PNG + SVG charts to `docs/assets/v1_9/`. No data needed — uses hardcoded metrics.

## Regenerate real-case gallery

```bash
python scripts/check_real_image_release_gate_v1_9.py   # run gate first
python scripts/generate_readme_assets_v1_9.py           # requires RSNA cache
```

Requires: `data/cache/rsna_auto_foraminal/manifest.parquet` (RSNA crops prepared).
Writes 12 anonymized PNG panels to `docs/assets/v1_9/real_cases/`.

## Verify model package checksums

```bash
python scripts/verify_release_assets_v1_9.py
```

Downloads (or checks local) `spinescoutx-best-raw-v1.9.tar.gz` and verifies SHA-256
against `docs/assets/v1_9/checksums.txt`.

## Full training pipeline (from scratch)

```bash
spinescoutx doctor --data                                        # verify datasets
spinescoutx prepare-rsna --rsna-root data/raw/rsna \
    --out data/cache/rsna_auto_foraminal                         # crop+manifest
spinescoutx prepare-spider --spider-root data/raw/spider \
    --out data/cache/spider                                      # SPIDER slices

spinescoutx train-classifier --config configs/real_e0_baseline_rsna.yaml  # canal/foraminal/subarticular
python scripts/run_multicondition_v1.py                          # all-condition eval
python scripts/run_canal_locked_test.py                          # canal locked-test
python scripts/run_foraminal_locked_test.py                      # foraminal locked-test
python scripts/run_subarticular_locked_test.py                   # subarticular locked-test
```

Trained model runs go to `runs/` (gitignored). Intermediate caches go to `data/cache/`
(gitignored). Outputs and metrics go to `outputs/real/` (gitignored).

## Fixed choices

- **Seed:** global 1337, data split_seed 1337
- **Crops:** 2.5D (prev/center/next), 224², robust percentile normalization
- **Architecture:** ConvNeXt-Tiny, ImageNet pretrained, 3-class severity head
- **Training:** frozen-backbone warmup → gentle fine-tuning (backbone lr × 0.2);
  early stopping on val_weighted_logloss (patience 7); class-weighted loss
- **Evaluation:** cluster-bootstrap 95% CIs (n_boot=2000); paired tests for deltas

## Determinism note

`spinescoutx.utils.seed.seed_everything` seeds Python/NumPy/Torch and requests deterministic
cuDNN. DataLoader workers are seeded. Minor floating-point differences may occur across GPU
models or PyTorch versions but headline metrics reproduce within CI bounds.

## Verification gate

```bash
python -m pytest -q           # 252 tests, all on synthetic fixtures (no real data needed)
ruff check .
ruff format --check .
python -m build
spinescoutx doctor --data
python scripts/check_release_safety_v1_9.py
```

## What is gitignored

Caches, masks, features, logits, runs, model weights, tokens, and output parquet files are
all gitignored (see `.gitignore`). Only code, configs, docs, and safe metric metadata are
committed. Intermediate files are reproducible from the commands above.
