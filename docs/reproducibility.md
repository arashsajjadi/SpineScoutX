# Reproducibility — SpineScoutX

> Research-only. No data, weights, caches, or runs are committed; everything below
> regenerates from the public datasets + this code.

## Environment
- Python 3.13, PyTorch 2.11 (CUDA), single NVIDIA RTX 5080 (16 GB). AMP enabled on CUDA.
- Install: `pip install -e ".[dicom,parquet]"` (+ `SimpleITK` for SPIDER `.mha`).
- Determinism: `spinescoutx.utils.seed.seed_everything` seeds Python/NumPy/Torch
  and requests deterministic cuDNN; DataLoader workers are seeded.

## Fixed choices
- **Seeds:** global `1337`; data `split_seed = 1337`.
- **Splits:** patient/study-level only (no image-level), leakage-checked. SPIDER
  uses its **official** subset split.
- **Crops:** 2.5D (prev/center/next) 224², robust percentile normalization.
- **Caches** (gitignored): `data/cache/rsna` (crops+manifest),
  `data/cache/rsna_anatomy_priors` (priors), `data/cache/spider` (slices+index).

## End-to-end commands
```bash
# datasets -> data/raw/{rsna,spider}  (see docs/data_status.md; credentials required)
spinescoutx doctor --data
spinescoutx prepare-rsna   --rsna-root data/raw/rsna   --out data/cache/rsna
spinescoutx prepare-spider --spider-root data/raw/spider --out data/cache/spider

spinescoutx train-segmenter      --config configs/real_e4_spider_segmentation.yaml   # E4
spinescoutx train-classifier     --config configs/real_e0_baseline_rsna.yaml         # E0
spinescoutx prepare-anatomy-priors --rsna-cache data/cache/rsna \
    --segmenter-run runs/e4_segmentation_spider_real --out data/cache/rsna_anatomy_priors
spinescoutx train-anatomy-guided --config configs/real_e1_anatomy_guided.yaml        # E1
spinescoutx ablate               --config configs/ablation.yaml                      # E2/E3
spinescoutx evaluate --run runs/e1_anatomy_guided_real
spinescoutx report   --study-id <ID> --run runs/e1_anatomy_guided_real
```
Or one resumable command: `bash scripts/run_full_spinescoutx_research.sh`
(skips completed phases; logs to `outputs/real/`).

## Training knobs that matter
- Frozen-backbone warmup (`freeze_backbone_epochs`) then **gentle** fine-tuning
  (`backbone_unfreeze_lr_scale = 0.2`, i.e. backbone lr = `lr × 0.2`) — avoids the
  unfreeze "shock". Early stopping on `val_weighted_logloss` (patience 7).
- Class-weighted loss for the skewed severity distribution (severe ≈ 6%).
- E0 and E1 share `split_seed` so their val sets match for a fair comparison.

## Verification gate
`pytest -q` · `ruff check .` · `ruff format --check .` · `python -m build` ·
`spinescoutx doctor --data` · no-data-committed test. CI passes on synthetic
fixtures with no real data.
