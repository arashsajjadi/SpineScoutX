# Overnight real-data run — baseline status

> Research-only. Not diagnostic. Synthetic smoke is kept clearly separate from real results.

**Branch:** `feature/spinescoutx-real-data-experiments` · start HEAD `7164bdb`

## Pre-flight (Phase 0)
- Working tree clean; tests pass; `ruff check` + `ruff format --check` clean.
- GPU: NVIDIA RTX 5080, 16 GB (≈14 GB free). Disk: 421 GB free.
- Tooling: `gh` authenticated (github.com / arashsajjadi); Ollama present
  (`openbmb/minicpm-v4.5:8b`); `pydicom` + `kaggle` 2.2.2 installed.

## Kaggle credentials (Phase 1)
- Token at `/home/arash/Documents/api_kaggle.txt` is a `KGAT_…` API token (new
  Kaggle format), written to `~/.kaggle/access_token` (chmod 600).
- `kaggle competitions list` → OK; competition files listing → OK (rules accepted).

## E4 SPIDER segmenter (already real, verified — Phase 5)
- `runs/e4_segmentation_spider_real/best.pt` present.
- mean Dice **0.884** · canal **0.902** · vertebra 0.903 · disc 0.846 (official SPIDER val split).

## Real RSNA label/coordinate parse (validated against downloaded CSVs)
- 1,975 studies; **48,803** label rows; **48,692** localizer crops; 6,294 series.
- Severity distribution: normal_mild 37,754 / moderate 7,960 / **severe 3,089 (6.3%)** —
  real class imbalance; severe recall + class weighting matter.
- Conditions/levels/sides all canonicalize correctly; sequence types: axial_t2 2,340 /
  sagittal_t1 1,980 / sagittal_t2 1,974. No parser changes required.

## Next
Full image download in progress (~35 GB) → prepare-rsna → E0 → priors → E1 →
ablation → evidence/calibration → reports/figures → scheduler → docs → gate → push.
