#!/usr/bin/env bash
# Turnkey REAL RSNA study: E0 -> anatomy priors -> E1 -> ablation -> report.
#
# Prerequisites (the ONLY manual step is obtaining the credential-gated RSNA data;
# see docs/data_status.md):
#   - RSNA/LumbarDISC under data/raw/rsna   (Kaggle or RSNA MIRA; needs your login)
#   - A trained SPIDER segmenter at runs/e4_segmentation_spider_real/best.pt
#     (produced by: spinescoutx train-segmenter --config configs/real_e4_spider_segmentation.yaml)
#
# Research-only. Not diagnostic. Does not fabricate metrics; if RSNA is absent it
# fails fast with the exact next action.
set -euo pipefail
cd "$(dirname "$0")/.."

RSNA_ROOT="${RSNA_ROOT:-data/raw/rsna}"
RSNA_CACHE="${RSNA_CACHE:-data/cache/rsna}"
PRIOR_CACHE="${PRIOR_CACHE:-data/cache/rsna_anatomy_priors}"
SEG_RUN="${SEG_RUN:-runs/e4_segmentation_spider_real}"

if [ ! -f "$RSNA_ROOT/train_label_coordinates.csv" ]; then
  echo "ERROR: RSNA data not found under $RSNA_ROOT." >&2
  echo "RSNA/LumbarDISC is credential-gated (Kaggle competition rules OR RSNA MIRA login)." >&2
  echo "See docs/data_status.md for the exact acquisition steps, then re-run this script." >&2
  exit 2
fi
if [ ! -f "$SEG_RUN/best.pt" ]; then
  echo "ERROR: no trained SPIDER segmenter at $SEG_RUN/best.pt." >&2
  echo "Run: spinescoutx train-segmenter --config configs/real_e4_spider_segmentation.yaml" >&2
  exit 2
fi

echo "==> [1/6] prepare RSNA crops + manifest"
spinescoutx prepare-rsna --rsna-root "$RSNA_ROOT" --out "$RSNA_CACHE" "$@"

echo "==> [2/6] train E0 image-only baseline"
spinescoutx train-classifier --config configs/real_e0_baseline_rsna.yaml
spinescoutx evaluate --run runs/e0_baseline_real

echo "==> [3/6] generate RSNA anatomy priors (SPIDER E4 -> RSNA transfer)"
spinescoutx prepare-anatomy-priors --rsna-cache "$RSNA_CACHE" --segmenter-run "$SEG_RUN" --out "$PRIOR_CACHE"

echo "==> [4/6] train E1 anatomy-guided classifier"
spinescoutx train-anatomy-guided --config configs/real_e1_anatomy_guided.yaml
spinescoutx evaluate --run runs/e1_anatomy_guided_real

echo "==> [5/6] counterfactual anatomy ablation (correct/shuffled/zero/noise)"
spinescoutx ablate --config configs/ablation.yaml

echo "==> [6/6] done. E0 vs E1 metrics are in runs/*/metrics.json and outputs/real/."
echo "Generate a finding-graph report with:"
echo "  spinescoutx report --study-id <STUDY_ID> --run runs/e1_anatomy_guided_real"
