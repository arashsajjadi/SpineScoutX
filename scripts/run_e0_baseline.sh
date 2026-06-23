#!/usr/bin/env bash
# E0 — image-only baseline classifier. Requires a prepared RSNA cache.
set -euo pipefail
cd "$(dirname "$0")/.."
spinescoutx train-classifier --config configs/baseline_image_only.yaml "$@"
spinescoutx evaluate --run runs/e0_baseline_image_only
