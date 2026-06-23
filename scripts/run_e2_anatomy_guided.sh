#!/usr/bin/env bash
# E1 — anatomy-guided classifier. Requires RSNA cache + a valid segmenter mask cache.
set -euo pipefail
cd "$(dirname "$0")/.."
spinescoutx train-anatomy-guided --config configs/anatomy_guided.yaml "$@"
spinescoutx evaluate --run runs/e1_anatomy_guided
