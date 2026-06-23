#!/usr/bin/env bash
# E2/E3 — counterfactual anatomy ablations over a trained anatomy-guided run.
set -euo pipefail
cd "$(dirname "$0")/.."
spinescoutx ablate --config configs/ablation.yaml "$@"
