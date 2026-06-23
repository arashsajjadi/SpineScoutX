#!/usr/bin/env bash
# E4 — SPIDER anatomy segmenter. Requires a prepared SPIDER cache.
set -euo pipefail
cd "$(dirname "$0")/.."
spinescoutx train-segmenter --config configs/segmentation_spider.yaml "$@"
spinescoutx evaluate --run runs/e4_segmentation_spider
