#!/usr/bin/env bash
# Build a finding-graph report + figures for one study from a finished run.
#   make_report.sh <study_id> <run_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY_ID="${1:?usage: make_report.sh <study_id> <run_dir>}"
RUN_DIR="${2:?usage: make_report.sh <study_id> <run_dir>}"
spinescoutx report --study-id "$STUDY_ID" --run "$RUN_DIR"
spinescoutx figure --report "outputs/reports/${STUDY_ID}.json"
