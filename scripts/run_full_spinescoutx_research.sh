#!/usr/bin/env bash
# Resumable end-to-end SpineScoutX research orchestrator.
#
# Runs every phase in order, SKIPS phases whose output already exists (resume),
# tees a per-phase log, records timings to outputs/real/run_timeline.md, and a
# machine-readable state to outputs/real/run_state.json. Fails fast on missing
# data or a failed critical step. Research-only; commits/pushes are NOT done here.
#
# Usage:
#   bash scripts/run_full_spinescoutx_research.sh            # full real run
#   RSNA_LIMIT=300 bash scripts/run_full_spinescoutx_research.sh   # quick subset
set -uo pipefail
cd "$(dirname "$0")/.."

RSNA_ROOT="${RSNA_ROOT:-data/raw/rsna}"
RSNA_CACHE="${RSNA_CACHE:-data/cache/rsna}"
PRIOR_CACHE="${PRIOR_CACHE:-data/cache/rsna_anatomy_priors}"
SEG_RUN="${SEG_RUN:-runs/e4_segmentation_spider_real}"
LOGDIR="outputs/real/logs"
STATE="outputs/real/run_state.json"
TIMELINE="outputs/real/run_timeline.md"
RSNA_LIMIT="${RSNA_LIMIT:-}"
mkdir -p "$LOGDIR" "$(dirname "$STATE")"
: > "$STATE.tmp"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

# run_step <name> <marker-file-or-empty> <command...>
# Skips when the marker exists; otherwise runs, tees a log, records status+seconds.
run_step() {
  local name="$1" marker="$2"; shift 2
  if [[ -z "${IN_PROFILE[$name]:-}" ]]; then
    return 0  # step not selected by the active --profile
  fi
  if [[ -n "$marker" && -e "$marker" ]]; then
    say "SKIP  $name (found $marker)"
    echo "| $name | skipped | - | $marker |" >> "$TIMELINE"
    echo "  \"$name\": {\"status\": \"skipped\"}," >> "$STATE.tmp"
    return 0
  fi
  say "RUN   $name"
  local start; start=$SECONDS
  if "$@" > "$LOGDIR/${name}.log" 2>&1; then
    local dur=$((SECONDS - start))
    say "OK    $name (${dur}s)"
    echo "| $name | ok | ${dur}s | $marker |" >> "$TIMELINE"
    echo "  \"$name\": {\"status\": \"ok\", \"seconds\": $dur}," >> "$STATE.tmp"
  else
    local rc=$? dur=$((SECONDS - start))
    say "FAIL  $name (rc=$rc, ${dur}s) — see $LOGDIR/${name}.log"
    echo "| $name | FAIL | ${dur}s | rc=$rc |" >> "$TIMELINE"
    echo "  \"$name\": {\"status\": \"fail\", \"rc\": $rc}," >> "$STATE.tmp"
    finalize 1
  fi
}

finalize() {
  { echo "{"; sort -u "$STATE.tmp" | sed '$ s/,$//'; echo "}"; } > "$STATE"
  rm -f "$STATE.tmp"
  exit "${1:-0}"
}

echo "# SpineScoutX research run timeline ($(ts))" > "$TIMELINE"
echo "" >> "$TIMELINE"
echo "| phase | status | duration | marker |" >> "$TIMELINE"
echo "|---|---|---|---|" >> "$TIMELINE"

# Fail fast on missing prerequisites.
if [[ ! -f "$RSNA_ROOT/train_label_coordinates.csv" ]]; then
  say "ERROR: RSNA data missing under $RSNA_ROOT (see docs/data_status.md)"; finalize 2
fi
if [[ ! -f "$SEG_RUN/best.pt" ]]; then
  say "ERROR: trained SPIDER segmenter missing at $SEG_RUN/best.pt (train E4 first)"; finalize 2
fi

PREP_ARGS=(--rsna-root "$RSNA_ROOT" --out "$RSNA_CACHE")
[[ -n "$RSNA_LIMIT" ]] && PREP_ARGS+=(--limit-studies "$RSNA_LIMIT")

# --- Profiles: which steps run. Usage: PROFILE=anatomy-forced bash <script>, or arg1.
PROFILE="${PROFILE:-${1:-full}}"
declare -A IN_PROFILE
case "$PROFILE" in
  baseline)        STEPS="doctor prepare_rsna train_e0 eval_e0" ;;
  anatomy-forced)  STEPS="doctor prepare_rsna train_e0 eval_e0 anatomy_priors train_e2 eval_e2 ablate_e2" ;;
  multiview)       STEPS="doctor prepare_rsna train_e0 eval_e0 anatomy_priors train_e2 eval_e2 ablate_e2" ;;  # E3 multi-view = future
  full)            STEPS="doctor prepare_rsna train_e0 eval_e0 anatomy_priors train_e1 eval_e1 ablate train_e2 eval_e2 ablate_e2" ;;
  *) say "ERROR: unknown profile '$PROFILE' (baseline|anatomy-forced|multiview|full)"; finalize 2 ;;
esac
for s in $STEPS; do IN_PROFILE[$s]=1; done
say "profile=$PROFILE steps=[$STEPS]"

run_step doctor          ""                                          spinescoutx doctor --data
run_step prepare_rsna    "$RSNA_CACHE/manifest.parquet"              spinescoutx prepare-rsna "${PREP_ARGS[@]}"
run_step train_e0        "runs/e0_baseline_real/best.pt"             spinescoutx train-classifier --config configs/real_e0_baseline_rsna.yaml
run_step eval_e0         "runs/e0_baseline_real/predictions.json"    spinescoutx evaluate --run runs/e0_baseline_real
run_step anatomy_priors  "$PRIOR_CACHE/anatomy_prior_manifest.csv"   spinescoutx prepare-anatomy-priors --rsna-cache "$RSNA_CACHE" --segmenter-run "$SEG_RUN" --out "$PRIOR_CACHE"
run_step train_e1        "runs/e1_anatomy_guided_real/best.pt"       spinescoutx train-anatomy-guided --config configs/real_e1_anatomy_guided.yaml
run_step eval_e1         "runs/e1_anatomy_guided_real/predictions.json" spinescoutx evaluate --run runs/e1_anatomy_guided_real
run_step ablate          "runs/e3_ablation/ablation.json"            spinescoutx ablate --config configs/ablation.yaml
run_step train_e2        "runs/e2_anatomy_forced_roi_real/best.pt"   spinescoutx train-anatomy-forced --config configs/real_e2_anatomy_forced_roi.yaml
run_step eval_e2         "runs/e2_anatomy_forced_roi_real/predictions.json" spinescoutx evaluate --run runs/e2_anatomy_forced_roi_real
run_step ablate_e2       "runs/e2_forced_ablation/ablation.json"     spinescoutx ablate --config configs/ablation_e2_forced.yaml

say "Profile '$PROFILE' complete. See $TIMELINE and $STATE."
say "Reports: spinescoutx report-assistant --study-id <ID> --run runs/e2_anatomy_forced_roi_real --baseline-run runs/e0_baseline_real"
finalize 0
