#!/usr/bin/env bash
# Submit AF3 GPU serve + CPU inference for beans_zero_zf_indiv only (Slurm).
#
# Usage (from repo root on the cluster):
#   bash examples/slurm/submit_beans_zero_zf_indiv_af3.sh
#
# Flow:
#   1) serve_af3.sh — GPU node, writes ~/beans-next-launchers/<jobid>.url
#   2) run_inference.sh — smoke (limit 5) on same config after serve starts
#   3) run_inference.sh — full zf_indiv after smoke succeeds
#
# Env overrides (optional):
#   BEANS_PRO_INC              increment tag for scratch paths (default: adhoc)
#   BEANS_PRO_SMOKE_LIMIT      smoke example cap (default: 5)
#   BEANS_PRO_OUT_DIR_SMOKE    scratch output for smoke run
#   BEANS_PRO_OUT_DIR_FULL     scratch output for full run
#
# License: NVIDIA OneWay Noncommercial License — non-commercial research only.

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.config/huggingface/hf_token" ]]; then
  export HF_TOKEN="$(< "$HOME/.config/huggingface/hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

INC="${BEANS_PRO_INC:-adhoc}"
MODEL_DIR="af3"
SUBSET_DIR="beans_zero_zf_indiv"
TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_RUN_ID="smoke_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FULL_RUN_ID="full_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
SMOKE_OUT_DIR="${BEANS_PRO_OUT_DIR_SMOKE:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${SMOKE_RUN_ID}}"
FULL_OUT_DIR="${BEANS_PRO_OUT_DIR_FULL:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${FULL_RUN_ID}}"
CONFIG_PATH="configs/benchmarks/beans_zero_zf_indiv_af3_esp_data.yaml"

echo "Submitting AF3 serving job (GPU)..."
SERVE_JOB=$(
  HF_TOKEN="${HF_TOKEN:-}" \
  HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  sbatch --parsable examples/slurm/serve_af3.sh
)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"
echo "  URL file: ~/beans-next-launchers/${SERVE_JOB}.url"

echo "Submitting zf_indiv smoke inference (depends on serve $SERVE_JOB)..."
SMOKE_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_DATA_SOURCE="esp_data" \
  BEANS_PRO_CONFIG="$CONFIG_PATH" \
  BEANS_PRO_LIMIT="${BEANS_PRO_SMOKE_LIMIT:-5}" \
  BEANS_PRO_RUN_KIND="smoke" \
  BEANS_PRO_RUN_ID="$SMOKE_RUN_ID" \
  BEANS_PRO_OUT_DIR="$SMOKE_OUT_DIR" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)
echo "  Smoke job: $SMOKE_JOB"
echo "  Log: ~/logs/$SMOKE_JOB.log"
echo "  Output: $SMOKE_OUT_DIR"

echo "Submitting zf_indiv full inference (afterok smoke $SMOKE_JOB)..."
FULL_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_DATA_SOURCE="esp_data" \
  BEANS_PRO_CONFIG="$CONFIG_PATH" \
  BEANS_PRO_RUN_KIND="full" \
  BEANS_PRO_RUN_ID="$FULL_RUN_ID" \
  BEANS_PRO_OUT_DIR="$FULL_OUT_DIR" \
  sbatch --parsable --dependency=afterok:"$SMOKE_JOB" examples/slurm/run_inference.sh
)
echo "  Full job: $FULL_JOB"
echo "  Log: ~/logs/$FULL_JOB.log"
echo "  Output: $FULL_OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel: scancel $SERVE_JOB $SMOKE_JOB $FULL_JOB"
