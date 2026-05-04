#!/usr/bin/env bash
# Submit a full beans_zero_core evaluation run for Gemini via the OpenAI-compatible proxy
# (esp_data backend).
#
# Usage (from repo root):
#   bash examples/slurm/submit_beans_zero_core_gemini_esp_data.sh
#
# Optional overrides:
#   OPENAI_MODEL=gemini-2.5-flash bash examples/slurm/submit_beans_zero_core_gemini_esp_data.sh
#
# Prereqs:
#   - API key available to the serving job (see serve_openai_proxy_gemini.sh)

set -euo pipefail

INC="${BEANS_PRO_INC:-adhoc}"
MODEL_DIR="gemini"
SUBSET_DIR="beans_zero_core"
TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_RUN_ID="smoke_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FULL_RUN_ID="full_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
SMOKE_OUT_DIR="${BEANS_PRO_OUT_DIR_SMOKE:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${SMOKE_RUN_ID}}"
FULL_OUT_DIR="${BEANS_PRO_OUT_DIR_FULL:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${FULL_RUN_ID}}"
CONFIG_PATH="configs/benchmarks/beans_zero_core_gemini_esp_data.yaml"

echo "Submitting serving job..."
SERVE_JOB=$(
  OPENAI_PROXY_STUB="${OPENAI_PROXY_STUB:-0}" \
  sbatch --parsable examples/slurm/serve_openai_proxy_gemini.sh
)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

echo "Submitting smoke inference job (depends on serve job $SERVE_JOB)..."
SMOKE_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_DATA_SOURCE="esp_data" \
  BEANS_PRO_CONFIG="$CONFIG_PATH" \
  BEANS_PRO_LIMIT="${BEANS_PRO_SMOKE_LIMIT:-5}" \
  BEANS_PRO_RUN_ID="$SMOKE_RUN_ID" \
  BEANS_PRO_OUT_DIR="$SMOKE_OUT_DIR" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/test_run_inference.sh
)
echo "  Smoke job: $SMOKE_JOB"
echo "  Log: ~/logs/$SMOKE_JOB.log"
echo "  Output: $SMOKE_OUT_DIR"

echo "Submitting full inference job (afterok smoke $SMOKE_JOB)..."
FULL_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_DATA_SOURCE="esp_data" \
  BEANS_PRO_CONFIG="$CONFIG_PATH" \
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

