#!/usr/bin/env bash
# Submit BirdSet core eval for Qwen3-Omni (esp_data) using an existing launcher URL file.
#
# Usage (from repo root on Slurm login):
#   # Reuse the same Qwen serve job as beans_zero_core (check ~/beans-next-launchers/*.url):
#   BEANS_NEXT_URL_FILE=$HOME/beans-next-launchers/57718.url \
#     bash examples/slurm/submit_birdset_core_qwen3_omni_esp_data.sh
#
# Or submit a new serve job first:
#   SERVE=$(sbatch --parsable examples/slurm/serve_qwen3_omni.sh)
#   BEANS_NEXT_URL_FILE=$HOME/beans-next-launchers/$SERVE.url \
#     bash examples/slurm/submit_birdset_core_qwen3_omni_esp_data.sh
#
# Uses `after:` (not `afterok`) so inference can run while the long-lived serve job stays up.

set -euo pipefail

URL_FILE="${BEANS_NEXT_URL_FILE:-${BEANS_PRO_URL_FILE:-}}"
if [[ -z "$URL_FILE" ]]; then
  echo "ERROR: BEANS_NEXT_URL_FILE must be set to the Qwen launcher .url file." >&2
  echo "Compat: BEANS_PRO_URL_FILE is also supported." >&2
  exit 2
fi
CONFIG_PATH="${BEANS_NEXT_CONFIG:-${BEANS_PRO_CONFIG:-configs/benchmarks/birdset_core_qwen3_omni_esp_data.yaml}}"
INC="${BEANS_NEXT_INCREMENT:-${BEANS_PRO_INCREMENT:-i30}}"
RUN_ID="${BEANS_NEXT_RUN_ID:-${BEANS_PRO_RUN_ID:-full_qwen3_omni_instruct_2_h100_birdset_core_r1}}"
OUT_DIR="${BEANS_NEXT_OUT_DIR:-${BEANS_PRO_OUT_DIR:-/scratch/${USER}/.cache/beans-next-results/${INC}/qwen3_omni_instruct_2/birdset_core/${RUN_ID}}}"

SERVE_JOB="${BEANS_NEXT_SERVE_JOB_ID:-${BEANS_PRO_SERVE_JOB_ID:-}}"
if [[ -z "$SERVE_JOB" ]]; then
  SERVE_JOB="$(basename "$URL_FILE" .url)"
fi

echo "URL file: $URL_FILE"
echo "Config: $CONFIG_PATH"
echo "RUN_ID: $RUN_ID"
echo "OUT_DIR: $OUT_DIR"
echo "Infer depends on serve job: $SERVE_JOB (after: — serve may stay running)"

INF="$(
  BEANS_NEXT_URL_FILE="$URL_FILE" \
    BEANS_NEXT_DATA_SOURCE=esp_data \
    BEANS_NEXT_CONFIG="$CONFIG_PATH" \
    BEANS_NEXT_RUN_ID="$RUN_ID" \
    BEANS_NEXT_OUT_DIR="$OUT_DIR" \
    BEANS_NEXT_INCREMENT="$INC" \
    sbatch --parsable --partition=cpu --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)"
echo "Inference job: $INF  (log: ~/logs/$INF.log)"
