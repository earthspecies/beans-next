#!/usr/bin/env bash
# Submit a full beans_zero_core evaluation run for NatureLM-audio v1.1 (esp_data backend).
#
# Usage (from repo root):
#   HF_TOKEN=hf_... bash examples/slurm/submit_beans_zero_core_naturelm_v1_1_esp_data.sh
#
# Optional overrides:
#   BEANS_PRO_OUT_DIR=/my/path HF_TOKEN=hf_... bash examples/slurm/submit_beans_zero_core_naturelm_v1_1_esp_data.sh
#
# Prereqs:
#   - Gated HuggingFace access to EarthSpeciesProject/naturelm-audio-1.1.00-private
#   - HF_TOKEN set in env, ~/.config/huggingface/hf_token, or HUGGINGFACE_HUB_TOKEN
#   - Serve script works on your cluster (examples/slurm/serve_naturelm_v1_1.sh)

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.config/huggingface/hf_token" ]]; then
  export HF_TOKEN="$(< "$HOME/.config/huggingface/hf_token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required for NatureLM v1.1 (gated weights)." >&2
  echo "Set HF_TOKEN or store it in ~/.config/huggingface/hf_token" >&2
  exit 1
fi
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Critical sweep policy: use the non-multiaudio checkpoint.
export NATURELM_GCS_CHECKPOINT_URI="${NATURELM_GCS_CHECKPOINT_URI:-gs://foundation-models/naturelm-audio-1.1/base_model/1290000}"

INC="${BEANS_PRO_INC:-adhoc}"
MODEL_DIR="naturelm_v1_1"
SUBSET_DIR="beans_zero_core"
TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_RUN_ID="smoke_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FULL_RUN_ID="full_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
SMOKE_OUT_DIR="${BEANS_PRO_OUT_DIR_SMOKE:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${SMOKE_RUN_ID}}"
FULL_OUT_DIR="${BEANS_PRO_OUT_DIR_FULL:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${FULL_RUN_ID}}"
CONFIG_PATH="configs/benchmarks/beans_zero_core_naturelm_v1_1_esp_data.yaml"

echo "Submitting serving job..."
SERVE_JOB=$(
  HF_TOKEN="$HF_TOKEN" \
  HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  NATURELM_GCS_CHECKPOINT_URI="$NATURELM_GCS_CHECKPOINT_URI" \
  sbatch --parsable examples/slurm/serve_naturelm_v1_1.sh
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

