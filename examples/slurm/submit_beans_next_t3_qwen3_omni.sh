#!/usr/bin/env bash
# Submit BEANS-Next Tier-3 sampling run (20 examples/task) for Qwen3-Omni 30B (vLLM).
#
# Usage (from repo root):
#   bash examples/slurm/submit_beans_next_t3_qwen3_omni.sh
#
# Model: Qwen/Qwen3-Omni-30B-A3B-Instruct via vLLM-Omni + BEANS-Next adapter sidecar.
# Uses the known-good "lean" single-stage config + cached weights in
#   ~/hf_cache_qwen3_omni_instruct_lean
# Override with:
#   VLLM_MODEL_ID=<other> bash examples/slurm/submit_beans_next_t3_qwen3_omni.sh

set -euo pipefail

INC="t3_sampling"
MODEL_DIR="qwen3_omni"
SUBSET_DIR="beans_next_tier_3_hf"
TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_RUN_ID="smoke_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FULL_RUN_ID="full_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
HOME_RESULTS="${BEANS_NEXT_RESULTS_HOME_DIR:-$HOME/beans-next-results/ingested}"
SMOKE_OUT_DIR="/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${SMOKE_RUN_ID}"
FULL_OUT_DIR="/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${FULL_RUN_ID}"

# Lean 30B single-stage config (known-good; weights pre-cached in hf_cache_qwen3_omni_instruct_lean).
BEANS_NEXT_STAGE_CONFIG="${BEANS_NEXT_STAGE_CONFIG:-/home/${USER}/code/beans-next/examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml}"
HF_CACHE="${BEANS_NEXT_HF_HOME:-/home/${USER}/hf_cache_qwen3_omni_instruct_lean}"

echo "Submitting serving job (Qwen3-Omni-30B-A3B-Instruct via vLLM-Omni, h100-80)..."
SERVE_JOB=$(
  VLLM_MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}" \
  VLLM_MAX_BATCH_SIZE="1" \
  VLLM_TENSOR_PARALLEL_SIZE="1" \
  VLLM_OMNI_INSTALL="1" \
  VLLM_INSTALL_VERSION="0.18.0" \
  VLLM_OMNI_VERSION="0.18.0" \
  VLLM_OMNI="1" \
  VLLM_MAX_MODEL_LEN="16384" \
  VLLM_AUDIO_CONTENT_FORMAT="audio_url_data" \
  HF_HUB_DISABLE_XET="1" \
  BEANS_NEXT_HF_HOME="$HF_CACHE" \
  VLLM_EXTRA_ARGS="--stage-configs-path ${BEANS_NEXT_STAGE_CONFIG} --download-dir ${HF_CACHE}" \
  BEANS_NEXT_DEBUG=1 \
  sbatch --parsable \
    --partition=h100-80 --gpus=1 --exclude=slurm-8x-h100-1 \
    examples/slurm/serve_vllm.sh
)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

echo "Submitting smoke inference job (limit 5, depends on serve job $SERVE_JOB)..."
SMOKE_JOB=$(
  BEANS_NEXT_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_NEXT_DATA_SOURCE="esp_data" \
  BEANS_NEXT_SUITE="$SUBSET_DIR" \
  BEANS_NEXT_LIMIT=5 \
  BEANS_NEXT_RUN_KIND="smoke" \
  BEANS_NEXT_RUN_ID="$SMOKE_RUN_ID" \
  BEANS_NEXT_OUT_DIR="$SMOKE_OUT_DIR" \
  BEANS_NEXT_RESULTS_HOME_DIR="$HOME_RESULTS" \
  BEANS_NEXT_INC="$INC" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)
echo "  Smoke job: $SMOKE_JOB"
echo "  Log: ~/logs/$SMOKE_JOB.log"
echo "  Output: $SMOKE_OUT_DIR"

echo "Submitting full inference job (limit 20/task, afterok smoke $SMOKE_JOB)..."
FULL_JOB=$(
  BEANS_NEXT_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_NEXT_DATA_SOURCE="esp_data" \
  BEANS_NEXT_SUITE="$SUBSET_DIR" \
  BEANS_NEXT_LIMIT=20 \
  BEANS_NEXT_RUN_KIND="full" \
  BEANS_NEXT_RUN_ID="$FULL_RUN_ID" \
  BEANS_NEXT_OUT_DIR="$FULL_OUT_DIR" \
  BEANS_NEXT_RESULTS_HOME_DIR="$HOME_RESULTS" \
  BEANS_NEXT_INC="$INC" \
  sbatch --parsable --dependency=afterok:"$SMOKE_JOB" examples/slurm/run_inference.sh
)
echo "  Full job: $FULL_JOB"
echo "  Log: ~/logs/$FULL_JOB.log"
echo "  Output: $FULL_OUT_DIR"
echo "  NFS view: /mnt${HOME_RESULTS}/${FULL_RUN_ID}"

echo "Submitting serve-cancel job (afterany full $FULL_JOB)..."
CANCEL_JOB=$(
  sbatch --parsable --dependency=afterany:"$FULL_JOB" \
    --partition=h100-80 --gpus=0 --cpus-per-task=1 \
    --output="/home/%u/logs/%A-cancel.log" \
    --job-name="cancel-serve-${SERVE_JOB}" \
    --wrap="scancel ${SERVE_JOB}; echo 'Cancelled serve job ${SERVE_JOB}'"
)
echo "  Cancel job: $CANCEL_JOB"
echo ""
echo "Monitor: squeue --me"
echo "Cancel all: scancel $SERVE_JOB $SMOKE_JOB $FULL_JOB $CANCEL_JOB"
