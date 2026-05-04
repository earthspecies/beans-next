#!/usr/bin/env bash
# Submit a full beans_zero_core evaluation run for Qwen3-Omni-7B via vLLM (esp_data backend).
#
# Usage (from repo root):
#   bash examples/slurm/submit_beans_zero_core_qwen3_omni_esp_data.sh
#
# Optional overrides:
#   VLLM_MODEL_ID=Qwen/Qwen3-Omni-72B \
#   VLLM_TENSOR_PARALLEL_SIZE=4 \
#   bash examples/slurm/submit_beans_zero_core_qwen3_omni_esp_data.sh
#
# Prereqs:
#   - Serve script works on your cluster (examples/slurm/serve_qwen3_omni.sh)
#   - vLLM installed on GPU nodes

set -euo pipefail

MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-Omni-7B}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
INC="${BEANS_PRO_INC:-adhoc}"
MODEL_DIR="qwen3_omni"
SUBSET_DIR="beans_zero_core"
TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_RUN_ID="smoke_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FULL_RUN_ID="full_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
SMOKE_OUT_DIR="${BEANS_PRO_OUT_DIR_SMOKE:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${SMOKE_RUN_ID}}"
FULL_OUT_DIR="${BEANS_PRO_OUT_DIR_FULL:-/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${FULL_RUN_ID}}"
CONFIG_PATH="configs/benchmarks/beans_zero_core_qwen3_omni_esp_data.yaml"

echo "Model: $MODEL_ID (tensor_parallel_size=$TP)"
echo "Submitting serving job..."

if [[ "$TP" -gt 1 ]]; then
  SERVE_JOB=$(
    VLLM_MODEL_ID="$MODEL_ID" \
    VLLM_TENSOR_PARALLEL_SIZE="$TP" \
    sbatch --parsable --gpus="$TP" examples/slurm/serve_qwen3_omni.sh
  )
else
  SERVE_JOB=$(
    VLLM_MODEL_ID="$MODEL_ID" \
    sbatch --parsable examples/slurm/serve_qwen3_omni.sh
  )
fi
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

