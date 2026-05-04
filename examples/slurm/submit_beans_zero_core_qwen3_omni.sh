#!/usr/bin/env bash
# Submit a full beans_zero_core evaluation run for Qwen3-Omni-7B via vLLM.
#
# Usage (from repo root):
#   bash examples/slurm/submit_beans_zero_core_qwen3_omni.sh
#
# Optional overrides:
#   VLLM_MODEL_ID=Qwen/Qwen3-Omni-72B \
#   VLLM_TENSOR_PARALLEL_SIZE=4 \
#   bash examples/slurm/submit_beans_zero_core_qwen3_omni.sh
#
# For multi-GPU (72B model), also edit serve_qwen3_omni.sh to set --gpus=4 or pass via:
#   sbatch --gpus=4 examples/slurm/serve_qwen3_omni.sh   (handled below if VLLM_TENSOR_PARALLEL_SIZE>1)
#
# Prereqs:
#   - Weights downloaded: snapshot_download Qwen/Qwen3-Omni-7B
#   - vLLM installed on GPU nodes

set -euo pipefail

MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-Omni-7B}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
RUN_TAG="qwen3_omni_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_PRO_OUT_DIR:-${SCRATCH:-$HOME}/beans-next-results/$RUN_TAG}"

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

echo "Submitting inference job (depends on serve job $SERVE_JOB)..."
INFER_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_SUITE=beans_zero_core \
  BEANS_PRO_RUN_ID="$RUN_TAG" \
  BEANS_PRO_OUT_DIR="$OUT_DIR" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)
echo "  Inference job: $INFER_JOB"
echo "  Log: ~/logs/$INFER_JOB.log"
echo "  Output: $OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel both: scancel $SERVE_JOB $INFER_JOB"
