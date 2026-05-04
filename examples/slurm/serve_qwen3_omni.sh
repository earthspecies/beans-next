#!/usr/bin/env bash
# SLURM job: start Qwen3-Omni via vLLM + BEANS-Next adapter sidecar.
#
# This is a model-specific wrapper around `examples/slurm/serve_vllm.sh` that
# sets a safe default `VLLM_MODEL_ID` for Qwen3-Omni.
#
# Submit:
#   sbatch examples/slurm/serve_qwen3_omni.sh
#
# Optional overrides:
#   VLLM_TENSOR_PARALLEL_SIZE=2 sbatch --gpus=2 examples/slurm/serve_qwen3_omni.sh
#   BEANS_PRO_PORT=8003 VLLM_PORT=8103 sbatch examples/slurm/serve_qwen3_omni.sh

set -euo pipefail

export VLLM_MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-Omni-7B}"

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

exec "${REPO}/examples/slurm/serve_vllm.sh"

