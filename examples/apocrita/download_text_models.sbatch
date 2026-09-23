#!/usr/bin/env bash
# Download public text-model snapshots after setup_text_eval_runtime.sbatch.

#SBATCH --job-name=beans-text-models
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly SERVER_RUNTIME="${BEANS_NEXT_VLLM_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-vllm-runtime}"
readonly SERVER_PYTHON="$SERVER_RUNTIME/.venv/bin/python"
readonly MODEL_CACHE="${BEANS_NEXT_MODEL_CACHE:-$SCRATCH_ROOT/beans-next-models}"
readonly MANIFEST="${BEANS_NEXT_MODEL_MANIFEST:-$MODEL_CACHE/models-manifest.${SLURM_JOB_ID}.json}"

if [[ ! -x "$SERVER_PYTHON" ]]; then
  echo "Missing vLLM runtime: $SERVER_PYTHON" >&2
  exit 1
fi

mkdir -p "$MODEL_CACHE" "$SERVER_RUNTIME/cache/huggingface" "$SERVER_RUNTIME/tmp"

export HF_HOME="$SERVER_RUNTIME/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"
export TMPDIR="$SERVER_RUNTIME/tmp"

models=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
if [[ "${BEANS_NEXT_INCLUDE_LLAMA:-0}" == "1" ]]; then
  models+=("meta-llama/Llama-3.1-8B-Instruct")
fi

args=()
for model in "${models[@]}"; do
  args+=(--model "$model")
done

"$SERVER_PYTHON" "$REPO/examples/apocrita/download_text_models.py" \
  "${args[@]}" \
  --cache-dir "$MODEL_CACHE" \
  --manifest "$MANIFEST" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "model_manifest=$MANIFEST"
