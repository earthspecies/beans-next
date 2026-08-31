#!/usr/bin/env bash
# Create the isolated vLLM runtime used by text-only BEANS evaluations.

#SBATCH --job-name=beans-text-setup
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly RUNNER_RUNTIME="${BEANS_NEXT_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-runtime}"
readonly SERVER_RUNTIME="${BEANS_NEXT_VLLM_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-vllm-runtime}"
readonly SERVER_VENV="$SERVER_RUNTIME/.venv"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
readonly VLLM_VERSION="${BEANS_NEXT_VLLM_VERSION:-0.20.2}"

if [[ ! -x "$RUNNER_PYTHON" ]]; then
  echo "Missing runner Python: $RUNNER_PYTHON" >&2
  exit 1
fi

mkdir -p "$SERVER_RUNTIME/cache/uv" "$SERVER_RUNTIME/cache/huggingface" \
  "$SERVER_RUNTIME/tmp"

module use /data/home/acw777/privatemodules
module load uv/latest

export UV_CACHE_DIR="$SERVER_RUNTIME/cache/uv"
export TMPDIR="$SERVER_RUNTIME/tmp"

if [[ ! -x "$SERVER_VENV/bin/python" ]]; then
  uv venv --python "$RUNNER_PYTHON" "$SERVER_VENV"
fi

uv pip install \
  --python "$SERVER_VENV/bin/python" \
  --torch-backend=cu129 \
  --only-binary=:all: \
  --compile-bytecode \
  "vllm==$VLLM_VERSION"

uv pip freeze --python "$SERVER_VENV/bin/python" \
  > "$SERVER_RUNTIME/requirements.freeze.txt"

"$SERVER_VENV/bin/python" -c \
  'import torch, vllm; print(f"torch={torch.__version__} vllm={vllm.__version__}")'
