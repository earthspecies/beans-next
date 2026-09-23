#!/usr/bin/env bash
# Create the isolated vLLM runtime used to serve NVIDIA Audex.
#
# Audex ships its own vLLM architecture plugin
# (inference_scripts_vllm/audioqa_scripts/, package audex_30b_a3b_vllm) that
# must be editable-installed into this venv; plain `vllm serve` will not find
# the architecture. The plugin targets vLLM 0.20.0 and this pins the closest
# already-validated build (0.20.2, matching beans-next-vllm-runtime). Give
# Audex its own venv rather than sharing the text-eval one, in case the
# plugin needs a version downgrade later.
#
# Run in two passes:
#   1. sbatch setup_audex_runtime.sbatch                       # base vLLM runtime
#   2. sbatch download_audex.sbatch                             # snapshot download
#   3. BEANS_NEXT_AUDEX_SNAPSHOT_PATH=<snapshot_path> \
#        sbatch setup_audex_runtime.sbatch                      # editable-install plugin

#SBATCH --job-name=beans-audex-setup
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
readonly SERVER_RUNTIME="${BEANS_NEXT_AUDEX_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-audex-runtime}"
readonly SERVER_VENV="$SERVER_RUNTIME/.venv"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
readonly VLLM_VERSION="${BEANS_NEXT_VLLM_VERSION:-0.20.2}"
readonly VLLM_WHEEL_URL="${BEANS_NEXT_VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl}"
readonly AUDEX_SNAPSHOT_PATH="${BEANS_NEXT_AUDEX_SNAPSHOT_PATH:-}"

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
  --extra-index-url=https://download.pytorch.org/whl/cu129 \
  --index-strategy=unsafe-best-match \
  --only-binary=:all: \
  --compile-bytecode \
  "$VLLM_WHEEL_URL" "vllm[audio]==${VLLM_VERSION}"

if [[ -n "$AUDEX_SNAPSHOT_PATH" ]]; then
  if [[ ! -d "$AUDEX_SNAPSHOT_PATH/inference_scripts_vllm/audioqa_scripts" ]]; then
    echo "Missing plugin source under $AUDEX_SNAPSHOT_PATH/inference_scripts_vllm/audioqa_scripts" >&2
    exit 1
  fi
  uv pip install \
    --python "$SERVER_VENV/bin/python" \
    --editable "$AUDEX_SNAPSHOT_PATH/inference_scripts_vllm/audioqa_scripts" \
    --no-deps --no-build-isolation

  "$SERVER_VENV/bin/python" -c \
    'import audex_30b_a3b_vllm; print("plugin import OK:", audex_30b_a3b_vllm.__file__)'
fi

uv pip freeze --python "$SERVER_VENV/bin/python" \
  > "$SERVER_RUNTIME/requirements.freeze.txt"

"$SERVER_VENV/bin/python" -c \
  'import torch, vllm; print(f"torch={torch.__version__} vllm={vllm.__version__}")'
