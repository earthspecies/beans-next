#!/usr/bin/env bash
# Create the isolated vLLM runtime used to serve google/gemma-4-12B-it.
#
# Gemma 4 12B (encoder-free Unified) needs vLLM >= 0.23.0 with the `audio`
# extra (PR #44429). Gets its own venv at $SCRATCH/beans-next-gemma4-runtime,
# separate from beans-next-audex-runtime (vLLM 0.20.2 + the Audex plugin) and
# the shared text-eval runtime — do not touch either.
#
# Run in two passes, mirroring setup_audex_runtime.sbatch's shape (there,
# pass 2 needed the downloaded snapshot to editable-install a vendor plugin;
# here, pass 2 needs the downloaded snapshot only to probe AutoProcessor):
#   1. sbatch setup_gemma4_runtime.sbatch                        # base vLLM runtime + fast checks
#   2. sbatch download_gemma4.sbatch                             # snapshot download (may run before or after pass 1)
#   3. BEANS_NEXT_GEMMA4_SNAPSHOT_PATH=<snapshot_path> \
#        sbatch setup_gemma4_runtime.sbatch                      # AutoProcessor/feature_extractor probe
#
# This is also where the Decision-1 serving-backend risk is meant to be
# resolved cheaply, in a CPU job:
#   - The vLLM recipe page for gemma-4-12B-it claims encoder-free Unified
#     "has not yet shipped in a stable release" and points at a nightly
#     wheel, which conflicts with the v0.23.0 release notes. If
#     `Gemma4UnifiedForConditionalGeneration` is absent from the installed
#     wheel's architecture registry (checked below), do not chase this on a
#     GPU: set BEANS_NEXT_VLLM_VERSION / BEANS_NEXT_VLLM_WHEEL_URL to a
#     pinned/nightly wheel and re-run pass 1.
#   - A missing `feature_extractor` on AutoProcessor is the documented cause
#     of silently-ignored audio (fluent, meaningless text with no error) —
#     checked in pass 2 below as a real assertion, not a placeholder.

#SBATCH --job-name=beans-gemma4-setup
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
readonly SERVER_RUNTIME="${BEANS_NEXT_GEMMA4_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-gemma4-runtime}"
readonly SERVER_VENV="$SERVER_RUNTIME/.venv"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
# Stable wheel target per Decision-1's serving-backend risk. Override both
# vars together to substitute a pinned/nightly wheel if the assertions below
# fail against this version.
readonly VLLM_VERSION="${BEANS_NEXT_VLLM_VERSION:-0.23.0}"
readonly VLLM_WHEEL_URL="${BEANS_NEXT_VLLM_WHEEL_URL:-}"
readonly GEMMA4_SNAPSHOT_PATH="${BEANS_NEXT_GEMMA4_SNAPSHOT_PATH:-}"

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

# Pass 1: base vLLM runtime + fallback backend deps. Unlike Audex, gemma4's
# vLLM support ships in an ordinary PyPI release (no vendor wheel URL
# required) unless the stable wheel turns out to lack encoder-free Unified
# support, in which case set BEANS_NEXT_VLLM_WHEEL_URL to a pinned/nightly
# wheel and re-run this pass.
if [[ -n "$VLLM_WHEEL_URL" ]]; then
  uv pip install \
    --python "$SERVER_VENV/bin/python" \
    --torch-backend=cu129 \
    --extra-index-url=https://download.pytorch.org/whl/cu129 \
    --index-strategy=unsafe-best-match \
    --only-binary=:all: \
    --compile-bytecode \
    "$VLLM_WHEEL_URL" "vllm[audio]==${VLLM_VERSION}"
else
  uv pip install \
    --python "$SERVER_VENV/bin/python" \
    --torch-backend=cu129 \
    --extra-index-url=https://download.pytorch.org/whl/cu129 \
    --index-strategy=unsafe-best-match \
    --compile-bytecode \
    "vllm[audio]>=${VLLM_VERSION}"
fi

# Fallback backend (GEMMA4_BACKEND=transformers) support, installed in the
# same venv so a failed Decision-1 probe needs no second setup job.
uv pip install \
  --python "$SERVER_VENV/bin/python" \
  --compile-bytecode \
  "transformers" "accelerate" "soundfile" "librosa"

uv pip freeze --python "$SERVER_VENV/bin/python" \
  > "$SERVER_RUNTIME/requirements.freeze.txt"

"$SERVER_VENV/bin/python" -c \
  'import torch, vllm; print(f"torch={torch.__version__} vllm={vllm.__version__}")'

"$SERVER_VENV/bin/python" -c '
import sys
import vllm

try:
    from vllm.model_executor.models.registry import ModelRegistry
except ImportError as exc:
    print(f"FATAL: could not import vLLM model registry: {exc}", file=sys.stderr)
    raise SystemExit(1)

target = "Gemma4UnifiedForConditionalGeneration"
archs = set()
get_archs = getattr(ModelRegistry, "get_supported_archs", None)
if callable(get_archs):
    archs = set(get_archs())
else:
    models_attr = getattr(ModelRegistry, "models", None)
    if models_attr is not None:
        archs = set(models_attr.keys()) if hasattr(models_attr, "keys") else set(models_attr)

vllm_version = vllm.__version__
if target not in archs:
    print(
        f"FATAL: {target} not found in vLLM architecture registry "
        f"(vllm=={vllm_version}). The installed vllm[audio] wheel does not "
        "support encoder-free Gemma 4 Unified. Per Decision 1, set "
        "BEANS_NEXT_VLLM_VERSION/BEANS_NEXT_VLLM_WHEEL_URL to a pinned/nightly "
        "wheel and re-run this job before spending any GPU time.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"vLLM architecture registry OK: {target} present")
'

"$SERVER_VENV/bin/python" -c \
  'import soundfile, librosa; print(f"soundfile={soundfile.__version__} librosa={librosa.__version__}")'

# Pass 2 (requires the downloaded snapshot): AutoProcessor for the
# gemma-4-12B-it snapshot must expose a `feature_extractor` — its absence is
# the documented cause of silently-ignored audio (fluent, well-formed,
# entirely meaningless text with no error surfaced). This is a correctness
# gate, not a smoke check: fail loudly, do not proceed to a GPU job.
if [[ -n "$GEMMA4_SNAPSHOT_PATH" ]]; then
  if [[ ! -f "$GEMMA4_SNAPSHOT_PATH/config.json" ]]; then
    echo "Missing snapshot config.json under $GEMMA4_SNAPSHOT_PATH" >&2
    exit 1
  fi
  export HF_HOME="$SERVER_RUNTIME/cache/huggingface"
  "$SERVER_VENV/bin/python" -c "
import sys
from transformers import AutoProcessor

snapshot_path = '$GEMMA4_SNAPSHOT_PATH'
processor = AutoProcessor.from_pretrained(snapshot_path, trust_remote_code=True)
feature_extractor = getattr(processor, 'feature_extractor', None)
if feature_extractor is None:
    print(
        'FATAL: AutoProcessor for ' + snapshot_path + ' has no feature_extractor. '
        'This is the documented cause of silently-ignored audio: requests would '
        'be accepted and would return fluent text, but the audio would never '
        'reach the model. Do not proceed to the audio-sensitivity probe or any '
        'GPU job until this is resolved (e.g. a processor_config.json fix or a '
        'different snapshot revision).',
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f'AutoProcessor.feature_extractor OK: {type(feature_extractor).__name__}')
"
else
  echo "BEANS_NEXT_GEMMA4_SNAPSHOT_PATH not set; skipping AutoProcessor/feature_extractor probe (pass 2)." >&2
  echo "Re-run with BEANS_NEXT_GEMMA4_SNAPSHOT_PATH=<snapshot_path> once download_gemma4.sbatch has completed." >&2
fi
