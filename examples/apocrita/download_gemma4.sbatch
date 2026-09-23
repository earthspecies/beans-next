#!/usr/bin/env bash
# Download the google/gemma-4-12B-it snapshot into its own model cache.
# Not gated, Apache 2.0, ~24 GB bf16 safetensors (no checkpoint variants to
# skip, unlike Audex).
#
# Deviation from the Audex download job: Audex's download ran against its own
# dedicated vLLM runtime venv, which already existed by that point because
# setup_audex_runtime.sbatch's first pass runs before download_audex.sbatch.
# The plan runs this download *before* setup_gemma4_runtime.sbatch (there is
# no plugin to editable-install here, so there is no ordering requirement the
# other way), so this job instead uses the base beans-next runner venv
# ($BEANS_NEXT_RUNTIME_ROOT/.venv), which already has huggingface_hub as a
# core dependency and does not depend on the gemma4 runtime existing yet.

#SBATCH --job-name=beans-gemma4-download
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next-audex repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly RUNNER_RUNTIME="${BEANS_NEXT_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-runtime}"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
readonly MODEL_CACHE="${BEANS_NEXT_MODEL_CACHE:-$SCRATCH_ROOT/beans-next-models}"
readonly MANIFEST="${BEANS_NEXT_GEMMA4_MANIFEST:-$MODEL_CACHE/gemma4-manifest.${SLURM_JOB_ID}.json}"

if [[ ! -x "$RUNNER_PYTHON" ]]; then
  echo "Missing runner Python: $RUNNER_PYTHON" >&2
  exit 1
fi

mkdir -p "$MODEL_CACHE" "$RUNNER_RUNTIME/cache/huggingface" "$RUNNER_RUNTIME/tmp"

export HF_HOME="$RUNNER_RUNTIME/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"
export TMPDIR="$RUNNER_RUNTIME/tmp"

"$RUNNER_PYTHON" "$REPO/examples/apocrita/download_gemma4.py" \
  --cache-dir "$MODEL_CACHE" \
  --manifest "$MANIFEST" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "gemma4_manifest=$MANIFEST"
