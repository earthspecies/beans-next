#!/usr/bin/env bash
# Download the NVIDIA Audex 30B-A3B snapshot into its own model cache.
# Not gated; ~67 GB after skipping enhancement_VAE/, audex_causal_speech_decoder/,
# checkpoint_folder_audiogen/, checkpoint_folder_textonly/, and assets/.

#SBATCH --job-name=beans-audex-download
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next-audex repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly SERVER_RUNTIME="${BEANS_NEXT_AUDEX_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-audex-runtime}"
readonly SERVER_PYTHON="$SERVER_RUNTIME/.venv/bin/python"
readonly MODEL_CACHE="${BEANS_NEXT_MODEL_CACHE:-$SCRATCH_ROOT/beans-next-models}"
readonly MANIFEST="${BEANS_NEXT_AUDEX_MANIFEST:-$MODEL_CACHE/audex-manifest.${SLURM_JOB_ID}.json}"

if [[ ! -x "$SERVER_PYTHON" ]]; then
  echo "Missing Audex runtime: $SERVER_PYTHON (run setup_audex_runtime.sbatch first)" >&2
  exit 1
fi

mkdir -p "$MODEL_CACHE" "$SERVER_RUNTIME/cache/huggingface" "$SERVER_RUNTIME/tmp"

export HF_HOME="$SERVER_RUNTIME/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"
export TMPDIR="$SERVER_RUNTIME/tmp"

"$SERVER_PYTHON" "$REPO/examples/apocrita/download_audex.py" \
  --cache-dir "$MODEL_CACHE" \
  --manifest "$MANIFEST" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "audex_manifest=$MANIFEST"
