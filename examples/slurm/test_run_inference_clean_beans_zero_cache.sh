#!/usr/bin/env bash
# SLURM job: tiny "smoke" inference run against a running launcher,
# but first delete ONLY the HuggingFace datasets cache entries for BEANS-Zero.
#
# Why:
# - Downgrading `datasets` can fail when BEANS-Zero was previously cached by a newer
#   `datasets` version (feature/schema parsing error).
# - This wrapper makes the CPU node state deterministic for BEANS-Zero loads.
#
# What it deletes:
# - Any `*.lock` in `$HF_DATASETS_CACHE/` that mentions `EarthSpeciesProject___beans-zero`
#
# It does NOT delete the dataset contents (to avoid re-downloading).
#
# Usage:
#   SERVE_JOB_ID=12345
#   BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
#   sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/test_run_inference_clean_beans_zero_cache.sh
#
# Optional override (rare):
#   HF_DATASETS_CACHE=/some/other/cache/path

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next test inference (clean BEANS-Zero cache)"

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Must be set: path to the URL file written by the serving job.
export BEANS_PRO_URL_FILE="${BEANS_PRO_URL_FILE:?BEANS_PRO_URL_FILE must be set}"

# Choose the datasets cache location (prefer explicit override).
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"

TARGET_DIR="${HF_DATASETS_CACHE%/}/EarthSpeciesProject___beans-zero"

echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "Clearing BEANS-Zero datasets cache locks (only): $TARGET_DIR"

shopt -s nullglob
locks=("${HF_DATASETS_CACHE%/}/"*.lock)
for lock in "${locks[@]}"; do
  if [[ "$(basename "$lock")" == *"EarthSpeciesProject___beans-zero"* ]]; then
    rm -f "$lock"
    echo "Deleted lock: $lock"
  fi
done
shopt -u nullglob

# Delegate to the normal tiny inference script (which delegates to run_inference.sh).
exec "${SLURM_SUBMIT_DIR}/examples/slurm/test_run_inference.sh"

