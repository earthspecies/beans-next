#!/usr/bin/env bash
# SLURM job: run beans-next inference against a running launcher,
# but first delete ONLY the HuggingFace datasets cache entries for BEANS-Zero.
#
# This is the general (non-smoke) variant of:
#   examples/slurm/test_run_inference_clean_beans_zero_cache.sh
#
# Usage:
#   SERVE_JOB_ID=12345
#   BEANS_NEXT_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
#   BEANS_NEXT_DATA_SOURCE=huggingface \
#   BEANS_NEXT_TASK_ID=beans_zero_esc50 \
#   BEANS_NEXT_DATASET_NAME=esc50 \
#   sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_inference_clean_beans_zero_cache.sh
#
# Optional override:
#   HF_DATASETS_CACHE=/some/other/cache/path

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next inference (clean BEANS-Zero cache)"

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Must be set: path to the URL file written by the serving job.
export BEANS_NEXT_URL_FILE="${BEANS_NEXT_URL_FILE:-${BEANS_PRO_URL_FILE:-}}"
if [[ -z "${BEANS_NEXT_URL_FILE:-}" ]]; then
  echo "ERROR: BEANS_NEXT_URL_FILE must be set." >&2
  echo "Compat: BEANS_PRO_URL_FILE is also supported." >&2
  exit 1
fi

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

exec "${SLURM_SUBMIT_DIR}/examples/slurm/run_inference.sh"

