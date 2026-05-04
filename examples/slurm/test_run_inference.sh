#!/usr/bin/env bash
# SLURM job: tiny "smoke" inference run against a running launcher.
#
# Purpose:
# - Validate end-to-end wiring (URL file → /health → beans-next run) with a small cap.
# - Intended for the two-job Slurm pattern, where the model is served by a separate serve_*.sh job.
#
# Requirements:
# - BEANS_PRO_URL_FILE must point at the URL file written by the serving job:
#     $HOME/beans-next-launchers/<serve_job_id>.url
#
# Defaults:
# - Suite: beans_zero_core
# - Limit: 3
# - Output: $SCRATCH/beans-next-results/test_run_<job_id>
# - Optional copy-back: BEANS_PRO_COPY_RESULTS_TO_HOME=1 (recommended)
#
# Example:
#   SERVE_JOB_ID=12345
#   BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
#   sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/test_run_inference.sh
#
# Note:
# - This script does not request GPUs (inference is remote; the launcher does the GPU work).
# - Partition names are site-specific; this repo's cluster provides a `cpu` partition.

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next test inference"

set -euo pipefail

# Must be set: path to the URL file written by the serving job.
export BEANS_PRO_URL_FILE="${BEANS_PRO_URL_FILE:?BEANS_PRO_URL_FILE must be set}"

# Safe defaults for a tiny run (override via env before sbatch).
# Use the smallest bundled suite by default to keep validation fast.
export BEANS_PRO_SUITE="${BEANS_PRO_SUITE:-beans_zero_smoke}"
export BEANS_PRO_LIMIT="${BEANS_PRO_LIMIT:-1}"

# Mark this run as a smoke validation so the standardized naming policy in
# examples/slurm/run_inference.sh encodes it in RUN_ID/OUT_DIR.
export BEANS_PRO_RUN_KIND="${BEANS_PRO_RUN_KIND:-smoke}"

# For BEANS-Zero smoke runs on the cluster, default to esp_data to avoid HF streaming on compute nodes.
if [[ -z "${BEANS_PRO_DATA_SOURCE:-}" && "${BEANS_PRO_SUITE:-}" =~ ^beans_zero_ ]]; then
  export BEANS_PRO_DATA_SOURCE="esp_data"
fi

# Work around rare interpreter-finalization crashes seen on this cluster by hard-exiting
# the CLI after producing outputs.
export BEANS_PRO_HARD_EXIT="${BEANS_PRO_HARD_EXIT:-1}"

# Recommended: copy artifacts into $HOME so they're visible on this repo host via /mnt/home.
export BEANS_PRO_COPY_RESULTS_TO_HOME="${BEANS_PRO_COPY_RESULTS_TO_HOME:-1}"

# Delegate to the main inference script.
# In Slurm, scripts are copied to a spool dir before execution, so $0 does not point at the repo.
if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi
exec "${SLURM_SUBMIT_DIR}/examples/slurm/run_inference.sh"

