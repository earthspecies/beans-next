#!/usr/bin/env bash
# SLURM job: run BEANS-Next ESC-50 (official) against a running launcher.
#
# This is a thin wrapper around `examples/slurm/run_inference.sh` that defaults to:
# - task_id:     beans_zero_esc50_official
# - dataset:     esc50
# - limit:       1 (override to 1–5 while debugging; scale only after green)
#
# It reads the predict URL from a URL file written by a serving job (standard URL-file pattern),
# then polls `GET /health` before running `beans-next run`.
#
# Usage (two-job pattern):
#   SERVE_JOB_ID=12345
#   BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
#   BEANS_PRO_LIMIT=1 \
#   BEANS_PRO_OUT_DIR=/scratch/$USER/results/esc50_official_$SERVE_JOB_ID \
#   BEANS_PRO_RUN_ID=esc50_official_${SERVE_JOB_ID} \
#   sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_esc50_official_inference.sh
#
# Notes:
# - This job does not require GPUs (inference is remote; handled by the serving job).
# - Readiness discipline for the serve job is still required:
#   poll `squeue --me` until the serve job is `R`, then wait for the URL file, then poll `/health`.
#
# CPUs: 8 matches the default BEANS_PRO_ESP_DATA_WORKERS=8 set by run_inference.sh.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --export=ALL
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next esc50 official (inference)"

set -euo pipefail

export BEANS_PRO_TASK_ID="${BEANS_PRO_TASK_ID:-beans_zero_esc50_official}"
export BEANS_PRO_DATASET_NAME="${BEANS_PRO_DATASET_NAME:-esc50}"
export BEANS_PRO_LIMIT="${BEANS_PRO_LIMIT:-1}"

exec "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/examples/slurm/run_inference.sh"

