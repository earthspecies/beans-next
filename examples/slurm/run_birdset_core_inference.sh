#!/usr/bin/env bash
# SLURM job wrapper: run the BirdSet core suite (all 8 subsets) against a running launcher.
#
# This wraps `examples/slurm/run_inference.sh` with BirdSet-specific defaults:
# - BEANS_NEXT_SUITE=birdset_core
# - BEANS_NEXT_DATA_SOURCE=esp_data
#
# Usage:
#   SERVE_JOB_ID=12345
#   BEANS_NEXT_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
#   BEANS_NEXT_OUT_DIR=/scratch/$USER/results/birdset_core_run \
#   sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_birdset_core_inference.sh
#
# Optional overrides:
# - BEANS_NEXT_LIMIT=5 (debug)
# - BEANS_NEXT_RUN_ID=custom_name
# - BEANS_NEXT_ESP_DATA_WORKERS=8
# - BEANS_NEXT_INFERENCE_WORKERS=1

# BirdSet inference is CPU-only (remote launcher performs model inference).
# Ensure these jobs land on the CPU partition by default.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next inference"

set -euo pipefail

export BEANS_NEXT_SUITE="${BEANS_NEXT_SUITE:-${BEANS_PRO_SUITE:-birdset_core}}"
export BEANS_NEXT_DATA_SOURCE="${BEANS_NEXT_DATA_SOURCE:-${BEANS_PRO_DATA_SOURCE:-esp_data}}"

# Default to parallel esp_data audio materialization/downloading on CPU nodes.
export BEANS_NEXT_ESP_DATA_WORKERS="${BEANS_NEXT_ESP_DATA_WORKERS:-${BEANS_PRO_ESP_DATA_WORKERS:-8}}"

# Keep inference workers conservative by default; remote launchers can bottleneck.
export BEANS_NEXT_INFERENCE_WORKERS="${BEANS_NEXT_INFERENCE_WORKERS:-${BEANS_PRO_INFERENCE_WORKERS:-1}}"

exec bash examples/slurm/run_inference.sh

