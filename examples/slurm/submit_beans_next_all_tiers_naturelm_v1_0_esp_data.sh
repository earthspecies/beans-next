#!/usr/bin/env bash
# Serve NatureLM-audio v1.0 on Slurm and run BEANS-Next inference for all tiers.
#
# Usage (from repo root on the Slurm login node):
#   export HF_TOKEN="$(< ~/.config/huggingface/hf_token)"
#   export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
#   bash examples/slurm/submit_beans_next_all_tiers_naturelm_v1_0_esp_data.sh
#
# Optional overrides:
#   BEANS_NEXT_OUT_DIR=/my/path \
#     bash examples/slurm/submit_beans_next_all_tiers_naturelm_v1_0_esp_data.sh
#
# Notes:
# - Dataset backend policy: always use `esp_data` in these workflows.
# - The serve job writes: $HOME/beans-next-launchers/<job_id>.url after /health passes.

set -euo pipefail

RUN_TAG="naturelm_v1_0_beans_next_all_tiers_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_NEXT_OUT_DIR:-${BEANS_PRO_OUT_DIR:-${SCRATCH:-$HOME}/beans-next-results/$RUN_TAG}}"

echo "Submitting serving job (NatureLM v1.0)..."
SERVE_JOB="$(sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh)"
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

echo "Submitting inference job (depends on serve job $SERVE_JOB)..."
INFER_JOB="$(
  BEANS_NEXT_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_NEXT_SUITE=beans_next_all_tiers \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_RUN_ID="$RUN_TAG" \
  BEANS_NEXT_OUT_DIR="$OUT_DIR" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)"
echo "  Inference job: $INFER_JOB"
echo "  Log: ~/logs/$INFER_JOB.log"
echo "  Output: $OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel both: scancel $SERVE_JOB $INFER_JOB"

