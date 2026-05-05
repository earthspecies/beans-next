#!/usr/bin/env bash
# Submit a full beans_zero_core evaluation run for NatureLM-audio v1.0.
#
# Usage (from repo root):
#   bash examples/slurm/submit_beans_zero_core_naturelm_v1_0.sh
#
# Optional overrides:
#   BEANS_NEXT_OUT_DIR=/my/path bash examples/slurm/submit_beans_zero_core_naturelm_v1_0.sh
#
# Prereqs:
#   - Weights downloaded: HF_HOME=/scratch/shared/hf_cache snapshot_download naturelm-audio-1.0
#   - NatureLM-audio code at $HOME/code/NatureLM-audio (or set NATURELM_CODE_DIR)
#   - HF_TOKEN set (or stored in ~/.config/huggingface/hf_token)

set -euo pipefail

RUN_TAG="naturelm_v1_0_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_NEXT_OUT_DIR:-${BEANS_PRO_OUT_DIR:-${SCRATCH:-$HOME}/beans-next-results/$RUN_TAG}}"

echo "Submitting serving job..."
SERVE_JOB=$(sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

echo "Submitting inference job (depends on serve job $SERVE_JOB)..."
INFER_JOB=$(
  BEANS_NEXT_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_NEXT_SUITE=beans_zero_core \
  BEANS_NEXT_RUN_ID="$RUN_TAG" \
  BEANS_NEXT_OUT_DIR="$OUT_DIR" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)
echo "  Inference job: $INFER_JOB"
echo "  Log: ~/logs/$INFER_JOB.log"
echo "  Output: $OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel both: scancel $SERVE_JOB $INFER_JOB"
