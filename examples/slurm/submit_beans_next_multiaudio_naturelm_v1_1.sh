#!/usr/bin/env bash
# Submit a beans_next_multiaudio_core evaluation run for NatureLM-audio v1.1.
#
# Usage (from repo root):
#   NATURELM_GCS_CHECKPOINT_URI=gs://foundation-models/naturelm-audio-1.1/base_model/1290000 \
#     bash examples/slurm/submit_beans_next_multiaudio_naturelm_v1_1.sh
#
# Optional overrides:
#   BEANS_PRO_OUT_DIR=/my/path \
#   NATURELM_GCS_CHECKPOINT_URI=gs://... \
#     bash examples/slurm/submit_beans_next_multiaudio_naturelm_v1_1.sh

set -euo pipefail

GCS_URI="${NATURELM_GCS_CHECKPOINT_URI:-gs://foundation-models/naturelm-audio-1.1/base_model/1290000}"
CKPT_TAG="${GCS_URI##*/}"
RUN_TAG="naturelm_v1_1_multiaudio_${CKPT_TAG}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_PRO_OUT_DIR:-${SCRATCH:-$HOME}/beans-next-results/$RUN_TAG}"

echo "Submitting serving job..."
SERVE_JOB=$(
  NATURELM_GCS_CHECKPOINT_URI="$GCS_URI" \
  sbatch --parsable examples/slurm/serve_naturelm_v1_1.sh
)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

echo "Submitting inference job (depends on serve job $SERVE_JOB)..."
INFER_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_SUITE=beans_next_multiaudio_core \
  BEANS_PRO_DATA_SOURCE=esp_data \
  BEANS_PRO_RUN_ID="$RUN_TAG" \
  BEANS_PRO_OUT_DIR="$OUT_DIR" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)
echo "  Inference job: $INFER_JOB"
echo "  Log: ~/logs/$INFER_JOB.log"
echo "  Output: $OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel both: scancel $SERVE_JOB $INFER_JOB"
