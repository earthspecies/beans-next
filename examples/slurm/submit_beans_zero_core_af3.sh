#!/usr/bin/env bash
# Submit a full beans_zero_core evaluation run for Audio Flamingo Next.
#
# Usage (from repo root):
#   bash examples/slurm/submit_beans_zero_core_af3.sh
#
# License: NVIDIA OneWay Noncommercial License — non-commercial research only.
#
# Prereqs:
#   - Weights downloaded: snapshot_download nvidia/audio-flamingo-next-hf
#   - Launcher deps installed: cd examples/servers/af3 && uv venv && uv pip install -r requirements.txt

set -euo pipefail

RUN_TAG="af3_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_PRO_OUT_DIR:-${SCRATCH:-$HOME}/beans-next-results/$RUN_TAG}"

echo "Submitting serving job..."
SERVE_JOB=$(sbatch --parsable examples/slurm/serve_af3.sh)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"
echo "  Note: AF3 takes up to 15 min to load weights."

echo "Submitting inference job (depends on serve job $SERVE_JOB)..."
INFER_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_SUITE=beans_zero_core \
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
