#!/usr/bin/env bash
# Submit a full BirdSet core evaluation run via the OpenAI-compatible proxy (Gemini).
#
# Usage (from repo root on Slurm login node):
#   bash examples/slurm/submit_birdset_core_openai_proxy_gemini.sh
#
# Optional overrides:
#   BEANS_PRO_OUT_DIR=/my/path bash examples/slurm/submit_birdset_core_openai_proxy_gemini.sh
#
# Notes:
# - Requires network egress from the proxy node to Google's OpenAI-compatible endpoint.
# - Secrets are loaded by the serve script from ~/.config/gemini/cfg (do not echo keys).
# - BirdSet uses esp_data: this script sets BEANS_PRO_DATA_SOURCE=esp_data.

set -euo pipefail

RUN_TAG="birdset_core_gemini_proxy_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_PRO_OUT_DIR:-${SCRATCH:-$HOME}/beans-next-results/$RUN_TAG}"

echo "Submitting proxy serving job..."
SERVE_JOB="$(sbatch --parsable examples/slurm/serve_openai_proxy_gemini.sh)"
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

echo "Submitting inference job (depends on serve job $SERVE_JOB)..."
INFER_JOB="$(
  BEANS_PRO_URL_FILE=\"$HOME/beans-next-launchers/$SERVE_JOB.url\" \
  BEANS_PRO_DATA_SOURCE=esp_data \
  BEANS_PRO_SUITE=birdset_core \
  BEANS_PRO_RUN_ID=\"$RUN_TAG\" \
  BEANS_PRO_OUT_DIR=\"$OUT_DIR\" \
  sbatch --parsable --dependency=after:\"$SERVE_JOB\" examples/slurm/run_inference.sh
)"
echo "  Inference job: $INFER_JOB"
echo "  Log: ~/logs/$INFER_JOB.log"
echo "  Output: $OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel both: scancel $SERVE_JOB $INFER_JOB"

