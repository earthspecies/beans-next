#!/usr/bin/env bash
# Submit a full beans_zero_core evaluation run for NatureLM-audio v1.1.
#
# Usage (from repo root):
#   HF_TOKEN=hf_... bash examples/slurm/submit_beans_zero_core_naturelm_v1_1.sh
#
# Optional overrides:
#   BEANS_PRO_OUT_DIR=/my/path HF_TOKEN=hf_... bash examples/slurm/submit_beans_zero_core_naturelm_v1_1.sh
#
# Prereqs:
#   - Gated HuggingFace access to EarthSpeciesProject/naturelm-audio-1.1.00-private
#   - HF_TOKEN set in env, ~/.config/huggingface/hf_token, or HUGGINGFACE_HUB_TOKEN
#   - Weights downloaded: HF_TOKEN=hf_... snapshot_download naturelm-audio-1.1.00-private

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.config/huggingface/hf_token" ]]; then
  export HF_TOKEN="$(< "$HOME/.config/huggingface/hf_token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required for NatureLM v1.1 (gated weights)." >&2
  echo "Set HF_TOKEN or store it in ~/.config/huggingface/hf_token" >&2
  exit 1
fi

RUN_TAG="naturelm_v1_1_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_PRO_OUT_DIR:-${SCRATCH:-$HOME}/beans-next-results/$RUN_TAG}"

echo "Submitting serving job..."
SERVE_JOB=$(HF_TOKEN="$HF_TOKEN" sbatch --parsable examples/slurm/serve_naturelm_v1_1.sh)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

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
