#!/usr/bin/env bash
# Submit a BEANS-Next tier 4 (in-context multi-audio) run for Qwen3-Omni.
#
# Tier 4 sends several audios per request, so the launcher must be started with
# a multimodal-per-prompt limit that covers the widest row (6 audios). Without
# it vLLM keeps only the first audio and the run silently degrades to the
# query-only reformulation, which is near chance on the 4-way tasks.
#
# Usage (from repo root, on the Slurm login node):
#   bash examples/slurm/submit_beans_next_tier4_qwen3_omni.sh
#
# Optional overrides:
#   BEANS_NEXT_LIMIT=200      cap rows per task (first-pass sweeps)
#   BEANS_NEXT_SUITE=...      default beans_next_tier4
#   BEANS_NEXT_OUT_DIR=...    default $HOME/beans-next-results/<run tag>

set -euo pipefail

SUITE="${BEANS_NEXT_SUITE:-beans_next_tier4}"
RUN_TAG="qwen3_omni_tier4_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BEANS_NEXT_OUT_DIR:-$HOME/beans-next-results/$RUN_TAG}"

echo "Submitting Qwen3-Omni serving job..."
SERVE_JOB=$(
  VLLM_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct" \
  VLLM_MAX_BATCH_SIZE=1 \
  VLLM_TENSOR_PARALLEL_SIZE=1 \
  VLLM_OMNI=1 \
  VLLM_OMNI_INSTALL=1 \
  VLLM_INSTALL_VERSION=0.18.0 \
  VLLM_OMNI_VERSION=0.18.0 \
  VLLM_MAX_MODEL_LEN=16384 \
  VLLM_AUDIO_CONTENT_FORMAT=audio_url_data \
  VLLM_LIMIT_MM_PER_PROMPT='{"audio": 6}' \
  HF_HUB_DISABLE_XET=1 \
  BEANS_NEXT_HF_HOME="$HOME/hf_cache_qwen3_omni_instruct_lean" \
  VLLM_EXTRA_ARGS="--stage-configs-path $PWD/examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml --download-dir $HOME/hf_cache_qwen3_omni_instruct_lean" \
  sbatch --parsable --partition=h100-80 --gpus=1 --mem=200G \
    --job-name="bn-tier4-qwen-serve" examples/slurm/serve_vllm.sh
)
echo "  Serving job: $SERVE_JOB (log: ~/logs/$SERVE_JOB.log)"

echo "Submitting inference job (depends on serve job $SERVE_JOB)..."
INFER_JOB=$(
  BEANS_NEXT_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_NEXT_SUITE="$SUITE" \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_RUN_ID="$RUN_TAG" \
  BEANS_NEXT_OUT_DIR="$OUT_DIR" \
  ${BEANS_NEXT_LIMIT:+BEANS_NEXT_LIMIT="$BEANS_NEXT_LIMIT"} \
  sbatch --parsable --partition=cpu --mem=20480 \
    --dependency=after:"$SERVE_JOB" examples/slurm/run_inference.sh
)
echo "  Inference job: $INFER_JOB (log: ~/logs/$INFER_JOB.log)"
echo "  Output: $OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel both: scancel $SERVE_JOB $INFER_JOB"
