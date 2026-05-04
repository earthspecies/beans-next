#!/usr/bin/env bash
# Submit a second NatureLM-audio v1.0 server + a small BEANS-Zero core fixtures run.
#
# Goal: collect 20 examples per BEANS-Zero core subset and store artifacts on the
# shared filesystem (local to the repo) so they can be curated into test fixtures.
#
# Usage (from repo root, on Slurm login node):
#   bash examples/slurm/submit_beans_zero_core_naturelm_v1_0_fixtures_esp_data.sh
#
# Optional overrides:
#   BEANS_PRO_FIXTURE_LIMIT=20 \
#   BEANS_PRO_FIXTURE_OUT_DIR=/some/shared/path \
#   bash examples/slurm/submit_beans_zero_core_naturelm_v1_0_fixtures_esp_data.sh
#
# Notes:
# - This script *does not* upload to GCS. It keeps everything local first.
# - Runs a tiny smoke job before the fixture run.

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.config/huggingface/hf_token" ]]; then
  export HF_TOKEN="$(< "$HOME/.config/huggingface/hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

REPO_ROOT="${PWD}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "ERROR: run from repo root (missing pyproject.toml in $REPO_ROOT)" >&2
  exit 1
fi

INC="${BEANS_PRO_INC:-adhoc}"
MODEL_DIR="naturelm_v1_0"
SUBSET_DIR="beans_zero_core"
TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_RUN_ID="smoke_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FIXTURE_RUN_ID="fixtures20_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FIXTURE_LIMIT="${BEANS_PRO_FIXTURE_LIMIT:-20}"

DEFAULT_FIXTURE_OUT_DIR="${REPO_ROOT}/results/fixtures/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${FIXTURE_RUN_ID}"
SMOKE_OUT_DIR="${BEANS_PRO_OUT_DIR_SMOKE:-${DEFAULT_FIXTURE_OUT_DIR}/smoke}"
FIXTURE_OUT_DIR="${BEANS_PRO_FIXTURE_OUT_DIR:-${DEFAULT_FIXTURE_OUT_DIR}/run}"

CONFIG_PATH="configs/benchmarks/beans_zero_core_naturelm_v1_0_esp_data.yaml"

mkdir -p "$SMOKE_OUT_DIR" "$FIXTURE_OUT_DIR"

echo "Submitting NatureLM v1.0 serving job..."
SERVE_JOB=$(
  HF_TOKEN="${HF_TOKEN:-}" \
  HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh
)
echo "  Serving job: $SERVE_JOB"
echo "  Log: ~/logs/$SERVE_JOB.log"

echo "Submitting smoke inference job (depends on serve job $SERVE_JOB)..."
SMOKE_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_DATA_SOURCE="esp_data" \
  BEANS_PRO_CONFIG="$CONFIG_PATH" \
  BEANS_PRO_LIMIT="${BEANS_PRO_SMOKE_LIMIT:-2}" \
  BEANS_PRO_RUN_ID="$SMOKE_RUN_ID" \
  BEANS_PRO_OUT_DIR="$SMOKE_OUT_DIR" \
  sbatch --parsable --dependency=after:"$SERVE_JOB" examples/slurm/test_run_inference.sh
)
echo "  Smoke job: $SMOKE_JOB"
echo "  Log: ~/logs/$SMOKE_JOB.log"
echo "  Output: $SMOKE_OUT_DIR"

echo "Submitting fixtures inference job (afterok smoke $SMOKE_JOB)..."
FIXTURE_JOB=$(
  BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/$SERVE_JOB.url" \
  BEANS_PRO_DATA_SOURCE="esp_data" \
  BEANS_PRO_CONFIG="$CONFIG_PATH" \
  BEANS_PRO_LIMIT="$FIXTURE_LIMIT" \
  BEANS_PRO_RUN_ID="$FIXTURE_RUN_ID" \
  BEANS_PRO_OUT_DIR="$FIXTURE_OUT_DIR" \
  sbatch --parsable --dependency=afterok:"$SMOKE_JOB" examples/slurm/run_inference.sh
)
echo "  Fixtures job: $FIXTURE_JOB"
echo "  Log: ~/logs/$FIXTURE_JOB.log"
echo "  Output: $FIXTURE_OUT_DIR"
echo ""
echo "Monitor: squeue --me"
echo "Cancel: scancel $SERVE_JOB $SMOKE_JOB $FIXTURE_JOB"

