#!/usr/bin/env bash
# SLURM job: run ESC-50 official inference on a CPU node against a running v1.1 server URL.
#
# NOTE: This is an exception workflow. The default project policy is to run inference locally
# (hybrid pattern). Use this only when explicitly requested.
#
# Submit:
#   sbatch examples/slurm/run_esc50_official_cpu_inference_naturelm_v1_1.sh
#
# Override server URL (optional):
#   BEANS_PRO_PREDICT_URL=http://... sbatch ...
#
# Or point at a URL file (optional):
#   BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/<serve_job_id>.url sbatch ...
#
# Defaults assume serve job 56631 wrote:
#   $HOME/beans-next-launchers/56631.url

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --export=ALL
#SBATCH --output="/home/%u/logs/%A_infer_esc50_v11_cpu.log"
#SBATCH --job-name="infer-esc50-v11-cpu"

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR not set; submit from repo root." >&2
  exit 1
fi

cd "$REPO"

# Ensure the project environment includes esp_data.
uv sync --group esp

URL_FILE_DEFAULT="$HOME/beans-next-launchers/56631.url"
URL_FILE="${BEANS_PRO_URL_FILE:-$URL_FILE_DEFAULT}"

PREDICT_URL="${BEANS_PRO_PREDICT_URL:-}"
if [[ -z "$PREDICT_URL" ]]; then
  if [[ ! -f "$URL_FILE" ]]; then
    echo "ERROR: URL file not found: $URL_FILE" >&2
    exit 1
  fi
  PREDICT_URL="$(head -n 1 "$URL_FILE" | tr -d '[:space:]')"
fi

if [[ -z "$PREDICT_URL" ]]; then
  echo "ERROR: predict URL is empty (file: $URL_FILE)" >&2
  exit 1
fi

OUT="/home/$USER/beans-next-results/raw/esc50_official_v11_full_cpu_${SLURM_JOB_ID}_esp_data"
mkdir -p "$OUT"

uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --backend "${BEANS_PRO_DATA_SOURCE:-esp_data}" \
  --split esc50 \
  --dataset-name esc50 \
  --task-id beans_zero_esc50_official \
  --prompt-yaml beans_next/registry/prompt/classification_beans_zero_official_v1.yaml \
  --limit 400 \
  --run-id "esc50_official_v11_full_cpu_${SLURM_JOB_ID}_${BEANS_PRO_DATA_SOURCE:-esp_data}" \
  -o "$OUT"

echo "DONE. Output dir: $OUT"

