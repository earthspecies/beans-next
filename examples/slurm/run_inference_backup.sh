#!/usr/bin/env bash
# SLURM job: run beans-next inference against a running launcher.
#
# NOTE: This is a backup of a modified `run_inference.sh` version.
# The canonical script is `run_inference.sh`.
#
# Reads the predict URL from a URL file written by a serving job.
# Can run on a CPU node — no GPU needed; all inference is done by the launcher.
# Note: the default `#SBATCH --partition` below is a placeholder; set it to your site’s
# CPU (or general) partition as appropriate.
#
# Usage:
#   # After submitting a serving job (e.g. serve_af3.sh), note its job id:
#   SERVE_JOB_ID=12345
#
#   BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
#   BEANS_PRO_SUITE=beans_zero_core \
#   BEANS_PRO_LIMIT=1 \
#   BEANS_PRO_OUT_DIR=/scratch/$USER/results/af3_run \
#   sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_inference.sh
#
# Keep BEANS_PRO_LIMIT in the 1–5 range while debugging wiring or launcher issues.
#
# The job waits for the URL file to appear (written by the serving job), then polls
# `GET /health` before starting inference.

# Inference does not need GPUs. Use a CPU (or general) partition if available.
# This cluster provides a `cpu` partition for non-GPU jobs. Check: sinfo; squeue --me
# Override partition at submit time if needed:
#   sbatch --partition=cpu examples/slurm/run_inference.sh
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next inference"

# This script does not request GPUs (inference is remote — handled by the serving job).

set -euo pipefail

# In Slurm, scripts are copied to a spool dir before execution, so $0 does not point at the repo.
# Use the submission directory as the repo root (we submit from the repo root in all runbooks).
REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Ensure all `uv` operations use a job-scoped environment on node-local scratch.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/beans-next-infer-${SLURM_JOB_ID}}"

# Audio materialization cache: shared scratch location, cleaned by scratch_guard when space is low.
export BEANS_PRO_HF_AUDIO_CACHE_DIR="${BEANS_PRO_HF_AUDIO_CACHE_DIR:-/scratch/.cache/huggingface/beans-next-audio}"
export BEANS_PRO_ESP_AUDIO_CACHE_DIR="${BEANS_PRO_ESP_AUDIO_CACHE_DIR:-/scratch/.cache/huggingface/beans-next-audio}"

# Shell helpers (bash-only; avoid non-portable external deps).
_pause() {
  # Wait between polls.
  #
  # Note: this is intentionally *not* a fixed "sleep N minutes then hope" delay; it is
  # only used as the polling interval for URL-file presence and HTTP `/health`.
  local timeout_sec="$1"
  sleep "$timeout_sec"
}

_trim_ws() {
  local s="$1"
  # Trim leading whitespace.
  s="${s#"${s%%[![:space:]]*}"}"
  # Trim trailing whitespace.
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

_poll_health() {
  local base_url="$1"
  local timeout_sec="$2"
  local interval_sec="$3"
  local connect_timeout_sec="$4"
  local max_time_sec="$5"

  local health_url="${base_url%/}/health"
  local deadline=$((SECONDS + timeout_sec))
  local attempt=0
  local last_err=""

  while (( SECONDS < deadline )); do
    attempt=$((attempt + 1))
    if curl -fsS -o /dev/null \
      --connect-timeout "$connect_timeout_sec" \
      --max-time "$max_time_sec" \
      "$health_url" >/dev/null 2>&1; then
      echo "Launcher healthy: $health_url (attempt=$attempt)"
      return 0
    fi

    # Capture the most recent error for debug output.
    local curl_out=""
    local curl_code=0
    set +e
    curl_out="$(
      curl -fsS -o /dev/null \
        --connect-timeout "$connect_timeout_sec" \
        --max-time "$max_time_sec" \
        "$health_url" 2>&1
    )"
    curl_code=$?
    set -e
    last_err="curl_exit_code=${curl_code} ${curl_out}"

    _pause "$interval_sec"
  done

  echo "ERROR: launcher did not become healthy within ${timeout_sec}s: $health_url" >&2
  if [[ -n "$last_err" ]]; then
    echo "ERROR: last health check failure: $last_err" >&2
  fi
  return 1
}

_is_http_url() {
  local url="$1"
  [[ "$url" =~ ^https?://[^[:space:]]+$ ]]
}

_normalize_predict_url() {
  local url="$1"
  url="$(_trim_ws "$url")"
  # Allow URL files like:
  # - http://host:port/predict
  # - http://host:port
  # - PREDICT_URL=http://host:port/predict
  # - http://host:port/predict  # comment
  if [[ "$url" == *"#"* ]]; then
    url="$(_trim_ws "${url%%#*}")"
  fi
  if [[ "$url" == *=* ]]; then
    local rhs="${url#*=}"
    rhs="$(_trim_ws "$rhs")"
    if [[ -n "$rhs" ]]; then
      url="$rhs"
    fi
  fi
  if [[ -z "$url" ]]; then
    echo ""
    return 0
  fi

  # Allow either:
  # - full predict URL: http(s)://host:port/predict
  # - base URL:         http(s)://host:port  (we append /predict)
  url="${url%/}"

  if ! _is_http_url "$url"; then
    echo ""
    return 0
  fi

  if [[ "$url" =~ /predict$ ]]; then
    echo "$url"
    return 0
  fi

  if [[ "$url" =~ /health$ ]]; then
    url="${url%/health}"
  fi

  echo "${url}/predict"
}

# Required: path to the URL file written by the serving job.
URL_FILE="${BEANS_PRO_URL_FILE:?BEANS_PRO_URL_FILE must be set to the serving job URL file}"

# Run parameters (override via env before sbatch).
SUITE="${BEANS_PRO_SUITE:-beans_zero_core}"
LIMIT="${BEANS_PRO_LIMIT:-}"
OUT_DIR="${BEANS_PRO_OUT_DIR:-$HOME/beans-next-results/run_${SLURM_JOB_ID}}"
CONFIG="${BEANS_PRO_CONFIG:-}"           # optional: path to a run config YAML
RUN_ID="${BEANS_PRO_RUN_ID:-slurm_${SLURM_JOB_ID}}"
TASK_ID="${BEANS_PRO_TASK_ID:-}"         # optional: run one eval task directly
DATASET_NAME="${BEANS_PRO_DATASET_NAME:-}"  # optional: e.g. esc50 (required when TASK_ID set)
SPLIT="${BEANS_PRO_SPLIT:-}"             # optional: default is CLI default (test)
HF_PATH="${BEANS_PRO_HF_PATH:-}"         # optional: override hub dataset id
HF_CONFIG="${BEANS_PRO_HF_CONFIG:-}"     # optional: override hub config name
WORKERS="${BEANS_PRO_WORKERS:-}"         # optional: override runner --workers

cd "$REPO"

# Dataset backend selection:
# - For BEANS-Zero suites, prefer esp_data on the cluster (compute nodes may not have outbound HF access).
# - Override any time by exporting BEANS_PRO_DATA_SOURCE=hf (or esp_data).
if [[ -z "${BEANS_PRO_DATA_SOURCE:-}" && "$SUITE" =~ ^beans_zero_ ]]; then
  export BEANS_PRO_DATA_SOURCE="esp_data"
fi

# Work around rare interpreter-finalization crashes seen on this cluster by hard-exiting
# the CLI after producing outputs (can be overridden).
export BEANS_PRO_HARD_EXIT="${BEANS_PRO_HARD_EXIT:-1}"

# Prefer Python 3.11+ for this project. Some compute nodes may not have a compatible system
# interpreter, so allow uv to download a managed Python when needed.
# Override if needed: BEANS_PRO_UV_PYTHON=3.11 (default).
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_PRO_UV_PYTHON:-3.11}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Ensure the job-scoped environment exists.
# If esp_data is requested, try to inherit any site-provided esp_data install via system site-packages.
if [[ ! -x "${UV_PROJECT_ENVIRONMENT%/}/bin/python" ]]; then
  if [[ "${BEANS_PRO_DATA_SOURCE:-hf}" == "esp_data" ]]; then
    uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT"
  else
    uv venv "$UV_PROJECT_ENVIRONMENT"
  fi
fi

# On clusters where compute nodes cannot reach the package index, `uv sync` inside the job may fail.
# Set `BEANS_PRO_SKIP_UV_SYNC=1` if you have pre-built the environment on a shared filesystem.
#
# If `esp_data` is selected, include the `esp` dependency group (configured in pyproject.toml).
if [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
  if [[ "${BEANS_PRO_DATA_SOURCE:-hf}" == "esp_data" ]]; then
    uv sync --group esp
  else
    uv sync
  fi
else
  echo "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync'"
fi

if [[ "${BEANS_PRO_DATA_SOURCE:-hf}" == "esp_data" ]]; then
  set +e
  esp_data_import_out="$(uv run python -c "import esp_data" 2>&1)"
  has_esp_data=$?
  set -e
  if [[ "$has_esp_data" != "0" ]]; then
    # Note: we do not run `uv pip install` here. If esp_data isn't present, add it to the
    # project's `esp` dependency group (via `uv add --group esp esp-data`) and rerun.
    echo "ERROR: BEANS_PRO_DATA_SOURCE=esp_data but 'esp_data' is not importable in this job environment." >&2
    if [[ -n "${esp_data_import_out:-}" ]]; then
      echo "ERROR: esp_data import output:" >&2
      echo "${esp_data_import_out}" >&2
    fi
    echo "Fix options:" >&2
    echo "  - Ensure this job ran 'uv sync --group esp' successfully (default when BEANS_PRO_DATA_SOURCE=esp_data)." >&2
    echo "  - Ensure your 'pyproject.toml' has an 'esp' dependency group with 'esp-data', and 'tool.uv.index'/'tool.uv.sources' configured for esp-pypi." >&2
    echo "  - Or force HuggingFace loading: export BEANS_PRO_DATA_SOURCE=hf" >&2
    exit 1
  fi
fi

# Wait for the URL file to appear (serving job may still be loading weights).
echo "Waiting for URL file: $URL_FILE"
URL_WAIT_TIMEOUT_SEC="${BEANS_PRO_URL_WAIT_TIMEOUT_SEC:-1800}"
URL_WAIT_INTERVAL_SEC="${BEANS_PRO_URL_WAIT_INTERVAL_SEC:-5}"
deadline=$((SECONDS + URL_WAIT_TIMEOUT_SEC))

while (( SECONDS < deadline )); do
  if [[ -s "$URL_FILE" ]]; then
    break
  fi
  _pause "$URL_WAIT_INTERVAL_SEC"
done

if [[ ! -s "$URL_FILE" ]]; then
  echo "ERROR: URL file did not appear within ${URL_WAIT_TIMEOUT_SEC}s: $URL_FILE" >&2
  exit 1
fi

PREDICT_URL=""
IFS= read -r PREDICT_URL <"$URL_FILE" || true
PREDICT_URL="$(_normalize_predict_url "$PREDICT_URL")"
if [[ -z "$PREDICT_URL" ]]; then
  echo "ERROR: URL file did not contain a valid http(s) URL: $URL_FILE" >&2
  exit 1
fi
echo "Using predict URL: $PREDICT_URL"

# Poll /health at the base URL (robust to URL file being present before network is ready).
BASE_URL="${PREDICT_URL%/predict}"
HEALTH_TIMEOUT_SEC="${BEANS_PRO_HEALTH_TIMEOUT_SEC:-900}"
HEALTH_INTERVAL_SEC="${BEANS_PRO_HEALTH_INTERVAL_SEC:-5}"
HEALTH_CONNECT_TIMEOUT_SEC="${BEANS_PRO_HEALTH_CONNECT_TIMEOUT_SEC:-2}"
HEALTH_MAX_TIME_SEC="${BEANS_PRO_HEALTH_MAX_TIME_SEC:-5}"
_poll_health \
  "$BASE_URL" \
  "$HEALTH_TIMEOUT_SEC" \
  "$HEALTH_INTERVAL_SEC" \
  "$HEALTH_CONNECT_TIMEOUT_SEC" \
  "$HEALTH_MAX_TIME_SEC"

# Build CLI args.
CLI_ARGS=(
  run
  --predict-url "$PREDICT_URL"
  --run-id "$RUN_ID"
  -o "$OUT_DIR"
)

if [[ -n "$WORKERS" ]]; then
  CLI_ARGS+=(--workers "$WORKERS")
fi

if [[ -n "$CONFIG" ]]; then
  CLI_ARGS+=(--config "$CONFIG")
elif [[ -n "$TASK_ID" ]]; then
  CLI_ARGS+=(--task-id "$TASK_ID")
  if [[ -z "$DATASET_NAME" ]]; then
    echo "ERROR: BEANS_PRO_TASK_ID was set but BEANS_PRO_DATASET_NAME is empty." >&2
    echo "Example: BEANS_PRO_TASK_ID=beans_zero_esc50 BEANS_PRO_DATASET_NAME=esc50" >&2
    exit 1
  fi
  CLI_ARGS+=(--dataset-name "$DATASET_NAME")
  if [[ -n "$SPLIT" ]]; then
    CLI_ARGS+=(--split "$SPLIT")
  fi
  if [[ -n "$HF_PATH" ]]; then
    CLI_ARGS+=(--hf-path "$HF_PATH")
  fi
  if [[ -n "$HF_CONFIG" ]]; then
    CLI_ARGS+=(--hf-config "$HF_CONFIG")
  fi
else
  CLI_ARGS+=(--suite "$SUITE")
fi

if [[ -n "$LIMIT" ]]; then
  CLI_ARGS+=(--limit "$LIMIT")
fi

echo "Running: uv run beans-next ${CLI_ARGS[*]}"

# Run the CLI in-process and hard-exit explicitly.
#
# Why: this cluster sometimes triggers interpreter-finalization crashes (PyGILState_Release).
# The console-script path relies on env propagation (`BEANS_PRO_HARD_EXIT=1`) being honored;
# in practice we've seen jobs still crash. This wrapper guarantees `os._exit()` is called
# after the handler returns.
ARGS_FILE="/scratch/$USER/beans-next-args-${SLURM_JOB_ID}.txt"
rm -f "$ARGS_FILE"
printf '%s\n' "${CLI_ARGS[@]}" >"$ARGS_FILE"
export ARGS_FILE

mkdir -p "$OUT_DIR"

# Run in background so we can emit a heartbeat. This helps distinguish:
# - dataset/backends stalled before first request
# - normal slow progress
# - interpreter-finalization hangs (should be avoided by os._exit below)
HEARTBEAT_SEC="${BEANS_PRO_HEARTBEAT_SEC:-30}"
(
  while true; do
    sleep "$HEARTBEAT_SEC"
    if [[ -f "${OUT_DIR%/}/predictions.jsonl" ]]; then
      n_lines="$(wc -l < "${OUT_DIR%/}/predictions.jsonl" 2>/dev/null || echo 0)"
      echo "Heartbeat: predictions.jsonl lines=$n_lines"
    else
      echo "Heartbeat: waiting (no predictions.jsonl yet)"
    fi
  done
) &
HEARTBEAT_PID=$!

set +e
uv run python -u - <<'PY'
import faulthandler
import os
from pathlib import Path

from beans_next.cli import main

# If the process stalls (e.g. dataset backend hang), emit periodic stack traces to stderr.
timeout_s = int(os.environ.get("BEANS_PRO_FAULTHANDLER_TIMEOUT_SEC", "300"))
faulthandler.enable()
faulthandler.dump_traceback_later(timeout_s, repeat=True)

args_file = Path(os.environ["ARGS_FILE"])
argv = [line.rstrip("\n") for line in args_file.read_text(encoding="utf-8").splitlines()]
exit_code = int(main(argv=argv))
os._exit(exit_code)
PY
exit_code=$?
set -e

kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true
wait "$HEARTBEAT_PID" >/dev/null 2>&1 || true
exit "$exit_code"

echo "Run complete. Artifacts in: $OUT_DIR"

# Optional: copy artifacts into $HOME so they're visible on this repo host via /mnt/home.
#
# Why: /scratch is node-local and may require manual rsync to inspect from this machine. If you
# copy the results into $HOME, they become visible under:
#   local host: /mnt/home/$USER/...
#   slurm nodes: /home/$USER/...
#
# Enable:
#   BEANS_PRO_COPY_RESULTS_TO_HOME=1
# Optional overrides:
#   BEANS_PRO_RESULTS_HOME_DIR=/home/$USER/beans-next-results/ingested
if [[ "${BEANS_PRO_COPY_RESULTS_TO_HOME:-0}" == "1" ]]; then
  RESULTS_HOME_DIR="${BEANS_PRO_RESULTS_HOME_DIR:-$HOME/beans-next-results/ingested}"
  DEST_DIR="${RESULTS_HOME_DIR}/${RUN_ID}"
  mkdir -p "$DEST_DIR"

  # Prefer rsync; fall back to cp if rsync isn't available.
  if command -v rsync >/dev/null 2>&1; then
    rsync -av --delete "${OUT_DIR%/}/" "${DEST_DIR%/}/"
  else
    rm -rf "$DEST_DIR"
    mkdir -p "$DEST_DIR"
    cp -a "${OUT_DIR%/}/." "$DEST_DIR/"
  fi

  echo "Copied artifacts to: $DEST_DIR"
  echo "Local host view (via NFS): /mnt${DEST_DIR}"
fi

