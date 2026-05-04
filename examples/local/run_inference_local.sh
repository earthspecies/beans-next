#!/usr/bin/env bash
# Local runner: run beans-next inference against a launcher running on Slurm.
#
# This script is intentionally parallel to `examples/slurm/run_inference.sh`, but:
# - runs on your local machine (where this repo is checked out)
# - reads the URL file from the shared NFS mount (`/mnt/home/...`) by default
#
# Prereqs:
# - You can reach the launcher URL from this machine (network / VPN / IAP, etc.)
# - You have `uv` available locally
#
# Typical usage (two-job pattern):
#   SERVE_JOB_ID=56464
#   BEANS_PRO_URL_FILE="/mnt/home/$USER/beans-next-launchers/${SERVE_JOB_ID}.url" \
#   BEANS_PRO_TASK_ID=beans_zero_esc50_official \
#   BEANS_PRO_DATASET_NAME=esc50 \
#   BEANS_PRO_LIMIT=5 \
#   BEANS_PRO_OUT_DIR="$PWD/beans-next-results/local_${SERVE_JOB_ID}" \
#   BEANS_PRO_RUN_ID="esc50_official_local_${SERVE_JOB_ID}" \
#   bash examples/local/run_inference_local.sh
#
# Notes:
# - If you want esp_data locally, you must have it installed (and credentials configured)
#   and run `uv sync --group esp`. Otherwise set: BEANS_PRO_DATA_SOURCE=hf

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

_pause() {
  local timeout_sec="$1"
  sleep "$timeout_sec"
}

_trim_ws() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
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

SERVE_JOB_ID="${SERVE_JOB_ID:-}"
DEFAULT_URL_FILE=""
if [[ -n "$SERVE_JOB_ID" ]]; then
  DEFAULT_URL_FILE="/mnt/home/$USER/beans-next-launchers/${SERVE_JOB_ID}.url"
fi

URL_FILE="${BEANS_PRO_URL_FILE:-$DEFAULT_URL_FILE}"
if [[ -z "$URL_FILE" ]]; then
  echo "ERROR: set BEANS_PRO_URL_FILE (or SERVE_JOB_ID) to the launcher URL file." >&2
  exit 1
fi

SUITE="${BEANS_PRO_SUITE:-beans_zero_core}"
LIMIT="${BEANS_PRO_LIMIT:-}"
OUT_DIR="${BEANS_PRO_OUT_DIR:-$REPO/beans-next-results/local_run}"
CONFIG="${BEANS_PRO_CONFIG:-}"
RUN_ID="${BEANS_PRO_RUN_ID:-local_run}"
TASK_ID="${BEANS_PRO_TASK_ID:-}"
DATASET_NAME="${BEANS_PRO_DATASET_NAME:-}"
SPLIT="${BEANS_PRO_SPLIT:-}"
HF_PATH="${BEANS_PRO_HF_PATH:-}"
HF_CONFIG="${BEANS_PRO_HF_CONFIG:-}"

# Local default: prefer HF unless user explicitly chooses esp_data.
if [[ -z "${BEANS_PRO_DATA_SOURCE:-}" && "$SUITE" =~ ^beans_zero_ ]]; then
  export BEANS_PRO_DATA_SOURCE="hf"
fi

export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_PRO_UV_PYTHON:-3.11}"

if [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
  if [[ "${BEANS_PRO_DATA_SOURCE:-hf}" == "esp_data" ]]; then
    uv sync --group esp
  else
    uv sync
  fi
else
  echo "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync'"
fi

echo "Waiting for URL file: $URL_FILE"
URL_WAIT_TIMEOUT_SEC="${BEANS_PRO_URL_WAIT_TIMEOUT_SEC:-1800}"
URL_WAIT_INTERVAL_SEC="${BEANS_PRO_URL_WAIT_INTERVAL_SEC:-2}"
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

BASE_URL="${PREDICT_URL%/predict}"
HEALTH_TIMEOUT_SEC="${BEANS_PRO_HEALTH_TIMEOUT_SEC:-900}"
HEALTH_INTERVAL_SEC="${BEANS_PRO_HEALTH_INTERVAL_SEC:-2}"
HEALTH_CONNECT_TIMEOUT_SEC="${BEANS_PRO_HEALTH_CONNECT_TIMEOUT_SEC:-2}"
HEALTH_MAX_TIME_SEC="${BEANS_PRO_HEALTH_MAX_TIME_SEC:-5}"
_poll_health \
  "$BASE_URL" \
  "$HEALTH_TIMEOUT_SEC" \
  "$HEALTH_INTERVAL_SEC" \
  "$HEALTH_CONNECT_TIMEOUT_SEC" \
  "$HEALTH_MAX_TIME_SEC"

mkdir -p "$OUT_DIR"

CLI_ARGS=(
  run
  --predict-url "$PREDICT_URL"
  --run-id "$RUN_ID"
  -o "$OUT_DIR"
)

if [[ -n "$CONFIG" ]]; then
  CLI_ARGS+=(--config "$CONFIG")
elif [[ -n "$TASK_ID" ]]; then
  CLI_ARGS+=(--task-id "$TASK_ID")
  if [[ -z "$DATASET_NAME" ]]; then
    echo "ERROR: BEANS_PRO_TASK_ID was set but BEANS_PRO_DATASET_NAME is empty." >&2
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
uv run beans-next "${CLI_ARGS[@]}"
echo "Run complete. Artifacts in: $OUT_DIR"

