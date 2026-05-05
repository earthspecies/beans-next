#!/usr/bin/env bash
# SLURM job: run beans-next inference against a running launcher.
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
#
# CPUs: 8 supports BEANS_PRO_ESP_DATA_WORKERS=8 parallel GCS downloads (3× speedup
# vs sequential; see scripts/bench/bench_beans_zero_load.py --full-audio --workers 8).
# Raise to 16+ if the node has enough CPUs and GCS throughput still bottlenecks.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next inference"

# This script does not request GPUs (inference is remote — handled by the serving job).

set -euo pipefail

# ---------------------------------------------------------------------------
# BEANS_NEXT_* environment variables (preferred) + BEANS_PRO_* compatibility.
#
# Policy:
# - All new workflows should use BEANS_NEXT_*.
# - Legacy BEANS_PRO_* variables are still accepted as fallbacks.
# ---------------------------------------------------------------------------

_env_first() {
  local new_name="$1"
  local old_name="$2"
  local default_val="$3"
  local v="${!new_name:-}"
  if [[ -n "$v" ]]; then
    printf '%s' "$v"
    return 0
  fi
  v="${!old_name:-}"
  if [[ -n "$v" ]]; then
    printf '%s' "$v"
    return 0
  fi
  printf '%s' "$default_val"
}

_export_compat() {
  local new_name="$1"
  local old_name="$2"
  local default_val="${3:-}"
  # shellcheck disable=SC2163
  export "${new_name}=$(_env_first "$new_name" "$old_name" "$default_val")"
}

# Canonical knobs (exported so subprocesses inherit them).
_export_compat "BEANS_NEXT_DEBUG" "BEANS_PRO_DEBUG" "0"
_export_compat "BEANS_NEXT_SKIP_UV_SYNC" "BEANS_PRO_SKIP_UV_SYNC" "0"
_export_compat "BEANS_NEXT_DATA_SOURCE" "BEANS_PRO_DATA_SOURCE" "esp_data"
_export_compat "BEANS_NEXT_URL_FILE" "BEANS_PRO_URL_FILE" ""
_export_compat "BEANS_NEXT_URL_WAIT_TIMEOUT_SEC" "BEANS_PRO_URL_WAIT_TIMEOUT_SEC" "1800"
_export_compat "BEANS_NEXT_URL_WAIT_INTERVAL_SEC" "BEANS_PRO_URL_WAIT_INTERVAL_SEC" "5"
_export_compat "BEANS_NEXT_HEALTH_TIMEOUT_SEC" "BEANS_PRO_HEALTH_TIMEOUT_SEC" "1800"
_export_compat "BEANS_NEXT_HEALTH_INTERVAL_SEC" "BEANS_PRO_HEALTH_INTERVAL_SEC" "5"
_export_compat "BEANS_NEXT_HEALTH_CONNECT_TIMEOUT_SEC" "BEANS_PRO_HEALTH_CONNECT_TIMEOUT_SEC" "2"
_export_compat "BEANS_NEXT_HEALTH_MAX_TIME_SEC" "BEANS_PRO_HEALTH_MAX_TIME_SEC" "10"
_export_compat "BEANS_NEXT_HTTP_TIMEOUT_SEC" "BEANS_PRO_HTTP_TIMEOUT_SEC" ""
_export_compat "BEANS_NEXT_UV_PYTHON" "BEANS_PRO_UV_PYTHON" "3.11"
_export_compat "BEANS_NEXT_HARD_EXIT" "BEANS_PRO_HARD_EXIT" "1"
_export_compat "BEANS_NEXT_COPY_RESULTS_TO_HOME" "BEANS_PRO_COPY_RESULTS_TO_HOME" "1"
_export_compat "BEANS_NEXT_RESULTS_HOME_DIR" "BEANS_PRO_RESULTS_HOME_DIR" ""
_export_compat "BEANS_NEXT_UPLOAD_GCS" "BEANS_PRO_UPLOAD_GCS" "1"
_export_compat "BEANS_NEXT_GCS_PREFIX" "BEANS_PRO_GCS_PREFIX" ""
_export_compat "BEANS_NEXT_GCS_REL_PATH" "BEANS_PRO_GCS_REL_PATH" ""
_export_compat "BEANS_NEXT_RUN_KIND" "BEANS_PRO_RUN_KIND" ""
_export_compat "BEANS_NEXT_CONFIG" "BEANS_PRO_CONFIG" ""
_export_compat "BEANS_NEXT_TASK_ID" "BEANS_PRO_TASK_ID" ""
_export_compat "BEANS_NEXT_DATASET_NAME" "BEANS_PRO_DATASET_NAME" ""
_export_compat "BEANS_NEXT_SUITE" "BEANS_PRO_SUITE" ""
_export_compat "BEANS_NEXT_SPLIT" "BEANS_PRO_SPLIT" ""
_export_compat "BEANS_NEXT_HF_PATH" "BEANS_PRO_HF_PATH" ""
_export_compat "BEANS_NEXT_HF_CONFIG" "BEANS_PRO_HF_CONFIG" ""
_export_compat "BEANS_NEXT_LIMIT" "BEANS_PRO_LIMIT" ""
_export_compat "BEANS_NEXT_RUN_ID" "BEANS_PRO_RUN_ID" ""
_export_compat "BEANS_NEXT_OUT_DIR" "BEANS_PRO_OUT_DIR" ""
_export_compat "BEANS_NEXT_RESUME" "BEANS_PRO_RESUME" "0"
_export_compat "BEANS_NEXT_INFERENCE_WORKERS" "BEANS_PRO_INFERENCE_WORKERS" "1"
_export_compat "BEANS_NEXT_ESP_DATA_WORKERS" "BEANS_PRO_ESP_DATA_WORKERS" ""
_export_compat "BEANS_NEXT_HF_WORKERS" "BEANS_PRO_HF_WORKERS" ""
_export_compat "BEANS_NEXT_DEBUG_FAULTHANDLER_SEC" "BEANS_PRO_DEBUG_FAULTHANDLER_SEC" "300"

# Optional debug mode.
# Enable with: BEANS_NEXT_DEBUG=1 (or true/yes).
_debug_enabled() {
  case "${BEANS_NEXT_DEBUG:-0}" in
    1|true|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

_ts() {
  date -Is 2>/dev/null || date
}

_step() {
  echo "$(_ts) [beans-next][inference] $*"
}

# In Slurm, scripts are copied to a spool dir before execution, so $0 does not point at the repo.
# Use the submission directory as the repo root (we submit from the repo root in all runbooks).
REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Ensure all `uv` operations use a job-scoped environment on node-local scratch.
#
# NOTE: this is intentionally node-local for performance; it can be overridden at submit
# time if the node has a small disk (e.g. set UV_PROJECT_ENVIRONMENT under $HOME on NFS).
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/beans-next-infer-${SLURM_JOB_ID}}"

# HuggingFace caches:
# - Default to NFS-backed $HOME so jobs don't fail on small /scratch disks.
# - You can still override at submit time if you want node-local speed.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Temporary files (HF Parquet → WAV materialization writes temp WAVs).
# Keep these on NFS by default to avoid filling node-local disks.
export TMPDIR="${TMPDIR:-$HOME/.cache/beans-next/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
mkdir -p "$TMPDIR"

# Audio materialization cache (shared, NFS-backed by default).
export BEANS_NEXT_HF_AUDIO_CACHE_DIR="$(_env_first "BEANS_NEXT_HF_AUDIO_CACHE_DIR" "BEANS_PRO_HF_AUDIO_CACHE_DIR" "$HF_HOME/beans-next-audio")"
export BEANS_NEXT_ESP_AUDIO_CACHE_DIR="$(_env_first "BEANS_NEXT_ESP_AUDIO_CACHE_DIR" "BEANS_PRO_ESP_AUDIO_CACHE_DIR" "$HF_HOME/beans-next-audio")"

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
URL_FILE_DEFAULT="$HOME/beans-next-launchers/${SLURM_JOB_ID}.url"
URL_FILE="${BEANS_NEXT_URL_FILE:-$URL_FILE_DEFAULT}"
if [[ -z "$URL_FILE" ]]; then
  echo "ERROR: BEANS_NEXT_URL_FILE must be set to the serving job URL file." >&2
  echo "Compat: BEANS_PRO_URL_FILE is also supported." >&2
  exit 1
fi
export BEANS_NEXT_URL_FILE="$URL_FILE"

# Run parameters (override via env before sbatch).
SUITE="${BEANS_NEXT_SUITE:-beans_zero_core}"
LIMIT="${BEANS_NEXT_LIMIT:-}"
OUT_DIR="${BEANS_NEXT_OUT_DIR:-}"
CONFIG="${BEANS_NEXT_CONFIG:-}"           # optional: path to a run config YAML
RUN_ID="${BEANS_NEXT_RUN_ID:-}"
RUN_KIND="${BEANS_NEXT_RUN_KIND:-full}"  # smoke | full | adhoc (free-form)
TASK_ID="${BEANS_NEXT_TASK_ID:-}"         # optional: run one eval task directly
DATASET_NAME="${BEANS_NEXT_DATASET_NAME:-}"  # optional: e.g. esc50 (required when TASK_ID set)
SPLIT="${BEANS_NEXT_SPLIT:-}"             # optional: default is CLI default (test)
HF_PATH="${BEANS_NEXT_HF_PATH:-}"         # optional: override hub dataset id
HF_CONFIG="${BEANS_NEXT_HF_CONFIG:-}"     # optional: override hub config name
# Resume policy: when enabled, beans-next skips eval tasks that already have summary.json.
RESUME="${BEANS_NEXT_RESUME:-0}"
# Inference HTTP workers: parallel /predict calls to the model server.
# Keep <= number of concurrent requests the model server can handle.
INFERENCE_WORKERS="${BEANS_NEXT_INFERENCE_WORKERS:-1}"

cd "$REPO"

if _debug_enabled; then
  export PYTHONUNBUFFERED=1
  _step "DEBUG enabled (BEANS_NEXT_DEBUG=1)"
  _step "Repo: $REPO"
  _step "URL_FILE: $URL_FILE"
  _step "OUT_DIR (preflight): ${OUT_DIR:-<unset>}"
  _step "RUN_ID (preflight): ${RUN_ID:-<unset>}"
  _step "TASK_ID: ${TASK_ID:-<unset>} DATASET_NAME: ${DATASET_NAME:-<unset>} SUITE: $SUITE"
  _step "INFERENCE_WORKERS: $INFERENCE_WORKERS  ESP_DATA_WORKERS: ${BEANS_NEXT_ESP_DATA_WORKERS:-<unset>}"
  _step "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
  # Bash trace for script-level flow (avoid leaking secrets; we don't print env wholesale).
  export PS4='+[$(_ts)] ${BASH_SOURCE##*/}:${LINENO}: '
  set -x
fi

# Dataset backend selection:
# - Default to esp_data everywhere.
# - Override by exporting BEANS_NEXT_DATA_SOURCE=esp_data|huggingface|hf before `sbatch`.
# (Compat: BEANS_PRO_DATA_SOURCE is also accepted.)

# Parallel GCS download workers for esp_data (BEANS_NEXT_ESP_DATA_WORKERS).
# Benchmarks on Slurm CPU nodes show ~3× speedup at workers=8 vs sequential
# (scripts/bench/bench_beans_zero_load.py --full-audio --workers 8).
# Default: match #SBATCH --cpus-per-task (8). Set to 1 to disable parallelism.
if [[ "${BEANS_NEXT_DATA_SOURCE:-esp_data}" == "esp_data" ]]; then
  export BEANS_NEXT_ESP_DATA_WORKERS="${BEANS_NEXT_ESP_DATA_WORKERS:-8}"
fi

# Parallel WAV-materialization workers for HF map-style loading (BEANS_NEXT_HF_WORKERS).
# soundfile.write releases the GIL, so threads give real speedup on WAV encoding.
# Default: match #SBATCH --cpus-per-task (8). Set to 1 to disable parallelism.
if [[ "${BEANS_NEXT_DATA_SOURCE:-esp_data}" == "huggingface" || "${BEANS_NEXT_DATA_SOURCE:-esp_data}" == "hf" ]]; then
  export BEANS_NEXT_HF_WORKERS="${BEANS_NEXT_HF_WORKERS:-8}"
fi

# Work around rare interpreter-finalization crashes seen on this cluster by hard-exiting
# the CLI after producing outputs (can be overridden).
export BEANS_NEXT_HARD_EXIT="${BEANS_NEXT_HARD_EXIT:-1}"

# Prefer Python 3.11+ for this project. Some compute nodes may not have a compatible system
# interpreter, so allow uv to download a managed Python when needed.
# Override if needed: BEANS_NEXT_UV_PYTHON=3.11 (default).
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_NEXT_UV_PYTHON:-3.11}"

# Ensure the job-scoped environment exists.
# If esp_data is requested, try to inherit any site-provided esp_data install via system site-packages.
if [[ ! -x "${UV_PROJECT_ENVIRONMENT%/}/bin/python" ]]; then
  if [[ "${BEANS_NEXT_DATA_SOURCE:-esp_data}" == "esp_data" ]]; then
    uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT"
  else
    uv venv "$UV_PROJECT_ENVIRONMENT"
  fi
fi

# On clusters where compute nodes cannot reach the package index, `uv sync` inside the job may fail.
# Set `BEANS_NEXT_SKIP_UV_SYNC=1` if you have pre-built the environment on a shared filesystem.
#
# If `esp_data` is selected, include the `esp` dependency group (configured in pyproject.toml).
if [[ "${BEANS_NEXT_SKIP_UV_SYNC:-0}" != "1" ]]; then
  if [[ "${BEANS_NEXT_DATA_SOURCE:-esp_data}" == "esp_data" ]]; then
    uv sync --group esp
  else
    uv sync
  fi
else
  echo "BEANS_NEXT_SKIP_UV_SYNC=1 set; skipping 'uv sync'"
fi

if [[ "${BEANS_NEXT_DATA_SOURCE:-esp_data}" == "esp_data" ]]; then
  set +e
  # Run the import check inside the same uv-managed interpreter environment this job uses.
  # Without an explicit --python, `uv run` may use a different interpreter/env and false-negative.
  esp_data_import_out="$(uv run --python "$UV_PROJECT_ENVIRONMENT" python -c "import esp_data" 2>&1)"
  has_esp_data=$?
  set -e
  if [[ "$has_esp_data" != "0" ]]; then
    # Note: we do not run `uv pip install` here. If esp_data isn't present, add it to the
    # project's `esp` dependency group (via `uv add --group esp esp-data`) and rerun.
    echo "ERROR: BEANS_NEXT_DATA_SOURCE=esp_data but 'esp_data' is not importable in this job environment." >&2
    if [[ -n "${esp_data_import_out:-}" ]]; then
      echo "ERROR: esp_data import output:" >&2
      echo "${esp_data_import_out}" >&2
    fi
    echo "Fix options:" >&2
    echo "  - Ensure this job ran 'uv sync --group esp' successfully (default when BEANS_NEXT_DATA_SOURCE=esp_data)." >&2
    echo "  - Ensure your 'pyproject.toml' has an 'esp' dependency group with 'esp-data', and 'tool.uv.index'/'tool.uv.sources' configured for esp-pypi." >&2
    echo "  - Or force HuggingFace loading: export BEANS_NEXT_DATA_SOURCE=huggingface" >&2
    exit 1
  fi
fi

# Wait for the URL file to appear (serving job may still be loading weights).
echo "Waiting for URL file: $URL_FILE"
URL_WAIT_TIMEOUT_SEC="${BEANS_NEXT_URL_WAIT_TIMEOUT_SEC:-1800}"
URL_WAIT_INTERVAL_SEC="${BEANS_NEXT_URL_WAIT_INTERVAL_SEC:-5}"
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
export BASE_URL

# ---------------------------------------------------------------------------
# Proxy bypass for cluster-internal launcher URLs.
#
# Some Slurm environments set HTTP(S)_PROXY for outbound traffic. Python's urllib
# honors those variables by default, which can break internal HTTP calls to the
# launcher (e.g., GET /info) even when curl-based health checks succeed.
#
# We defensively add the launcher's host to NO_PROXY/no_proxy so urllib bypasses
# proxies for this internal endpoint.
# ---------------------------------------------------------------------------
_launcher_hostport="${BASE_URL#*://}"
_launcher_hostport="${_launcher_hostport%%/*}"
_launcher_host="${_launcher_hostport%%:*}"
if [[ -n "${_launcher_host:-}" ]]; then
  _no_proxy_base="${NO_PROXY:-${no_proxy:-}}"
  _no_proxy_list="${_no_proxy_base},${_launcher_host},${_launcher_hostport},localhost,127.0.0.1,metadata.google.internal"
  # Normalize and export both variants (common across tools).
  export NO_PROXY="${_no_proxy_list#,}"
  export no_proxy="${NO_PROXY}"
fi
HEALTH_TIMEOUT_SEC="${BEANS_NEXT_HEALTH_TIMEOUT_SEC:-900}"
HEALTH_INTERVAL_SEC="${BEANS_NEXT_HEALTH_INTERVAL_SEC:-5}"
HEALTH_CONNECT_TIMEOUT_SEC="${BEANS_NEXT_HEALTH_CONNECT_TIMEOUT_SEC:-2}"
HEALTH_MAX_TIME_SEC="${BEANS_NEXT_HEALTH_MAX_TIME_SEC:-5}"
_poll_health \
  "$BASE_URL" \
  "$HEALTH_TIMEOUT_SEC" \
  "$HEALTH_INTERVAL_SEC" \
  "$HEALTH_CONNECT_TIMEOUT_SEC" \
  "$HEALTH_MAX_TIME_SEC"

# Preflight: verify python urllib can reach /info from this node.
# This matches the code path used by HttpClient.probe_info() and helps catch
# environment-specific connectivity issues early (e.g., proxies, resolver quirks).
uv run python - <<'PY'
import os
import time
from urllib.parse import urljoin
from urllib.request import getproxies, urlopen

base = os.environ["BASE_URL"].rstrip("/")
url = urljoin(base + "/", "info")

print(f"Preflight python /info: url={url}")
print(f"Preflight python getproxies()={getproxies()}")

last_exc: Exception | None = None
for attempt in range(1, 6):
    try:
        with urlopen(url, timeout=10) as resp:
            # Read a small prefix only; this is a connectivity check.
            _ = resp.read(256)
            print(f"Preflight python /info OK (attempt={attempt}, status={resp.status})")
            last_exc = None
            break
    except Exception as exc:  # noqa: BLE001 - diagnostic preflight
        last_exc = exc
        print(f"Preflight python /info failed (attempt={attempt}): {exc!r}")
        time.sleep(min(2 * attempt, 5))

if last_exc is not None:
    raise SystemExit(2)
PY

# ---------------------------------------------------------------------------
# Standardized run id + output directory policy
#
# Goal: durable artifact paths that remain interpretable without Slurm context:
# - include run kind: smoke/full
# - include a stable model directory derived from /info
# - include the "run definition": suite name OR config basename OR task id
#
# Output dir default:
#   /scratch/$USER/.cache/beans-next-results/<INC>/<MODEL_DIR>/<RUN_DEF>/<RUN_ID>
#
# Where:
# - INC defaults to "adhoc" (set BEANS_NEXT_INCREMENT to override; compat: BEANS_PRO_INCREMENT)
# - RUN_DEF is derived from BEANS_NEXT_CONFIG, BEANS_NEXT_TASK_ID, or BEANS_NEXT_SUITE
# - RUN_ID is generated if BEANS_NEXT_RUN_ID is unset
# ---------------------------------------------------------------------------
if [[ -z "${OUT_DIR:-}" || -z "${RUN_ID:-}" ]]; then
  run_identity="$(
    uv run python - <<'PY'
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urljoin
from urllib.request import urlopen


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


base = os.environ["BASE_URL"].rstrip("/")
info_url = urljoin(base + "/", "info")

info: dict[str, object] = {}
with urlopen(info_url, timeout=10) as resp:
    info = json.loads(resp.read().decode("utf-8"))

name = str(info.get("name") or "")
model = str(info.get("model") or "")

model_dir = _slug(name) if name else _slug(model)

run_kind = _slug(os.environ.get("BEANS_NEXT_RUN_KIND", os.environ.get("BEANS_PRO_RUN_KIND", "full")))
suite = os.environ.get("BEANS_NEXT_SUITE", os.environ.get("BEANS_PRO_SUITE", "beans_zero_core"))
task_id = os.environ.get("BEANS_NEXT_TASK_ID", os.environ.get("BEANS_PRO_TASK_ID", ""))
config = os.environ.get("BEANS_NEXT_CONFIG", os.environ.get("BEANS_PRO_CONFIG", ""))

run_def = ""
if config:
    run_def = _slug(os.path.splitext(os.path.basename(config))[0])
elif task_id:
    run_def = _slug(f"task_{task_id}")
else:
    run_def = _slug(suite)

ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get("SLURM_JOB_ID", "nojob")

run_id = os.environ.get("BEANS_NEXT_RUN_ID", os.environ.get("BEANS_PRO_RUN_ID", "")).strip()
if not run_id:
    run_id = f"{run_kind}_{model_dir}_{run_def}_{ts}_j{job_id}"

out_dir = os.environ.get("BEANS_NEXT_OUT_DIR", os.environ.get("BEANS_PRO_OUT_DIR", "")).strip()
if not out_dir:
    inc = _slug(os.environ.get("BEANS_NEXT_INCREMENT", os.environ.get("BEANS_PRO_INCREMENT", "adhoc")))
    scratch_root = f"/scratch/{os.environ.get('USER', 'unknown')}/.cache/beans-next-results"
    out_dir = f"{scratch_root}/{inc}/{model_dir}/{run_def}/{run_id}"

print(run_id)
print(out_dir)
PY
  )"
  mapfile -t _run_identity_lines <<<"$run_identity"
  if [[ -z "${RUN_ID:-}" && -n "${_run_identity_lines[0]:-}" ]]; then
    RUN_ID="$(_trim_ws "${_run_identity_lines[0]}")"
  fi
  if [[ -z "${OUT_DIR:-}" && -n "${_run_identity_lines[1]:-}" ]]; then
    OUT_DIR="$(_trim_ws "${_run_identity_lines[1]}")"
  fi
fi

# Ensure output directory exists before launching the runner.
mkdir -p "${OUT_DIR%/}"

if _debug_enabled; then
  _step "RUN_KIND: $RUN_KIND"
  _step "RUN_ID: $RUN_ID"
  _step "OUT_DIR: $OUT_DIR"
fi

if _debug_enabled; then
  set +e
  _step "Fetching /info for debug"
  curl -fsS --connect-timeout 2 --max-time 10 "${BASE_URL%/}/info" || true
  set -e
fi

# Build CLI args.
CLI_ARGS=(
  run
  --predict-url "$PREDICT_URL"
  --run-id "$RUN_ID"
  -o "$OUT_DIR"
  --workers "$INFERENCE_WORKERS"
)

# Respect dataset backend selection for registry-driven eval tasks.
# Without this, per-task `data_source` fields in eval-task YAMLs override
# BEANS_NEXT_DATA_SOURCE, making Slurm runs hard to redirect between backends.
#
# CLI uses `--backend` with choices: esp_data, huggingface, hf.
case "${BEANS_NEXT_DATA_SOURCE:-esp_data}" in
  esp_data|huggingface|hf)
    CLI_ARGS+=(--backend "${BEANS_NEXT_DATA_SOURCE}")
    ;;
  *)
    echo "ERROR: Unsupported BEANS_NEXT_DATA_SOURCE=${BEANS_NEXT_DATA_SOURCE@Q}. Use esp_data, huggingface, or hf." >&2
    exit 1
    ;;
esac

case "${RESUME}" in
  1|true|TRUE|yes|YES)
    CLI_ARGS+=(--resume)
    ;;
  *) ;;
esac

if [[ -n "$CONFIG" ]]; then
  CLI_ARGS+=(--config "$CONFIG")
elif [[ -n "$TASK_ID" ]]; then
  CLI_ARGS+=(--task-id "$TASK_ID")
  if [[ -z "$DATASET_NAME" ]]; then
    echo "ERROR: BEANS_NEXT_TASK_ID was set but BEANS_NEXT_DATASET_NAME is empty." >&2
    echo "Example: BEANS_NEXT_TASK_ID=beans_zero_esc50 BEANS_NEXT_DATASET_NAME=esc50" >&2
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

# Debug heartbeat: periodically report predictions file growth, if any.
if _debug_enabled; then
  (
    pred="${OUT_DIR%/}/predictions.jsonl"
    while true; do
      if [[ -f "$pred" ]]; then
        n="$(wc -l <"$pred" 2>/dev/null || echo 0)"
        _step "heartbeat: predictions.jsonl lines=$n"
      else
        _step "heartbeat: predictions.jsonl not created yet"
      fi
      sleep "${BEANS_NEXT_DEBUG_HEARTBEAT_SEC:-30}"
    done
  ) &
  _HEARTBEAT_PID="$!"
  trap 'kill "${_HEARTBEAT_PID:-}" 2>/dev/null || true' EXIT
fi

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

uv run python - <<'PY'
import os
from pathlib import Path

from beans_next.cli import main

debug = os.environ.get("BEANS_NEXT_DEBUG", os.environ.get("BEANS_PRO_DEBUG", "")).lower() in {"1", "true", "yes"}
if debug:
    import faulthandler

    faulthandler.enable()
    # Periodic traceback for stuck jobs (default 5 minutes).
    # Keep interval configurable; this emits to stderr (captured by Slurm logs).
    interval_sec = int(os.environ.get("BEANS_NEXT_DEBUG_FAULTHANDLER_SEC", os.environ.get("BEANS_PRO_DEBUG_FAULTHANDLER_SEC", "300")))
    faulthandler.dump_traceback_later(interval_sec, repeat=True)

args_file = Path(os.environ["ARGS_FILE"])
argv = [line.rstrip("\n") for line in args_file.read_text(encoding="utf-8").splitlines()]

# Ensure output directory exists even if the wrapper script didn't create it
# (some clusters sanitize job env or delay NFS visibility).
try:
    out_idx = argv.index("-o")
except ValueError:
    out_idx = -1
if out_idx != -1 and out_idx + 1 < len(argv):
    out_dir = Path(argv[out_idx + 1]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

exit_code = int(main(argv=argv))
os._exit(exit_code)
PY

echo "Run complete. Artifacts in: $OUT_DIR"

# Optional: copy artifacts into $HOME so they're visible on this repo host via /mnt/home.
#
# Why: /scratch is node-local and may require manual rsync to inspect from this machine. If you
# copy the results into $HOME, they become visible under:
#   local host: /mnt/home/$USER/...
#   slurm nodes: /home/$USER/...
#
# Enable (default is enabled for this sweep):
#   BEANS_NEXT_COPY_RESULTS_TO_HOME=1
# Optional overrides:
#   BEANS_NEXT_RESULTS_HOME_DIR=/home/$USER/beans-next-results/ingested
# For the BEANS-Zero sweep we always want durable artifacts on NFS.
if [[ "${BEANS_NEXT_COPY_RESULTS_TO_HOME:-1}" == "1" ]]; then
  RESULTS_HOME_DIR="${BEANS_NEXT_RESULTS_HOME_DIR:-$HOME/beans-next-results/ingested}"
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

  # -------------------------------------------------------------------------
  # Always upload to GCS for this sweep (durable sharing / reprocessing).
  #
  # Policy: upload under:
  #   gs://foundation-model-data/synthetic/predictions/beans-next-results/<REL_PATH>/
  #
  # Where REL_PATH is relative to:
  #   /scratch/$USER/.cache/beans-next-results/
  #
  # Example:
  #   i29/naturelm_v1_0/beans_zero_core/<RUN_ID>
  # -------------------------------------------------------------------------
  if [[ "${BEANS_NEXT_UPLOAD_GCS:-1}" == "1" ]]; then
    if command -v gsutil >/dev/null 2>&1; then
      SCRATCH_ROOT="/scratch/$USER/.cache/beans-next-results/"
      rel_path="$RUN_ID"
      if [[ "${OUT_DIR%/}/" == "${SCRATCH_ROOT}"* ]]; then
        rel_path="${OUT_DIR#${SCRATCH_ROOT}}"
        rel_path="${rel_path%/}"
      fi

      GCS_PREFIX_BASE="${BEANS_NEXT_GCS_PREFIX_BASE:-gs://foundation-model-data/synthetic/predictions/beans-next-results}"
      GCS_DEST="${GCS_PREFIX_BASE%/}/${rel_path}/"
      echo "Uploading artifacts to: ${GCS_DEST}"
      gsutil -m rsync -r "${DEST_DIR%/}/" "${GCS_DEST}"
      echo "Uploaded artifacts to: ${GCS_DEST}"
    else
      echo "WARNING: gsutil not found; skipping GCS upload (set BEANS_NEXT_UPLOAD_GCS=0 to silence)." >&2
    fi
  fi
fi
