#!/usr/bin/env bash
# Reproduce a paper workflow config, keeping launcher servers warm across tasks.
#
# DESIGN.md §2.3: iterate paper table rows by model, start each server once,
# run all its tasks, then shut it down. For the side-by-side NatureLM config,
# this means starting both servers, running BEANS-Zero for each (best-effort),
# then shutting both down.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat >&2 <<'EOF'
usage: scripts/reproduce_all.sh <config_yaml>

Example:
  HF_TOKEN=hf_... scripts/reproduce_all.sh configs/paper/beans_zero_naturelm_side_by_side.yaml

Notes:
  - This script is defensive: it will skip models whose /health does not come up.
  - For naturelm-v1.1, HF_TOKEN is required unless NATURELM_STUB_MODE=1.

Environment (optional):
  LIMIT=5                  max samples per eval task (default 5)
  RUN_ID=...               base run id (default: reproduce-all-<config-stem>)
  OUT_DIR=results/<run_id> artifact directory root (default: results/<run_id>)
EOF
  exit 2
}

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]] || [[ $# -lt 1 ]]; then
  usage
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "ERROR: missing required command: ${name}" >&2
    exit 1
  fi
}

require_cmd uv

LIMIT="${LIMIT:-5}"

port_in_use() {
  local port="$1"
  uv run python - <<PY
import socket, sys
host = "127.0.0.1"
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.2)
rc = s.connect_ex((host, port))
s.close()
sys.exit(0 if rc == 0 else 1)
PY "$port"
}

ensure_launcher_env() {
  local dir="$1"
  if [[ -f "${dir}/pyproject.toml" ]]; then
    (cd "$dir" && uv sync)
    return
  fi
  if [[ -f "${dir}/requirements.txt" ]]; then
    (cd "$dir" && uv venv && uv pip install -r requirements.txt)
    return
  fi
  echo "ERROR: launcher has neither pyproject.toml nor requirements.txt: ${dir}" >&2
  exit 1
}

wait_for_health() {
  local url="$1"
  local seconds="${2:-45}"
  uv run python - <<PY
import sys, time
from urllib.request import urlopen

url = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
last = None
while time.time() < deadline:
    try:
        with urlopen(url, timeout=1.0) as resp:
            status = int(getattr(resp, "status", 200))
            if status == 200:
                raise SystemExit(0)
            last = f"HTTP {status}"
    except Exception as e:
        last = repr(e)
    time.sleep(0.2)
print(f"not healthy; last_err={last}", file=sys.stderr)
raise SystemExit(1)
PY "$url" "$seconds"
}

parse_config() {
  local path="$1"
  uv run python - <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1]).resolve()
doc = yaml.safe_load(cfg_path.read_text("utf-8"))
if not isinstance(doc, dict):
    print(f"ERROR: config root must be a mapping, got {type(doc).__name__}", file=sys.stderr)
    sys.exit(2)

suite = doc.get("suite", "beans_zero_core")
models = doc.get("models", {})
if not isinstance(models, dict) or not models:
    print("ERROR: config must define `models:` mapping.", file=sys.stderr)
    sys.exit(2)

print(f"SUITE={suite}")
for key, body in models.items():
    if not isinstance(key, str) or not isinstance(body, dict):
        continue
    launcher = body.get("launcher", {})
    if not isinstance(launcher, dict):
        launcher = {}
    workdir = launcher.get("workdir", "")
    port = launcher.get("port", "")
    kind = launcher.get("kind", key)
    serve_sh = launcher.get("serve_sh", "./serve.sh")
    print(f"MODEL={key}")
    print(f"KIND={kind}")
    print(f"WORKDIR={workdir}")
    print(f"SERVE_SH={serve_sh}")
    print(f"PORT={port}")
    print("ENDMODEL=1")
PY "$path"
}

RUN_ID_DEFAULT="reproduce-all-$(basename "$CONFIG_PATH" .yaml)"
RUN_ID="${RUN_ID:-$RUN_ID_DEFAULT}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/${RUN_ID}}"

mkdir -p "$OUT_DIR"

echo "## Config"
echo "- path: ${CONFIG_PATH}"
echo "- run_id: ${RUN_ID}"
echo "- out_dir: ${OUT_DIR}"
echo "- limit: ${LIMIT}"
echo

server_pids=()
cleanup() {
  set +e
  echo
  echo "## Cleanup"
  for pid in "${server_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "- stopping pid=${pid}"
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${server_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      sleep 0.2
    fi
  done
  for pid in "${server_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "- forcing pid=${pid}"
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

cfg_lines="$(parse_config "$CONFIG_PATH")"
eval "$cfg_lines"

healthy_models=()

current_model=""
current_kind=""
current_workdir=""
current_serve_sh=""
current_port=""

start_one_model() {
  local model="$1"
  local kind="$2"
  local workdir="$3"
  local serve_sh="$4"
  local port="$5"

  if [[ -z "$workdir" ]]; then
    echo "ERROR: model ${model} missing launcher.workdir in config." >&2
    return 1
  fi
  local abs_dir="$REPO_ROOT/$workdir"
  if [[ ! -d "$abs_dir" ]]; then
    echo "ERROR: model ${model} launcher directory not found: ${abs_dir}" >&2
    return 1
  fi
  if [[ ! -x "$abs_dir/${serve_sh#./}" ]] && [[ ! -x "$abs_dir/$serve_sh" ]]; then
    echo "ERROR: model ${model} serve script not executable: ${abs_dir}/${serve_sh}" >&2
    return 1
  fi

  if [[ -z "$port" ]]; then
    echo "ERROR: model ${model} missing launcher.port in config." >&2
    return 1
  fi

  if port_in_use "$port"; then
    echo "ERROR: port ${port} already in use; cannot start model ${model} (${kind})." >&2
    return 1
  fi

  if [[ "$kind" == "naturelm-v1.1" ]] && [[ "${NATURELM_STUB_MODE:-}" != "1" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARN: skipping model ${model} (${kind}): HF_TOKEN not set (and NATURELM_STUB_MODE!=1)." >&2
    return 0
  fi

  echo "## Start launcher: ${model} (${kind})"
  ensure_launcher_env "$abs_dir"

  local host="127.0.0.1"
  local bind_env=""
  case "$kind" in
    dummy) bind_env="DUMMY_BIND_HOST" ;;
    naturelm-v1.0) bind_env="NATURELM_V1_0_BIND_HOST" ;;
    naturelm-v1.1) bind_env="NATURELM_BIND_HOST" ;;
    *) bind_env="" ;;
  esac

  (
    cd "$abs_dir"
    if [[ -n "$bind_env" ]]; then
      PORT="$port" "${bind_env}=${host}" bash "$serve_sh"
    else
      PORT="$port" bash "$serve_sh"
    fi
  ) &
  local pid="$!"
  server_pids+=("$pid")
  echo "- pid: ${pid}"

  local base_url="http://${host}:${port}"
  if wait_for_health "${base_url}/health" 60; then
    echo "- health: OK (${base_url})"
    uv run bash "$REPO_ROOT/scripts/check_launcher.sh" "$base_url"
    healthy_models+=("${model}::${base_url}")
  else
    echo "WARN: model ${model} did not become healthy; leaving it running for inspection and skipping." >&2
  fi
  echo
}

while IFS= read -r line; do
  case "$line" in
    SUITE=*) SUITE="${line#SUITE=}" ;;
    MODEL=*) current_model="${line#MODEL=}" ;;
    KIND=*) current_kind="${line#KIND=}" ;;
    WORKDIR=*) current_workdir="${line#WORKDIR=}" ;;
    SERVE_SH=*) current_serve_sh="${line#SERVE_SH=}" ;;
    PORT=*) current_port="${line#PORT=}" ;;
    ENDMODEL=1)
      start_one_model "$current_model" "$current_kind" "$current_workdir" "$current_serve_sh" "$current_port"
      current_model=""; current_kind=""; current_workdir=""; current_serve_sh=""; current_port=""
      ;;
  esac
done <<<"$cfg_lines"

if [[ "${#healthy_models[@]}" -eq 0 ]]; then
  echo "ERROR: no healthy models started; nothing to run." >&2
  exit 1
fi

echo "## Run suite (best-effort)"
echo "- suite: ${SUITE}"
echo

for entry in "${healthy_models[@]}"; do
  model="${entry%%::*}"
  base_url="${entry#*::}"
  predict_url="${base_url}/predict"

  model_run_id="${RUN_ID}__${model}"
  model_out="${OUT_DIR}/${model}"
  mkdir -p "$model_out"

  echo "### Model: ${model}"
  echo "- predict_url: ${predict_url}"
  echo "- run_id: ${model_run_id}"
  echo "- out_dir: ${model_out}"
  echo

  uv run beans-next run \
    --predict-url "$predict_url" \
    --suite "$SUITE" \
    --limit "$LIMIT" \
    --run-id "$model_run_id" \
    -o "$model_out"

  echo
done

echo "## Done"

