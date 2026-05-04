#!/usr/bin/env bash
# Reproduce one BEANS-Zero task (or small suite) for one launcher.
#
# DESIGN.md §2.3: start one launcher, wait for /health, conformance check,
# run `beans-next run`, then stop the server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat >&2 <<'EOF'
usage: scripts/reproduce.sh <launcher> <task>

Launchers:
  dummy | naturelm-v1.0 | naturelm-v1.1

Tasks:
  - Full eval-task ids, e.g.:
      beans_zero_esc50
      beans_zero_unseen_species_tax
  - Shorthands (paper prose), e.g.:
      esc50
      unseen-species-tax

Environment:
  HF_TOKEN                 required for naturelm-v1.1 (unless NATURELM_STUB_MODE=1)
  NATURELM_STUB_MODE=1     conformance-only mode for naturelm-v1.1 (no HF, no weights)

Common options (env vars):
  LIMIT=5                  max samples for the task (default 5)
  OUT_DIR=results/<run_id> artifact directory (default: results/<run_id>)
  RUN_ID=...               run id (default: reproduce-<launcher>-<task>)
  HOST=127.0.0.1           bind host (default 127.0.0.1)
  PORT=8000/8001           bind port (default depends on launcher)

Examples:
  HF_TOKEN=hf_... scripts/reproduce.sh naturelm-v1.1 unseen-species-tax
  scripts/reproduce.sh naturelm-v1.0 beans_zero_esc50
EOF
  exit 2
}

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]] || [[ $# -lt 2 ]]; then
  usage
fi

LAUNCHER="$1"
TASK_KEY="$2"

LIMIT="${LIMIT:-5}"
HOST="${HOST:-127.0.0.1}"

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "ERROR: missing required command: ${name}" >&2
    exit 1
  fi
}

require_cmd uv

launcher_dir=""
default_port=""
bind_host_env=""
case "$LAUNCHER" in
  dummy)
    launcher_dir="$REPO_ROOT/examples/servers/dummy"
    default_port="8000"
    bind_host_env="DUMMY_BIND_HOST"
    ;;
  naturelm-v1.0)
    launcher_dir="$REPO_ROOT/examples/servers/naturelm-v1.0"
    default_port="8000"
    bind_host_env="NATURELM_V1_0_BIND_HOST"
    ;;
  naturelm-v1.1)
    launcher_dir="$REPO_ROOT/examples/servers/naturelm-v1.1"
    default_port="8001"
    bind_host_env="NATURELM_BIND_HOST"
    if [[ "${NATURELM_STUB_MODE:-}" != "1" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
      echo "ERROR: HF_TOKEN is required for launcher ${LAUNCHER}." >&2
      echo "Hint: export HF_TOKEN=hf_... with access to the gated repo." >&2
      echo "Hint: or set NATURELM_STUB_MODE=1 for conformance-only mode." >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unknown launcher: ${LAUNCHER}" >&2
    usage
    ;;
esac

if [[ ! -d "$launcher_dir" ]]; then
  echo "ERROR: launcher directory not found: ${launcher_dir}" >&2
  exit 1
fi
if [[ ! -x "$launcher_dir/serve.sh" ]]; then
  echo "ERROR: launcher serve.sh not found or not executable: ${launcher_dir}/serve.sh" >&2
  exit 1
fi

PORT="${PORT:-$default_port}"

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

if port_in_use "$PORT"; then
  echo "ERROR: port ${PORT} is already in use on localhost." >&2
  echo "Hint: set PORT=... to a free port or stop the process using it." >&2
  exit 1
fi

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
  local seconds="${2:-30}"
  uv run python - <<PY
import sys, time
from urllib.error import URLError
from urllib.request import urlopen

url = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
last = None
while time.time() < deadline:
    try:
        with urlopen(url, timeout=1.0) as resp:
            status = int(getattr(resp, "status", 200))
            if status == 200:
                print("OK: launcher healthy")
                raise SystemExit(0)
            last = f"HTTP {status}"
    except Exception as e:
        last = repr(e)
    time.sleep(0.2)
print(f"FAIL: launcher not healthy within {sys.argv[2]}s; last_err={last}", file=sys.stderr)
raise SystemExit(1)
PY "$url" "$seconds"
}

resolve_eval_task() {
  local key="$1"
  uv run python - <<'PY'
import re
import sys
from pathlib import Path

import yaml

repo_root = Path(__file__).resolve().parents[2]
registry = repo_root / "beans_next" / "registry" / "eval_task"

key = sys.argv[1].strip()
aliases = {
    "esc50": "beans_zero_esc50",
    "cbi": "beans_zero_cbi",
    "watkins": "beans_zero_watkins",
    "humbugdb": "beans_zero_humbugdb",
    "lifestage": "beans_zero_lifestage",
    "call-type": "beans_zero_call_type",
    "captioning": "beans_zero_captioning",
    "enabirds": "beans_zero_enabirds",
    "hiceas": "beans_zero_hiceas",
    "rfcx": "beans_zero_rfcx",
    "gibbons": "beans_zero_gibbons",
    "dcase": "beans_zero_dcase",
    "zf-indiv": "beans_zero_zf_indiv",
    "unseen-species-cmn": "beans_zero_unseen_species_cmn",
    "unseen-species-sci": "beans_zero_unseen_species_sci",
    "unseen-species-tax": "beans_zero_unseen_species_tax",
    "unseen-genus-cmn": "beans_zero_unseen_genus_cmn",
    "unseen-genus-sci": "beans_zero_unseen_genus_sci",
    "unseen-genus-tax": "beans_zero_unseen_genus_tax",
    "unseen-family-cmn": "beans_zero_unseen_family_cmn",
    "unseen-family-sci": "beans_zero_unseen_family_sci",
    "unseen-family-tax": "beans_zero_unseen_family_tax",
}
task_id = aliases.get(key, key)
if not task_id.startswith("beans_zero_"):
    # tolerate the paper prose style without prefix.
    task_id = aliases.get(key, f"beans_zero_{key}")

path = registry / f"{task_id}.yaml"
if not path.is_file():
    print(f"ERROR: unknown eval task {key!r} (resolved {task_id!r}); expected {path}", file=sys.stderr)
    sys.exit(2)

doc = yaml.safe_load(path.read_text("utf-8"))
if isinstance(doc, dict) and len(doc) == 1:
    only_key = next(iter(doc.keys()))
    body = doc[only_key]
    if isinstance(only_key, str) and isinstance(body, dict):
        body = dict(body)
        body.setdefault("eval_task_id", only_key)
    else:
        body = dict(doc)
else:
    body = dict(doc or {})
body.setdefault("eval_task_id", path.stem)

hf_path = body.get("hf_path") or body.get("dataset_hf_path") or "EarthSpeciesProject/BEANS-Zero"
hf_config = body.get("hf_config") or body.get("config_name") or "BEANS-Zero"
split = body.get("split") or "test"
dataset_name = body.get("dataset_name") or body.get("dataset") or body.get("subset") or "esc50"
prompt = body.get("prompt_yaml") or body.get("prompt") or body.get("prompt_filename") or body.get("prompt_file") or ""

def norm_prompt(p: str) -> str:
    p = str(p).strip()
    if not p:
        return ""
    if p.endswith(".yaml"):
        return p
    return f"{p}.yaml"

prompt = norm_prompt(prompt)

# Print as shell assignments, one per line.
print(f"EVAL_TASK_ID={body.get('eval_task_id')}")
print(f"HF_PATH={hf_path}")
print(f"HF_CONFIG={hf_config}")
print(f"SPLIT={split}")
print(f"DATASET_NAME={dataset_name}")
print(f"PROMPT_YAML={prompt}")
PY "$key"
}

RUN_ID="${RUN_ID:-reproduce-${LAUNCHER}-${TASK_KEY}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/${RUN_ID}}"

BASE_URL="http://${HOST}:${PORT}"
PREDICT_URL="${BASE_URL}/predict"
HEALTH_URL="${BASE_URL}/health"

server_pid=""
cleanup() {
  set +e
  if [[ -n "${server_pid}" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
    echo
    echo "Stopping launcher (pid=${server_pid})"
    kill "$server_pid" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if kill -0 "$server_pid" >/dev/null 2>&1; then
        sleep 0.2
      else
        break
      fi
    done
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "Forcing launcher shutdown (pid=${server_pid})"
      kill -9 "$server_pid" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

echo "## Environment"
echo "- launcher: ${LAUNCHER}"
echo "- task: ${TASK_KEY}"
echo "- base_url: ${BASE_URL}"
echo "- predict_url: ${PREDICT_URL}"
echo "- out_dir: ${OUT_DIR}"
echo "- limit: ${LIMIT}"
echo

echo "## Prepare launcher environment"
ensure_launcher_env "$launcher_dir"

echo
echo "## Start launcher"
(
  cd "$launcher_dir"
  PORT="$PORT" "${bind_host_env}=${HOST}" bash ./serve.sh
) &
server_pid="$!"
echo "- pid: ${server_pid}"

echo
echo "## Wait for /health"
wait_for_health "$HEALTH_URL" 60

echo
echo "## Conformance check"
uv run bash "$REPO_ROOT/scripts/check_launcher.sh" "$BASE_URL"

echo
echo "## Resolve eval task"
task_kv="$(resolve_eval_task "$TASK_KEY")"
eval "$task_kv"
echo "- eval_task_id: ${EVAL_TASK_ID}"
echo "- hf_path: ${HF_PATH}"
echo "- hf_config: ${HF_CONFIG}"
echo "- split: ${SPLIT}"
echo "- dataset_name: ${DATASET_NAME}"
echo "- prompt_yaml: ${PROMPT_YAML:-<default>}"

echo
echo "## Run benchmark"
mkdir -p "$OUT_DIR"

cmd=(uv run beans-next run
  --predict-url "$PREDICT_URL"
  --hf-path "$HF_PATH"
  --hf-config "$HF_CONFIG"
  --split "$SPLIT"
  --dataset-name "$DATASET_NAME"
  --task-id "$EVAL_TASK_ID"
  --limit "$LIMIT"
  --run-id "$RUN_ID"
  -o "$OUT_DIR"
)
if [[ -n "${PROMPT_YAML:-}" ]]; then
  cmd+=(--prompt-yaml "$REPO_ROOT/beans_next/registry/prompt/${PROMPT_YAML}")
fi

echo "- Command: ${cmd[*]}"
"${cmd[@]}"

echo
echo "## Done"

