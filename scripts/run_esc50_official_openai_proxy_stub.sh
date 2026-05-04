#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run ESC-50 official minimal benchmark against openai_compatible_proxy in stub mode.

Owns the server lifecycle (start -> readiness poll -> run -> stop) via scripts/with_uvicorn.py.

Usage:
  scripts/run_esc50_official_openai_proxy_stub.sh [-o OUTPUT_DIR]

Options:
  -o, --output-dir DIR  Output directory for run artifacts. If omitted, a
                        deterministic directory under results/ is chosen.
  -h, --help            Show this help.

What it runs (inside the managed server lifecycle):
  1) uv run bash scripts/check_launcher.sh http://127.0.0.1:19085
  2) uv run beans-next run --predict-url http://127.0.0.1:19085/predict \
       --task-id beans_zero_esc50_official --limit 1 -o <output_dir>
EOF
}

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -o|--output-dir)
      shift
      OUTPUT_DIR="${1-}"
      if [[ -z "$OUTPUT_DIR" ]]; then
        echo "error: --output-dir requires a value" >&2
        exit 2
      fi
      shift
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

choose_output_dir() {
  local base="$1"
  local i
  local candidate
  mkdir -p "$base"
  for i in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000; do
    candidate="$base/run_$(printf '%04d' "$i")"
    if [[ ! -e "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "error: could not choose an unused run dir under: $base" >&2
  return 1
}

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(choose_output_dir "$REPO_ROOT/results/esc50_official_openai_proxy_stub")"
else
  mkdir -p "$OUTPUT_DIR"
fi

BASE_URL="http://127.0.0.1:19085"
PREDICT_URL="$BASE_URL/predict"
READY_URL="$BASE_URL/health"

cd "$REPO_ROOT"

uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/openai_compatible_proxy \
  --cmd-cwd "$REPO_ROOT" \
  --app serve:app \
  --host 127.0.0.1 \
  --port 19085 \
  --ready-url "$READY_URL" \
  --env OPENAI_PROXY_STUB=1 \
  --env PORT=19085 \
  -- uv run bash -lc "set -euo pipefail; uv run bash scripts/check_launcher.sh \"$BASE_URL\"; uv run beans-next run --predict-url \"$PREDICT_URL\" --data-source hf --task-id beans_zero_esc50_official --limit 1 --output-dir \"$OUTPUT_DIR\""

echo "ok: wrote artifacts to $OUTPUT_DIR"

