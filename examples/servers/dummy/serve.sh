#!/usr/bin/env bash
# Start the BEANS-Next dummy launcher (FastAPI + uvicorn).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
export PORT

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage: PORT=8000 ./serve.sh"
  echo "  Requires a venv with dependencies from requirements.txt (see README.md)."
  exit 0
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn serve:app --host "${DUMMY_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn serve:app --host "${DUMMY_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
echo "Or from repo root with uv:" >&2
echo "  cd $ROOT && uv venv && uv pip install -r requirements.txt && . .venv/bin/activate && ./serve.sh" >&2
exit 1
