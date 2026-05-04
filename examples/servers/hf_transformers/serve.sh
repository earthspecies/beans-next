#!/usr/bin/env bash
# Start the BEANS-Next Tier-2 hf_transformers reference launcher (FastAPI + uvicorn).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-19083}"
export PORT

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage:"
  echo "  PORT=19083 HF_TRANSFORMERS_STUB=1 ./serve.sh"
  echo ""
  echo "Environment:"
  echo "  HF_TRANSFORMERS_BIND_HOST       bind address (default: 127.0.0.1)"
  echo "  HF_TRANSFORMERS_STUB=1          stub-only conformance mode (recommended)"
  echo "  HF_TRANSFORMERS_MAX_BATCH_SIZE  max batch size (default: 8)"
  echo "  HF_TRANSFORMERS_MODEL           model id string to advertise in /info"
  echo "  HF_TRANSFORMERS_MODEL_REVISION  model revision string to advertise in /info"
  exit 0
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn serve:app --host "${HF_TRANSFORMERS_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn serve:app --host "${HF_TRANSFORMERS_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
echo "Or from repo root with uv:" >&2
echo "  cd $ROOT && uv venv && uv pip install -r requirements.txt && . .venv/bin/activate && ./serve.sh" >&2
exit 1

