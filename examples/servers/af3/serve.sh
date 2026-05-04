#!/usr/bin/env bash
# Start the BEANS-Next Tier-2 af3 reference launcher (FastAPI + uvicorn).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-19084}"
export PORT

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage:"
  echo "  PORT=19084 AF3_STUB=1 ./serve.sh          # stub / conformance mode (CPU)"
  echo "  PORT=19084 ./serve.sh                      # real inference (GPU required)"
  echo ""
  echo "Environment:"
  echo "  AF3_BIND_HOST         bind address (default: 127.0.0.1)"
  echo "  AF3_STUB=1            stub-only conformance mode; no model weights loaded"
  echo "  AF3_MAX_BATCH_SIZE    max batch size (default: 4)"
  echo "  AF3_MODEL             HuggingFace model id (default: nvidia/audio-flamingo-next-hf)"
  echo "  AF3_MODEL_REVISION    HuggingFace revision / git ref (default: main)"
  echo ""
  echo "Real inference requires a GPU and the ML deps in requirements.txt:"
  echo "  torch>=2.2.0, transformers>=4.47.0, accelerate>=0.27.0, soundfile"
  echo ""
  echo "License note:"
  echo "  nvidia/audio-flamingo-next-hf is released under the NVIDIA OneWay"
  echo "  Noncommercial License. Only non-commercial research use is permitted."
  exit 0
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn serve:app --host "${AF3_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn serve:app --host "${AF3_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
echo "Or from repo root with uv:" >&2
echo "  cd $ROOT && uv venv && uv pip install -r requirements.txt && . .venv/bin/activate && ./serve.sh" >&2
exit 1

