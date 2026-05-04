#!/usr/bin/env bash
# Start the BEANS-Next vLLM adapter sidecar (FastAPI + uvicorn).
# The upstream model server (typically `vllm serve`) runs separately.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
export PORT

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage:"
  echo "  PORT=8000 VLLM_ADAPTER_STUB=1 ./serve.sh                 # stub / conformance"
  echo "  PORT=8000 VLLM_ADAPTER_STUB=0 \\"
  echo "    VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 ./serve.sh  # real proxy"
  echo ""
  echo "Environment:"
  echo "  VLLM_ADAPTER_BIND_HOST       bind address (default: 127.0.0.1)"
  echo "  VLLM_ADAPTER_STUB            1=stub mode (default), 0=real proxy mode"
  echo "  VLLM_UPSTREAM_BASE_URL       vllm serve base URL (required in proxy mode)"
  echo "  VLLM_MODEL_ID                model id forwarded upstream (e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct)"
  echo "  VLLM_MODEL_REVISION          revision reported by /info (default: unknown)"
  echo "  VLLM_MAX_BATCH_SIZE          max items per /predict call (default: 32)"
  echo "  VLLM_UPSTREAM_TIMEOUT_SEC    upstream request timeout in seconds (default: 30)"
  echo "  VLLM_UPSTREAM_RETRIES        retry attempts for transient errors (default: 1)"
  echo ""
  echo "Typical Qwen3-Omni flow:"
  echo ""
  echo "  # 1. Start the model (separate terminal, GPU required):"
  echo "  uv sync --group upstream"
  echo "  uv run vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \\"
  echo "    --host 127.0.0.1 --port 8001 --dtype bfloat16 \\"
  echo "    --max-model-len 32768 --allowed-local-media-path /"
  echo ""
  echo "  # 2. Start this adapter sidecar:"
  echo "  VLLM_ADAPTER_STUB=0 \\"
  echo "    VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \\"
  echo "    VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct \\"
  echo "    PORT=8000 ./serve.sh"
  exit 0
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn adapter:app --host "${VLLM_ADAPTER_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn adapter:app --host "${VLLM_ADAPTER_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  cd $ROOT && uv sync && uv run ./serve.sh" >&2
exit 1
