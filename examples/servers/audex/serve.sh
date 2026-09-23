#!/usr/bin/env bash
# Start the BEANS-Next Audex adapter sidecar (FastAPI + uvicorn).
# The upstream model server (vllm serve with the audex_30b_a3b_vllm plugin
# editable-installed) runs separately.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
export PORT

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage:"
  echo "  PORT=8000 AUDEX_ADAPTER_STUB=1 ./serve.sh                 # stub / conformance"
  echo "  PORT=8000 AUDEX_ADAPTER_STUB=0 \\"
  echo "    AUDEX_UPSTREAM_BASE_URL=http://127.0.0.1:8001 ./serve.sh  # real proxy"
  echo ""
  echo "Environment:"
  echo "  AUDEX_ADAPTER_BIND_HOST       bind address (default: 127.0.0.1)"
  echo "  AUDEX_ADAPTER_STUB            1=stub mode (default), 0=real proxy mode"
  echo "  AUDEX_UPSTREAM_BASE_URL       vllm serve base URL (required in proxy mode)"
  echo "  AUDEX_MODEL_ID                model id forwarded upstream (served-model-name)"
  echo "  AUDEX_MODEL_REVISION          revision reported by /info (default: unknown)"
  echo "  AUDEX_MAX_BATCH_SIZE          max items per /predict call (default: 32)"
  echo "  AUDEX_MAX_CONCURRENCY         concurrent upstream calls per batch (default: 8)"
  echo "  AUDEX_UPSTREAM_TIMEOUT_SEC    upstream request timeout in seconds (default: 120)"
  echo "  AUDEX_UPSTREAM_RETRIES        retry attempts for transient errors (default: 1)"
  echo "  AUDEX_EXTRA_BODY_JSON         JSON object merged into the upstream request body"
  echo "                                (chat_template_kwargs, temperature, top_p, logit_bias)"
  echo ""
  echo "Typical flow:"
  echo ""
  echo "  # 1. Start the model (separate terminal, GPU required, plugin venv active):"
  echo "  vllm serve nvidia/Nemotron-Labs-Audex-30B-A3B \\"
  echo "    --served-model-name audex_audio --tensor-parallel-size 8 \\"
  echo "    --gpu-memory-utilization 0.85 --max-model-len 32768 --trust-remote-code \\"
  echo "    --limit-mm-per-prompt '{\"audio\": 1}' --allowed-local-media-path <root> \\"
  echo "    --host 127.0.0.1 --port 8001"
  echo ""
  echo "  # 2. Start this adapter sidecar:"
  echo "  AUDEX_ADAPTER_STUB=0 \\"
  echo "    AUDEX_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \\"
  echo "    AUDEX_MODEL_ID=audex_audio \\"
  echo "    AUDEX_EXTRA_BODY_JSON='{\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0.0}' \\"
  echo "    PORT=8000 ./serve.sh"
  echo ""
  echo "License note:"
  echo "  nvidia/Nemotron-Labs-Audex-30B-A3B is released under the NVIDIA OneWay"
  echo "  Noncommercial License. Only non-commercial research use is permitted."
  exit 0
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn serve:app --host "${AUDEX_ADAPTER_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn serve:app --host "${AUDEX_ADAPTER_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  cd $ROOT && uv sync && uv run ./serve.sh" >&2
exit 1
