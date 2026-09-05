#!/usr/bin/env bash
# Start the BEANS-Next Gemma 4 12B adapter sidecar (FastAPI + uvicorn).
# The upstream model server (vllm serve google/gemma-4-12B-it, or the
# in-process transformers backend) runs separately / in-process depending on
# GEMMA4_BACKEND.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
export PORT

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage:"
  echo "  PORT=8000 GEMMA4_ADAPTER_STUB=1 ./serve.sh                 # stub / conformance"
  echo "  PORT=8000 GEMMA4_ADAPTER_STUB=0 GEMMA4_BACKEND=vllm \\"
  echo "    GEMMA4_UPSTREAM_BASE_URL=http://127.0.0.1:8001 ./serve.sh  # real proxy"
  echo "  PORT=8000 GEMMA4_ADAPTER_STUB=0 GEMMA4_BACKEND=transformers ./serve.sh"
  echo "                                                              # in-process fallback"
  echo ""
  echo "Environment:"
  echo "  GEMMA4_ADAPTER_BIND_HOST      bind address (default: 127.0.0.1)"
  echo "  GEMMA4_ADAPTER_STUB           1=stub mode (default), 0=real inference"
  echo "  GEMMA4_BACKEND                vllm (default) | transformers"
  echo "  GEMMA4_UPSTREAM_BASE_URL      vllm serve base URL (required for backend=vllm)"
  echo "  GEMMA4_MODEL_ID                model id forwarded upstream / loaded in-process"
  echo "                                (default: google/gemma-4-12B-it)"
  echo "  GEMMA4_MODEL_REVISION          revision reported by /info (default: unknown)"
  echo "  GEMMA4_MAX_BATCH_SIZE          max items per /predict call (default: 32)"
  echo "  GEMMA4_MAX_CONCURRENCY         concurrent upstream calls per batch (default: 8)"
  echo "  GEMMA4_UPSTREAM_TIMEOUT_SEC    upstream request timeout in seconds (default: 120)"
  echo "  GEMMA4_UPSTREAM_RETRIES        retry attempts for transient errors (default: 1)"
  echo "  GEMMA4_EXTRA_BODY_JSON         JSON object merged into the upstream request body"
  echo "                                (chat_template_kwargs, temperature, top_p, top_k)"
  echo "  GEMMA4_MAX_AUDIO_SECONDS       hard audio-length cap in seconds (default: 30)"
  echo "  GEMMA4_AUDIO_POSITION          after_text (default) | before_text"
  echo "  GEMMA4_TRIMMED_CACHE_DIR       on-disk cache dir for head-truncated clips"
  echo ""
  echo "Typical flow (vllm backend):"
  echo ""
  echo "  # 1. Start the model (separate terminal, GPU required, vllm[audio]>=0.23.0):"
  echo "  vllm serve google/gemma-4-12B-it \\"
  echo "    --served-model-name gemma4_12b --max-model-len 16384 \\"
  echo "    --gpu-memory-utilization 0.90 --trust-remote-code \\"
  echo "    --limit-mm-per-prompt '{\"image\": 4, \"audio\": 1}' \\"
  echo "    --reasoning-parser gemma4 --allowed-local-media-path <root> \\"
  echo "    --host 127.0.0.1 --port 8001"
  echo ""
  echo "  # 2. Start this adapter sidecar:"
  echo "  GEMMA4_ADAPTER_STUB=0 GEMMA4_BACKEND=vllm \\"
  echo "    GEMMA4_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \\"
  echo "    GEMMA4_MODEL_ID=gemma4_12b \\"
  echo "    GEMMA4_EXTRA_BODY_JSON='{\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0.0}' \\"
  echo "    PORT=8000 ./serve.sh"
  echo ""
  echo "License note:"
  echo "  google/gemma-4-12B-it is released under the Apache 2.0 license; no"
  echo "  gating or non-commercial restriction."
  exit 0
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn serve:app --host "${GEMMA4_ADAPTER_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn serve:app --host "${GEMMA4_ADAPTER_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  cd $ROOT && uv sync && uv run ./serve.sh" >&2
exit 1
