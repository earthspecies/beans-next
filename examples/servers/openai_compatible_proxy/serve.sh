#!/usr/bin/env bash
# Start the BEANS-Next OpenAI-compatible proxy launcher (FastAPI + uvicorn).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
export PORT

# Default to a conservative but useful batch size for both local and Slurm usage.
# Override by setting OPENAI_PROXY_MAX_BATCH_SIZE explicitly.
export OPENAI_PROXY_MAX_BATCH_SIZE="${OPENAI_PROXY_MAX_BATCH_SIZE:-8}"

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage:"
  echo "  PORT=8000 OPENAI_PROXY_STUB=1 ./serve.sh                   # stub / conformance"
  echo "  PORT=8000 OPENAI_PROXY_STUB=0 OPENAI_BASE_URL=... ./serve.sh  # real proxy"
  echo ""
  echo "Environment:"
  echo "  OPENAI_PROXY_BIND_HOST       bind address (default: 127.0.0.1)"
  echo "  OPENAI_PROXY_STUB            1=stub mode (default), 0=real proxy mode"
  echo "  OPENAI_BASE_URL              upstream base URL (required in proxy mode)"
  echo "  OPENAI_API_KEY               upstream API key"
  echo "  GEMINI_API_KEY               Gemini API key fallback for Gemini upstreams"
  echo "  GOOGLE_API_KEY               Gemini API key fallback for Gemini upstreams"
  echo "  OPENAI_MODEL                 upstream model id"
  echo "  OPENAI_MODEL_REVISION        model revision reported by /info"
  echo "  OPENAI_PROXY_TIMEOUT_SEC     upstream request timeout in seconds (default: 120)"
  echo "  OPENAI_PROXY_RETRIES         retry attempts for transient errors (default: 2)"
  echo "  OPENAI_PROXY_MAX_BATCH_SIZE  max items per /predict call (default: 32)"
  echo "  OPENAI_PROXY_MAX_CONCURRENCY max upstream calls in flight per batch (default: 1)"
  echo "  GEMINI_MIN_MAX_TOKENS        min max_tokens for Gemini upstreams (default: 1024)"
  echo "  GEMINI_REASONING_EFFORT      thinking level for Gemini upstreams: none (default), low, medium, high"
  echo ""
  echo "Examples:"
  echo ""
  echo "  # GPT-4o-audio-preview (OpenAI API):"
  echo "  OPENAI_PROXY_STUB=0 \\"
  echo "    OPENAI_BASE_URL=https://api.openai.com \\"
  echo "    OPENAI_API_KEY=sk-... \\"
  echo "    OPENAI_MODEL=gpt-4o-audio-preview \\"
  echo "    PORT=8000 ./serve.sh"
  echo ""
  echo "  # Gemini via Google OpenAI-compatible endpoint:"
  echo "  OPENAI_PROXY_STUB=0 \\"
  echo "    OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \\"
  echo "    OPENAI_API_KEY=<google-ai-studio-key> \\"
  echo "    OPENAI_MODEL=gemini-3.1-pro-preview \\"
  echo "    OPENAI_PROXY_CANONICALIZE_WAV=1 \\"
  echo "    OPENAI_PROXY_MAX_AUDIO_SECONDS=30 \\"
  echo "    PORT=8000 ./serve.sh"
  exit 0
fi

_trim_ws() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

_read_cfg_api_key() {
  local path="$1"
  shift

  [[ -f "$path" ]] || return 1

  local line key val accepted
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(_trim_ws "$line")"
    [[ -n "$line" && "${line:0:1}" != "#" ]] || continue

    if [[ "${line,,}" == export\ * ]]; then
      line="$(_trim_ws "${line:7}")"
    fi

    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*[:=][[:space:]]*(.+)[[:space:]]*$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="$(_trim_ws "${BASH_REMATCH[2]}")"
      val="${val%\'}"
      val="${val#\'}"
      val="${val%\"}"
      val="${val#\"}"
      [[ -n "$val" ]] || continue

      if [[ "${key,,}" == "api_key" ]]; then
        printf '%s' "$val"
        return 0
      fi

      for accepted in "$@"; do
        if [[ "$key" == "$accepted" ]]; then
          printf '%s' "$val"
          return 0
        fi
      done
    elif [[ "$line" != *":"* && "$line" != *"="* ]]; then
      val="$line"
      val="${val%\'}"
      val="${val#\'}"
      val="${val%\"}"
      val="${val#\"}"
      if [[ -n "$val" ]]; then
        printf '%s' "$val"
        return 0
      fi
    fi
  done < "$path"

  return 1
}

_is_gemini_upstream() {
  local base_url="${OPENAI_BASE_URL:-}"
  local model="${OPENAI_MODEL:-}"
  [[ "${base_url,,}" == *"generativelanguage.googleapis.com"* || "${model,,}" == gemini-* ]]
}

# Load provider API keys from protected cfg files if not already set (same
# convention as ~/.config/huggingface/hf_token and ~/.config/github/github_token).
if [[ -z "${OPENAI_API_KEY:-}" ]] && _is_gemini_upstream; then
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$GEMINI_API_KEY"
  elif [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$GOOGLE_API_KEY"
  elif key="$(_read_cfg_api_key "${GEMINI_CFG_PATH:-$HOME/.config/gemini/cfg}" GEMINI_API_KEY GOOGLE_API_KEY)"; then
    export OPENAI_API_KEY="$key"
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]] && key="$(_read_cfg_api_key "${OPENAI_CFG_PATH:-$HOME/.config/openai/cfg}" OPENAI_API_KEY)"; then
  export OPENAI_API_KEY="$key"
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn serve:app --host "${OPENAI_PROXY_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn serve:app --host "${OPENAI_PROXY_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
echo "Or from repo root with uv:" >&2
echo "  cd $ROOT && uv venv && uv pip install -r requirements.txt && . .venv/bin/activate && ./serve.sh" >&2
exit 1
