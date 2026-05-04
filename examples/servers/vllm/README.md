# BEANS-Next launcher: `vllm` (adapter sidecar)

Tier-1 launcher that bridges `vllm serve` to the BEANS-Next `predictions_v1` HTTP contract. It translates each `/predict` request item into an OpenAI-compatible `POST /v1/chat/completions` call and converts the response back.

Works with any model served by vLLM that exposes an OpenAI-compat API, including:

- **Qwen3-Omni** (`Qwen/Qwen3-Omni-30B-A3B-Instruct`) — omni model with native audio understanding
- **Qwen2-Audio** (`Qwen/Qwen2-Audio-7B-Instruct`)
- Any other audio/multimodal model vLLM supports

The adapter sidecar is **GPU-free**: run in stub mode for conformance without vLLM installed.

## Endpoints

- `POST /predict` — batched inference via upstream vLLM
- `GET /info` — capability document
- `GET /health` — readiness probe

## Stub mode (CPU-only conformance)

`VLLM_ADAPTER_STUB=1` (default) returns deterministic placeholder predictions without calling any upstream.

```bash
PORT=8000 VLLM_ADAPTER_STUB=1 ./serve.sh
```

## Real proxy mode

### Qwen3-Omni

Before running Qwen3-Omni, agents must read the Qwen notes in `examples/servers/af3/README.md`.
They record the current upstream commands, the known `python -m vllm` failure, and the non-vLLM fallback.

```bash
# 1. Start vLLM (separate terminal, GPU required):
uv sync --group upstream
uv run vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --allowed-local-media-path /

# 2. Start the adapter sidecar:
VLLM_ADAPTER_STUB=0 \
  VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  PORT=8000 ./serve.sh
```

Then run BEANS-Next:

```bash
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --limit 10
```

### Qwen2-Audio

```bash
uv sync --group upstream
uv run vllm serve Qwen/Qwen2-Audio-7B-Instruct --host 127.0.0.1 --port 8001

VLLM_ADAPTER_STUB=0 \
  VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  VLLM_MODEL_ID=Qwen/Qwen2-Audio-7B-Instruct \
  PORT=8000 ./serve.sh
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_ADAPTER_BIND_HOST` | `127.0.0.1` | Bind address |
| `VLLM_ADAPTER_STUB` | `1` | `1`=stub mode, `0`=proxy mode |
| `VLLM_UPSTREAM_BASE_URL` | — | vLLM base URL (required in proxy mode) |
| `VLLM_MODEL_ID` | `vllm/unknown` | Model id forwarded upstream |
| `VLLM_MODEL_REVISION` | `unknown` | Revision reported by `/info` |
| `VLLM_MAX_BATCH_SIZE` | `32` | Max items per `/predict` call |
| `VLLM_UPSTREAM_TIMEOUT_SEC` | `30` | Request timeout (seconds) |
| `VLLM_UPSTREAM_RETRIES` | `1` | Retries for transient errors |
| `PORT` | `8000` | Adapter listen port |

### Qwen3-Omni audio capping (recommended)

Qwen Omni models can be sensitive to long audio inputs (multimodal token pressure can exceed
`--max-model-len`, especially on Watkins). The adapter supports an **OpenAI-proxy-equivalent**
“canonicalize + clip” step for `base64_wav` inputs:

- `VLLM_ADAPTER_MAX_AUDIO_SECONDS` (float, e.g. `30`)
- `VLLM_ADAPTER_CANONICALIZE_WAV` (`1`/`0`)

When enabled, the adapter re-encodes to canonical PCM16 WAV and clips to the first N seconds.

## Audio handling

Audio must be `base64_wav`. The adapter currently attaches it as `{"type": "input_audio", ...}` content items on the last user message.
For Qwen3-Omni, upstream vLLM serve examples use OpenAI-compatible `audio_url` items, so a Qwen run that reaches vLLM but returns request-schema errors should update `adapter.py` to translate BEANS-Next `base64_wav` into an `audio_url` data URL or a temporary WAV file under `--allowed-local-media-path`.

## Setup

```bash
cd examples/servers/vllm
uv sync
```

## Conformance check

```bash
uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/vllm --cmd-cwd . \
  --app adapter:app --host 127.0.0.1 --port 8000 \
  --env VLLM_ADAPTER_STUB=1 \
  -- uv run bash scripts/check_launcher.sh "http://127.0.0.1:8000"
```
