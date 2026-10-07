# BEANS-Next launcher: `vllm` (adapter sidecar)

This adapter connects an OpenAI-compatible model server to the BEANS-Next `predictions_v1` HTTP contract. It translates each `/predict` request into a `POST /v1/chat/completions` call and converts the response back.

Use a model server that accepts text and audio content in chat messages.
Text-only evaluation requires a server that accepts messages without audio.

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

Start the upstream model in its own environment, following its runtime's serving instructions.
Set the adapter's model ID to the name advertised by the upstream `/v1/models` endpoint.
For Qwen servers that accept audio data URLs, use `VLLM_AUDIO_CONTENT_FORMAT=audio_url_data`.

```bash
cd examples/servers/vllm
uv sync
VLLM_ADAPTER_STUB=0 \
  VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  VLLM_MODEL_ID=<served-model-id> \
  VLLM_AUDIO_CONTENT_FORMAT=audio_url_data \
  PORT=8000 ./serve.sh
```

Then run BEANS-Next from the repository root:

```bash
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --limit 10
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
| `VLLM_AUDIO_CONTENT_FORMAT` | `input_audio` | `input_audio` or `audio_url_data`, according to the upstream API |
| `PORT` | `8000` | Adapter listen port |

### Qwen3-Omni audio capping (recommended)

Qwen Omni models can be sensitive to long audio inputs (multimodal token pressure can exceed
`--max-model-len`). The adapter can convert `base64_wav` inputs to PCM16 WAV and limit their duration:

- `VLLM_ADAPTER_MAX_AUDIO_SECONDS` (float, e.g. `30`)
- `VLLM_ADAPTER_CANONICALIZE_WAV` (`1`/`0`)

When enabled, the adapter re-encodes to canonical PCM16 WAV and clips to the first N seconds.

## Audio handling

Audio must be `base64_wav`. The adapter replaces each audio placeholder with its corresponding clip,
preserving the order of text, reference clips, and query audio.
`VLLM_AUDIO_CONTENT_FORMAT` selects either `input_audio` objects or `audio_url` data URLs.

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
  -- uv run python scripts/check_launcher.py "http://127.0.0.1:8000"
```
