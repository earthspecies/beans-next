# BEANS-Next launcher: `openai_compatible_proxy`

Tier-1 launcher that implements the BEANS-Next `predictions_v1` HTTP contract by proxying to any **OpenAI-compatible** `POST /v1/chat/completions` endpoint. Works with:

- **GPT-4o-audio-preview** (and other OpenAI audio models) via `api.openai.com`
- **Gemini** models via Google's OpenAI-compatible endpoint
- **LiteLLM**, **Azure OpenAI**, or any other OpenAI-compat proxy

This launcher does **not** import `beans_next` and has no heavy ML dependencies.

## Endpoints

- `POST /predict` — batched inference; returns HTTP 413 if batch exceeds `max_batch_size`
- `GET /info` — capability document
- `GET /health` — readiness probe

## Stub mode (CPU-only conformance)

`OPENAI_PROXY_STUB=1` (default) returns deterministic placeholder predictions. No upstream calls are made. Use this to run `scripts/check_launcher.sh` without an API key.

```bash
PORT=8000 OPENAI_PROXY_STUB=1 ./serve.sh
```

## Real proxy mode

### Recommended: store your API key in `~/.config/openai/cfg`

To keep secrets out of shell history and Slurm scripts, the launcher will automatically load
`OPENAI_API_KEY` from `~/.config/openai/cfg` if `OPENAI_API_KEY` is not already set in the
environment.

Create the protected file:

```bash
mkdir -p "$HOME/.config/openai"
chmod 700 "$HOME/.config/openai"

printf "sk-...\n" >"$HOME/.config/openai/cfg"

chmod 600 "$HOME/.config/openai/cfg"
```

Notes:

- If you *do* set `OPENAI_API_KEY` in the environment, it takes precedence.
- You can override the config path with `OPENAI_CFG_PATH=/path/to/cfg`.

### Recommended for Gemini: store your API key in `~/.config/gemini/cfg`

For Gemini upstreams, identified by a Gemini model id or Google's OpenAI-compatible base URL,
the launcher also loads a bare token, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` from
`~/.config/gemini/cfg` when `OPENAI_API_KEY` is not already set.

```bash
mkdir -p "$HOME/.config/gemini"
chmod 700 "$HOME/.config/gemini"

printf "AIza...\n" >"$HOME/.config/gemini/cfg"

chmod 600 "$HOME/.config/gemini/cfg"
```

You can override the config path with `GEMINI_CFG_PATH=/path/to/cfg`.

### GPT-4o-audio-preview

```bash
OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://api.openai.com \
  OPENAI_MODEL=gpt-4o-audio-preview \
  PORT=8000 ./serve.sh
```

Then run BEANS-Next against it:

```bash
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --limit 10
```

### Gemini (Google AI Studio)

Gemini exposes an OpenAI-compatible endpoint. Get a key at [aistudio.google.com](https://aistudio.google.com).

```bash
OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
  OPENAI_MODEL=gemini-3.1-pro-preview \
  OPENAI_PROXY_CANONICALIZE_WAV=1 \
  OPENAI_PROXY_MAX_AUDIO_SECONDS=30 \
  PORT=8000 ./serve.sh
```

Other Gemini model ids: `gemini-2.5-pro`, `gemini-2.5-flash-lite`, etc.

#### Practical limitations (Gemini)

Gemini models can be **significantly slower and more expensive** than local GPU launchers. In practice:

- **Latency**: a single audio+text request can take tens of seconds (or more), depending on audio length and thinking budget.
- **Cost**: running large sweeps (e.g. 100×11 = 1100 requests) can be costly.
- **Stability**: long-running jobs can fail if the local proxy is stopped/restarted (then clients see `Connection refused`).

**Recommended settings for long runs**

- Run the proxy with bounded work per request:
  - `OPENAI_PROXY_MAX_CONCURRENCY=1`
  - `OPENAI_PROXY_RETRIES=1`
  - `OPENAI_PROXY_TIMEOUT_SEC=180` (or higher if you still see upstream timeouts)
  - `OPENAI_PROXY_CANONICALIZE_WAV=1`
  - `OPENAI_PROXY_MAX_AUDIO_SECONDS=10` (reduce cost/latency; increase if needed)
  - `GEMINI_MIN_MAX_TOKENS=2048` (prevents empty/short responses when thinking is enabled upstream)
- Run BEANS-Next with a larger client timeout:
  - `BEANS_PRO_HTTP_TIMEOUT_SEC=1800`

**Recommended sweep strategy (Pairs)**

For Gemini, prefer smaller, resumable batches:

```bash
# Example: 10 samples per subset, but run 1–2 subsets at a time.
BEANS_PRO_HTTP_TIMEOUT_SEC=1800 uv run beans-next pairs \
  --predict-url http://127.0.0.1:19085/predict \
  --model-tag gemini_3_1_pro_preview \
  --k 10 \
  --subsets crow-description,call-type-fixed-vocab
```

If a run is interrupted, re-run the same command with the same `--output-dir` and add `--resume`
to skip already-collected sample ids and continue appending to `pairs.jsonl`.

### Using a `--config` YAML

With the `openai_proxy_local_8000` registry preset (`beans_next/registry/model/openai_proxy_local_8000.yaml`):

```yaml
# my_run.yaml
model: openai_proxy_local_8000
suite: beans_zero_core
limit: 20
```

```bash
uv run beans-next run --config my_run.yaml
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_PROXY_BIND_HOST` | `127.0.0.1` | Bind address |
| `OPENAI_PROXY_STUB` | `1` | `1`=stub mode, `0`=proxy mode |
| `OPENAI_BASE_URL` | — | Upstream base URL (required in proxy mode) |
| `OPENAI_API_KEY` | — | API key (optional if stored in provider cfg) |
| `GEMINI_API_KEY` | — | Gemini API key fallback for Gemini upstreams |
| `GOOGLE_API_KEY` | — | Gemini API key fallback for Gemini upstreams |
| `OPENAI_CFG_PATH` | `~/.config/openai/cfg` | Optional path to a protected cfg file containing a bare OpenAI token or `OPENAI_API_KEY=...` |
| `GEMINI_CFG_PATH` | `~/.config/gemini/cfg` | Optional path to a protected cfg file containing a bare Gemini token, `GEMINI_API_KEY=...`, or `GOOGLE_API_KEY=...` |
| `OPENAI_MODEL` | `openai-compatible/unknown` | Model id sent upstream |
| `OPENAI_MODEL_REVISION` | `unknown` | Revision reported by `/info` |
| `OPENAI_AUTH_HEADER` | `Authorization` | Auth header name |
| `OPENAI_PROXY_TIMEOUT_SEC` | `120` | Request timeout (seconds) |
| `OPENAI_PROXY_RETRIES` | `2` | Retries for transient errors |
| `GEMINI_MIN_MAX_TOKENS` | `1024` | Minimum `max_tokens` enforced for Gemini upstreams; thinking models (2.5 Pro, 3.1 Pro-preview) consume thinking tokens from this budget — too low a value leaves zero tokens for the response |
| `GEMINI_REASONING_EFFORT` | `none` | Thinking level for Gemini upstreams (`none` disables thinking, `low`/`medium`/`high` enables it at increasing cost and latency) |
| `OPENAI_PROXY_MAX_BATCH_SIZE` | `32` | Max items per `/predict` call |
| `OPENAI_PROXY_MAX_CONCURRENCY` | `1` | Max upstream Chat Completions calls in flight per `/predict` batch |
| `PORT` | `8000` | Listen port |

## Audio handling

Audio must be sent as `base64_wav`. The launcher converts each audio input to an `{"type": "input_audio", ...}` content item appended to the last user message — compatible with GPT-4o-audio and Gemini's audio understanding.

`file_path` and `file_url` payload types are not forwarded (those request items return a per-sample error). Set `audio_payload: base64_wav` in your registry YAML or run config.

## Setup

```bash
cd examples/servers/openai_compatible_proxy
uv venv && uv pip install -r requirements.txt
. .venv/bin/activate
```

## Conformance check

```bash
uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/openai_compatible_proxy --cmd-cwd . \
  --app serve:app --host 127.0.0.1 --port 8000 \
  --env OPENAI_PROXY_STUB=1 \
  -- uv run bash scripts/check_launcher.sh "http://127.0.0.1:8000"
```
