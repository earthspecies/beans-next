# BEANS-Next serving kit (`examples/servers/`)

Launchers are standalone HTTP servers that host a model and implement the BEANS-Next `predictions_v1` HTTP contract. The core library (`beans_next/`) never runs inference in-process — it only talks to launchers over HTTP.

## All launchers at a glance

| Launcher | Tier | Models | Audio in | GPU | Default port |
|---|---|---|---|---|---|
| `dummy` | 1 | Deterministic stub | any | No | 8000 |
| `naturelm-v1.0` | 1 | NatureLM-audio v1.0 | `base64_wav` | Yes (stub avail.) | 8000 |
| `naturelm-v1.1` | 1 | NatureLM-audio v1.1 | `base64_wav` | Yes (stub avail.) | 8001 |
| `openai_compatible_proxy` | 1 | GPT-4o-audio, Gemini, any OpenAI-compat | `base64_wav` | No (remote API) | 8000 |
| `vllm` | 1 | Qwen3-Omni, Qwen2-Audio, any vLLM model | `base64_wav` | Yes (GPU for vLLM) | 8000 |
| `af3` | 2 | Audio Flamingo Next (nvidia/audio-flamingo-next-hf) | `base64_wav`, `file_path`, `file_url` | Yes (stub avail.) | 19084 |
| `hf_transformers` | 2 | Generic HF Transformers starter | any | Depends on model | 19083 |

**Tier-1**: validated, conformance-tested, expected to be correct.
**Tier-2**: best-effort reference implementations; may drift as upstream APIs change.

## Mandatory endpoints (all launchers)

Every launcher implements:

- `POST /predict` — batched inference; HTTP 413 if batch exceeds `max_batch_size`
- `GET /info` — capability document (name, model, audio_payload_types, schema_versions, …)
- `GET /health` — readiness probe

Wire schema: **`predictions_v1`**.

## Quick start

### CPU-only smoke test (no GPU, no API keys)

```bash
uv run bash scripts/smoke_test.sh
```

Starts the `dummy` launcher, checks conformance, runs a small capped suite.

### Conformance check against any running launcher

```bash
uv run bash scripts/check_launcher.sh http://127.0.0.1:<port>
```

---

## Launcher details

### `dummy` — deterministic reference

Pure CPU. Primary iteration-1 validation backend.

```bash
cd examples/servers/dummy
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate
./serve.sh
```

---

### `naturelm-v1.0` — NatureLM-audio v1.0

Stub mode (default, CPU-only):

```bash
cd examples/servers/naturelm-v1.0
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate
NATURELM_V1_0_STUB=1 ./serve.sh
```

Real inference (GPU, HuggingFace weights):

```bash
HF_TOKEN=hf_... ./serve.sh
```

---

### `naturelm-v1.1` — NatureLM-audio v1.1

Stub mode:

```bash
cd examples/servers/naturelm-v1.1
NATURELM_STUB_MODE=1 ./serve.sh
```

Real inference (gated weights, requires `HF_TOKEN`):

```bash
HF_TOKEN=hf_... ./serve.sh
```

Check gated access:

```bash
HF_TOKEN=hf_... ./serve.sh --check-access
```

---

### `openai_compatible_proxy` — GPT-4o-audio, Gemini, any OpenAI-compat API

Proxies each `predictions_v1` request to an upstream `/v1/chat/completions` endpoint.

Recommended key storage (protected file):

- Put your API key in `~/.config/openai/cfg` as a bare token or `OPENAI_API_KEY=...`
  (`chmod 600`).
- The launcher auto-loads `OPENAI_API_KEY` from that file if the env var is not set.
- For Gemini, put your key in `~/.config/gemini/cfg` as a bare token,
  `GEMINI_API_KEY=...`, or `GOOGLE_API_KEY=...`; the launcher maps it to
  `OPENAI_API_KEY` for Gemini upstreams.

**GPT-4o-audio-preview:**

```bash
cd examples/servers/openai_compatible_proxy
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate

OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://api.openai.com \
  OPENAI_MODEL=gpt-4o-audio-preview \
  PORT=8000 ./serve.sh
```

**Gemini (Google AI Studio key):**

```bash
OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
  OPENAI_MODEL=gemini-2.5-flash \
  PORT=8000 ./serve.sh
```

Stub mode (no API key needed):

```bash
OPENAI_PROXY_STUB=1 PORT=8000 ./serve.sh
```

Registry presets: `openai_proxy_local_8000`, `gpt4o_audio_openai_api`, `gemini_openai_api`.

---

### `vllm` — Qwen3-Omni, Qwen2-Audio, any vLLM model

Adapter sidecar: translates `predictions_v1` ↔ OpenAI-compatible Chat Completions.

**Qwen3-Omni:**

```bash
# Terminal 1 — vLLM model server (GPU required):
vllm serve Qwen/Qwen3-Omni-7B --host 127.0.0.1 --port 8001

# Terminal 2 — BEANS-Next adapter:
cd examples/servers/vllm
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate

VLLM_ADAPTER_STUB=0 \
  VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B \
  PORT=8000 ./serve.sh
```

Stub mode:

```bash
VLLM_ADAPTER_STUB=1 PORT=8000 ./serve.sh
```

Registry preset: `qwen3_omni_vllm_local_8000`, `vllm_local_8000`.

---

### `af3` — Audio Flamingo Next

`nvidia/audio-flamingo-next-hf` (8B, BF16). NVIDIA OneWay Noncommercial License.

Stub mode:

```bash
cd examples/servers/af3
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate
AF3_STUB=1 PORT=19084 ./serve.sh
```

Real inference (GPU, ~20 GB VRAM):

```bash
PORT=19084 ./serve.sh
```

Registry preset: `af_next_local_8000`.

---

### `hf_transformers` — generic HF Transformers starter

Tier-2 template for any HF Transformers model not covered by the above launchers.
Real inference is not implemented — extend `serve.py` for a specific model.

```bash
cd examples/servers/hf_transformers
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate
HF_TRANSFORMERS_STUB=1 PORT=19083 ./serve.sh
```

---

## Running BEANS-Next against a launcher

With a launcher running at `http://127.0.0.1:8000`:

```bash
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --limit 10
```

Or with a registry preset and YAML config:

```bash
uv run beans-next run --config my_run.yaml
```

See the root `README.md` and `beans-next run --help` for full CLI options.
