## BEANS-Next launcher: `naturelm-v1.0`

This is a Tier-1 BEANS-Next serving-kit launcher implementing the **`predictions_v1`** HTTP contract:

- `POST /predict`
- `GET /info`
- `GET /health`

### Status (iteration 1)

This launcher supports both a **real mode** (attempts real NatureLM-audio inference) and a **stub mode** (contract-only, no weights loaded).

- **Stub mode**: `NATURELM_V1_0_STUB=1` (recommended for conformance on CPU-only hosts)
- **Real inference**: enabled when `NATURELM_V1_0_STUB` is **unset** or falsy. This requires extra (launcher-local) dependencies and may be infeasible on a CPU-only machine.

This is intentional: iteration 1’s goal is proving end-to-end benchmark plumbing and launcher conformance without requiring GPU access.

### Real mode prerequisites (non-stub)

Real mode is **not** expected to work in the repo-default CPU environment without additional installs. The minimal blocking failure when attempting `NATURELM_V1_0_STUB=0` in a fresh repo venv is:

- `ModuleNotFoundError: No module named 'torch'`

#### Tokens / credentials you will likely need

NatureLM-audio inference typically requires access to **gated** HuggingFace model weights (the Llama base). In addition, installing NatureLM-audio from GitHub can fail on clusters due to auth / rate limits.

- **HuggingFace token (required for real inference)**:
  - Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) to a token that has access to the gated base model used by NatureLM-audio (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`).
  - On Slurm, `examples/slurm/serve_naturelm_v1_0.sh` will try to populate `HF_TOKEN` automatically from `HUGGINGFACE_HUB_TOKEN` or `~/.config/huggingface/hf_token`, but you can also export it explicitly.

- **GitHub token (sometimes required for install reliability)**:
  - The launcher installs NatureLM-audio via `git+https://...` in real mode. If your environment hits GitHub rate limits or needs auth, export a classic token as `GITHUB_TOKEN` (or set up a GitHub credential helper).
  - One robust pattern is to embed the token in the git URL at install time (be careful not to log it):
    - `https://$GITHUB_TOKEN@github.com/earthspecies/NatureLM-audio.git`
  - If you already have network + anonymous access working, you do **not** need a GitHub token.

In addition, the launcher is intentionally **fail-fast** in real mode: if required dependencies are missing or the model fails to load, the process will exit during startup rather than serving `/health` as HTTP 503 indefinitely. This prevents readiness polling from hanging on repeated 503s.

To attempt real-mode bring-up you must install launcher-local deps (heavy) into the same environment that runs uvicorn, including at minimum:

- `torch` (CPU or CUDA build)
- `numpy`
- `soundfile`
- `huggingface_hub`
- NatureLM-audio (preferred): `git+https://github.com/earthspeciesproject/NatureLM-audio.git`

Example install (choose an appropriate torch build for your platform/GPU):

```bash
# From repo root (installs into this repo's `.venv/`)
uv sync
uv pip install torch numpy soundfile huggingface_hub

# Recommended: install NatureLM-audio from GitHub (may require GITHUB_TOKEN on some clusters)
uv pip install git+https://github.com/earthspecies/NatureLM-audio.git
```

Then start the server with HuggingFace auth available:

```bash
export HF_TOKEN="hf_..."
# or: export HUGGINGFACE_HUB_TOKEN="hf_..."
NATURELM_V1_0_STUB=0 uv run python -m uvicorn serve:app --app-dir examples/servers/naturelm-v1.0 --host 127.0.0.1 --port 19081
```

#### Non-GitHub fallback (best-effort): Transformers `trust_remote_code`

On hosts where GitHub access/auth is unavailable, the launcher can **try** a best-effort fallback path that loads the model directly from HuggingFace using `transformers` with `trust_remote_code=True`.

Important:

- This is **not guaranteed** to work: it depends on the model repo exposing a compatible Transformers interface for audio-conditioned text generation.
- If it fails at runtime, the launcher will raise a clear error telling you to install the official NatureLM-audio package instead.

Install (launcher-local):

```bash
uv sync
uv pip install torch numpy soundfile huggingface_hub transformers
```

Then run real mode as usual (no `NATURELM_V1_0_STUB` flag, or set it to `0`), optionally choosing a model id:

```bash
NATURELM_V1_0_STUB=0 NATURELM_V1_0_MODEL=EarthSpeciesProject/NatureLM-audio \
  uv run python -m uvicorn serve:app --app-dir examples/servers/naturelm-v1.0 --host 127.0.0.1 --port 19081
```

CPU note: `serve.py` will fall back to `device="cpu"` when CUDA is unavailable, but inference is expected to be extremely slow.

### Quickstart

From repo root:

```bash
uv sync

# IMPORTANT: this launcher directory name contains a '-' and '.', so it cannot be imported via a
# dotted module path like `examples.servers.naturelm-v1.0.serve:app`.
# Use `--app-dir` and refer to `serve:app` instead.
#
# Execution-policy-safe launcher lifecycle (recommended): use `scripts/with_uvicorn.py`.
# This owns the server lifecycle, polls `/health` for readiness (no `sleep`), runs a child command,
# then shuts the server down best-effort before exiting.
NATURELM_V1_0_STUB=1 uv run python scripts/with_uvicorn.py \
  --app serve:app \
  --app-dir examples/servers/naturelm-v1.0 \
  --host 127.0.0.1 \
  --port 19081 \
  --ready-url http://127.0.0.1:19081/health \
  -- -- uv run bash scripts/check_launcher.sh http://127.0.0.1:19081
```

If you want to run the server manually (two-terminal workflow), you can also start uvicorn directly:

```bash
NATURELM_V1_0_STUB=1 uv run python -m uvicorn serve:app \
  --app-dir examples/servers/naturelm-v1.0 \
  --host 127.0.0.1 \
  --port 19081
```

Then (in another terminal) you can hit:

- `GET http://127.0.0.1:19081/health`
- `GET http://127.0.0.1:19081/info`
- `POST http://127.0.0.1:19081/predict`

### Environment variables

- **`PORT`**: Port to bind (default `19081`)
- **`NATURELM_V1_0_BIND_HOST`**: Bind host (default `127.0.0.1`)
- **`NATURELM_V1_0_STUB`**: When truthy, enables stub mode (default: unset → real mode)
- **`NATURELM_V1_0_MAX_BATCH_SIZE`**: Maximum batch size before returning HTTP 413 (default `8`)
- **`NATURELM_V1_0_MODEL`**: Model identifier reported in `/info` (default `EarthSpeciesProject/NatureLM-audio`)
- **`NATURELM_V1_0_MODEL_REVISION`**: Revision string reported in `/info` (default `stub` in stub mode; otherwise resolved from HF if reachable, else `unknown`)

### Notes on audio

The BEANS-Next prompt convention may include `<Audio><AudioHere></Audio>` placeholders in message
content. The launcher only enforces placeholder ↔ `audio_inputs` matching when placeholders are
present; if there are zero placeholders, requests may still include `audio_inputs` out-of-band.

In real mode (`NATURELM_V1_0_STUB=0`), `payload_type="file_url"` is supported by downloading the
URL and decoding it as WAV bytes (same as `base64_wav`), so it may require outbound network access.

### Known limitation: `scripts/with_uvicorn.py --app ...` import path

Some repo automation starts launchers via a dotted import path (e.g.
`--app examples.servers.dummy.serve:app`). That pattern does **not** work for `naturelm-v1.0`
because the directory name is not a valid Python module segment.

This launcher is still compatible with `scripts/with_uvicorn.py` by using `--app-dir`:

```bash
NATURELM_V1_0_STUB=1 uv run python scripts/with_uvicorn.py \
  --app serve:app \
  --app-dir examples/servers/naturelm-v1.0 \
  --host 127.0.0.1 \
  --port 19081 \
  --ready-url http://127.0.0.1:19081/health \
  -- \
  uv run bash scripts/check_launcher.sh http://127.0.0.1:19081
```

