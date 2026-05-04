# BEANS-Next `naturelm-v1.1` launcher

Tier-1 FastAPI launcher implementing the BEANS-Next HTTP contract (`predictions_v1`) for **NatureLM-audio 1.1**.

Endpoints:

- `GET /health`
- `GET /info`
- `POST /predict`

## HuggingFace gated access (`HF_TOKEN`) — required

NatureLM-audio 1.1 weights are gated on HuggingFace at:

- `EarthSpeciesProject/naturelm-audio-1.1.00-private`

You must provide a HuggingFace token with access to this repo.

Accepted env vars (launcher + Slurm):

- `HF_TOKEN` (preferred)
- `HUGGINGFACE_HUB_TOKEN` (fallback; `serve_naturelm_v1_1.sh` will copy it into `HF_TOKEN`)

On Slurm, `examples/slurm/serve_naturelm_v1_1.sh` also falls back to reading:

- `~/.config/huggingface/hf_token`

### Expected failure modes (from `GET /health`)

- **Missing token** (`HF_TOKEN` unset/empty) → HTTP 503 with:

  - `HF_TOKEN is not set. NatureLM-audio 1.1 weights are gated on HuggingFace. Export HF_TOKEN with access to 'EarthSpeciesProject/naturelm-audio-1.1.00-private'.`

- **Token present but no gated access** → HTTP 503 with:

  - `HF_TOKEN lacks access to gated repo 'EarthSpeciesProject/naturelm-audio-1.1.00-private'. Request access on the model page (HuggingFace) and wait for approval.`

- **Token invalid / unauthorized** → HTTP 503 with:

  - `HF_TOKEN is invalid or unauthorized for 'EarthSpeciesProject/naturelm-audio-1.1.00-private' (HTTP 401).`
  - or `... (HTTP 403).`

- **Weights cannot be downloaded** → HTTP 503 with:

  - `failed to download weights for 'EarthSpeciesProject/naturelm-audio-1.1.00-private': ...`

## How to request access

1. Sign into HuggingFace in your browser.
2. Navigate to the model page for `EarthSpeciesProject/naturelm-audio-1.1.00-private`.
3. Click **Request access** and wait for approval.
4. Create a token and export it as `HF_TOKEN`.

## Setup

```bash
cd examples/servers/naturelm-v1.1
uv venv
uv pip install -r requirements.txt
. .venv/bin/activate
```

## Real inference dependencies (non-stub) — REQUIRED

The `naturelm-v1.1` launcher can run in **stub mode** for contract conformance without any
model-side code installed.

To serve **real** NatureLM-audio 1.1 inference, you must additionally install:

- **`esp-research`** (Earth Species Project internal research code; launcher-only dependency)
- **NatureLM-audio (v1.1) Python package / code** (launcher-only dependency)

These must **NOT** be added to BEANS-Next core dependencies. They are required only inside the
launcher environment (e.g., a Slurm job venv).

### Python version requirement

The authoritative `esp-research` NatureLM-audio inference environment (`NatureLM-audio-v1.5`) targets
Python **3.12**. When serving v1.1 on Slurm, prefer Python 3.12 (the Slurm serve script defaults to
`BEANS_PRO_UV_PYTHON=3.12`).

### Recommended: local `esp-research` checkout (authoritative inference instructions)

The most complete, actively maintained instructions for NatureLM-audio inference live in
`esp-research` on the `david-multi-audio` branch:

- Project docs: `projects/NatureLM-audio-v1.5/`
- Upstream reference: `https://github.com/earthspecies/esp-research/blob/david-multi-audio/projects/NatureLM-audio-v1.5/`

Recommended one-time clone (standalone folder):

```bash
git clone --single-branch --branch david-multi-audio \
  git@github.com:earthspecies/esp-research.git \
  "$HOME/code/esp-research__david-multi-audio"
```

Then, inside the launcher venv, install `esp-research` from that local checkout:

```bash
# In examples/servers/naturelm-v1.1/.venv
uv pip install -e "$HOME/code/esp-research__david-multi-audio"
```

### Alternative: install `esp-research` directly from Git

If you cannot use a local checkout, install from Git (launcher venv):

```bash
# In examples/servers/naturelm-v1.1/.venv
uv pip install "git+ssh://git@github.com/earthspecies/esp-research.git@david-multi-audio"
# Also install NatureLM-audio v1.1 code as required by your serving setup.
```

Then run with real inference enabled:

```bash
export HF_TOKEN=hf_...
export NATURELM_ENABLE_INFERENCE=1
./serve.sh
```

## Run

Default port is **8001**:

```bash
export HF_TOKEN=hf_...
./serve.sh
```

Custom port:

```bash
export HF_TOKEN=hf_...
PORT=8001 ./serve.sh
```

Check gated access without starting a server:

```bash
export HF_TOKEN=hf_...
./serve.sh --check-access
```

## Conformance-only “stub mode”

If you only need to validate HTTP contract conformance (schema/batching) without
access to gated weights, you can run in stub mode:

```bash
export NATURELM_STUB_MODE=1
./serve.sh
```

In stub mode:

- `/health` returns 200 with `{ "mode": "stub" }`
- `/predict` returns deterministic placeholder predictions
- no HuggingFace calls are made

## Launcher check (best-effort)

From the repo root:

```bash
cd examples/servers/naturelm-v1.1
export HF_TOKEN=hf_...
./serve.sh
```

Then in a second terminal:

```bash
uv run bash scripts/check_launcher.sh http://localhost:8001
```

