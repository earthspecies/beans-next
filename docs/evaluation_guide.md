# BEANS-Next Evaluation Guide

Step-by-step instructions for evaluating each supported model on BEANS-Zero: hardware requirements, serving setup, running inference, and interpreting metrics.

---

## Architecture overview

```
BEANS-Zero dataset (HuggingFace or esp_data)
        │
        ▼
beans-next run  ──HTTP──►  launcher (model server)
        │                       │
        │ ◄─── predictions ──────┘
        │
        ▼
post-processing → metrics → run_summary.json
```

The core library never loads model weights. Every model runs as a separate HTTP server (the **launcher**). `beans-next run` sends audio + prompt over HTTP, receives text predictions, and computes metrics locally on CPU.

This means:
- **Serving** (GPU node / API call) and **inference orchestration** (CPU) are decoupled.
- You can re-score any completed run without re-running the model: `beans-next score-from-file`.
- Cloud models (GPT-4o-audio, Gemini) need no GPU at all.

### Dataset backend selection

BEANS-Next supports two dataset backends. The backend controls how audio and metadata are loaded; task definitions and metrics are identical.

| Backend | Flag | When to use |
|---|---|---|
| `esp_data` | `--backend esp_data` | ESP infrastructure with GCS access and the `esp` dependency group installed |
| `huggingface` | `--backend huggingface` | Public access — loads from the two-table Parquet bundle on HuggingFace Hub; no private credentials needed |

Switch via CLI flag, run-config YAML, or env var (env var is the lowest-priority fallback):

```bash
# CLI flag (takes priority over everything)
beans-next run --backend huggingface ...

# Run-config YAML field
data_source: huggingface

# Env var fallback (used when flag and YAML field are both absent)
BEANS_PRO_DATA_SOURCE=huggingface beans-next run ...
```

The `huggingface` backend requires `HF_TOKEN` if the dataset is private:

```bash
export HF_TOKEN=hf_...
```

The `esp_data` backend requires the `esp` dependency group and GCS credentials:

```bash
uv sync --group esp
gcloud auth application-default login
```

---

## BEANS-Zero tasks and metrics

### Suites

| Suite | Tasks | Purpose |
|---|---|---|
| `beans_zero_smoke` | 3 tasks (esc50, enabirds, captioning) | Quick sanity check; use with `--limit 10` |
| `beans_zero_core` | 22 tasks (all below) | Full evaluation |

### Task types and metrics

**Classification** — 16 tasks

| Task | Dataset | Labels |
|---|---|---|
| `beans_zero_esc50` | ESC-50 | Environmental sound classes |
| `beans_zero_cbi` | CBI | Bird species |
| `beans_zero_watkins` | Watkins | Marine mammal species |
| `beans_zero_humbugdb` | HumbugDB | Mosquito species |
| `beans_zero_lifestage` | — | Life stage |
| `beans_zero_call_type` | — | Call type |
| `beans_zero_zf_indiv` | Zebra finch | Individual identity |
| `beans_zero_unseen_{species,genus,family}_{cmn,sci,tax}` | — | Zero-shot taxonomic generalization |

Metrics: **accuracy**, **macro F1**, **macro precision**, **macro recall**

**Detection** — 5 tasks

| Task | Dataset | Notes |
|---|---|---|
| `beans_zero_enabirds` | ENABirds | Multi-label presence detection |
| `beans_zero_hiceas` | HICEAS | Cetacean detection |
| `beans_zero_rfcx` | RFCX | Rainforest species |
| `beans_zero_gibbons` | Gibbons | Gibbon call detection |
| `beans_zero_dcase` | DCASE | Acoustic event detection |

Metrics: **macro average precision**, **macro F1**, **macro precision**, **macro recall**

**Captioning** — 1 task

| Task | Metrics |
|---|---|
| `beans_zero_captioning` | **CIDEr** (corpus mean, `summary.metrics.mean.cider`) |

**Open-ended / counting** — tasks added as BEANS-Zero grows

These tasks produce free-form text output that cannot be reliably parsed into a fixed label set. Examples: "Count the vocalizations per species in this recording. Use scientific names." Metrics are computed by an **LLM judge** rather than deterministic scorers. See [LLM-as-judge guide](llm_judge.md).

| `task_type` value | Post-processing | Scoring |
|---|---|---|
| `classification` | comma-split + fuzzy label match | accuracy, F1, precision, recall |
| `detection` | comma-split + fuzzy label match | macro AP, F1, precision, recall |
| `captioning` | whitespace normalise only | CIDEr (corpus); optional SPIDEr via `spider` + Java |
| `open_ended` | whitespace normalise only | LLM judge |
| `counting` | whitespace normalise only | LLM judge |
| `qa` | whitespace normalise only | LLM judge |

---

### Output artifacts

Every `beans-next run` writes to `--output-dir` (default `results/<run_id>/`):

| File | Content |
|---|---|
| `predictions.jsonl` | Raw launcher responses (sample_id, predictions, latency, usage) |
| `processed_predictions.jsonl` | Post-processed predictions with ground truth labels |
| `scored_predictions.jsonl` | Post-processed predictions with computed scores |
| `summary.json` | Aggregate summary (model info, per-task metrics, timing) |
| `judge_outputs.jsonl` | LLM judge outputs (only if `--judge-url` used) |

---

## Metrics reference

### Classification metrics

`accuracy`, `precision`, `recall`, `f1` are all in `beans_next.metrics.classification`.

**`accuracy`** — subset accuracy: fraction of samples where prediction exactly matches target.
- Accepts flat integer class labels or binary indicator matrices.
- For multi-reference targets (comma-separated string like `"cat, feline"`), use `top1_accuracy` instead.

**`top1_accuracy`** — correct if the prediction matches any option in a comma-separated target string.

```python
from beans_next.metrics.classification import top1_accuracy
top1_accuracy(["cat"], ["cat, feline"])   # → 1.0
top1_accuracy(["feline"], ["cat, feline"]) # → 1.0
top1_accuracy(["dog"], ["cat, feline"])   # → 0.0
```

**`precision`, `recall`, `f1`** — all support `average=` parameter:

| `average` | Meaning |
|---|---|
| `"macro"` | Unweighted mean across classes (default) |
| `"micro"` | Pool all TP/FP/FN, then compute |
| `"weighted"` | Macro weighted by per-class support |
| `"binary"` | Binary 0/1 inputs, positive label = 1 |

When precision or recall is undefined (no positive predictions / no positive targets), `zero_division=` controls the substituted value (default `0`).

### Detection metrics

**`average_precision`** in `beans_next.metrics.detection` — multi-label AP (PR-curve integral).

- `predictions`: nested `(n_samples, n_labels)` float score matrix
- `targets`: same-shaped binary indicator matrix, **or** ragged integer label-index rows
- `average="macro"` (default): mean AP across labels; `average="micro"`: pooled across labels × samples

Columns with no positive targets contribute 0.0 to the macro average (standard sklearn behaviour).

```python
from beans_next.metrics.detection import average_precision
y_score = [[0.9, 0.1], [0.1, 0.8]]
y_true  = [[1, 0], [0, 1]]
average_precision(y_score, y_true)             # → 1.0  (macro)
average_precision(y_score, y_true, average="micro")  # → 1.0
```

### Captioning metrics

**`cider`** / **`cider_corpus_mean_normalized`** in `beans_next.metrics.captioning` — corpus-level mean CIDEr in `[0, 1]` (no Java). The benchmark runner merges this into `summary.metrics.mean.cider`. Per-sample `score_sample` rows are empty for captioning because single-pair CIDEr is degenerate.

**`spider`** — SPIDEr = `(CIDEr/10 + SPICE) / 2`. **SPICE** requires Java ≥ 8 and Stanford CoreNLP JARs; if SPICE is missing, the implementation logs a warning and uses `0.0` for SPICE.

SPICE setup (one-time, CPU node), only if you use SPIDEr:

```bash
uv run beans-next setup-spice
```

Check availability:

```python
from beans_next.metrics._spice import check_spice_available, SpiceUnavailableError
try:
    check_spice_available()
    print("SPICE ready")
except SpiceUnavailableError as e:
    print(f"SPICE unavailable: {e}")
```

### Score routing

`beans_next.metrics.score_sample` is the single entry point used by the runner. It inspects `example.metadata["task"]` and `example.labels` to pick the right metric family automatically:

| `labels` type | `metadata["task"]` | Metrics returned |
|---|---|---|
| `str` | contains `"caption"` | (none per sample; use `mean.cider`) |
| `str` | other | `accuracy`, `precision`, `recall`, `f1` (exact match) |
| `list` | contains `"classification"` (not `"detection"`) | `top1_accuracy`, `accuracy` |
| `list` | `"detection"` or unspecified | `average_precision`, `precision`, `recall`, `f1` |
| anything else | — | `{}` — no score |

For `open_ended`, `counting`, and `qa` tasks the runner does not call `score_sample`; scores come from the LLM judge instead.

### Fuzzy label matching

Before scoring, classification and detection predictions go through the post-processing pipeline:

1. **`parse_labels_comma`** — splits `"cat, dog"` into `["cat", "dog"]`
2. **`normalize_whitespace`** — strips extra spaces
3. **`fuzzy_match_to_labels`** — maps each token to the nearest label in the task's vocabulary using Levenshtein distance

Fuzzy matching is **skipped** for captioning, open-ended, counting, and qa tasks so that free-form text is not corrupted.

The vocabulary is loaded from `beans_next/registry/beans_zero_labels.json` for BEANS-Zero tasks (one entry per dataset subset) and falls back to the labels seen in the current batch when no registry entry exists.

---

## LLM-as-judge

For open-ended tasks where outputs cannot be reliably parsed into fixed labels, beans-next supports an HTTP judge service. See the [LLM-as-judge guide](llm_judge.md) for full setup and configuration.

**Quick usage:**

```bash
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --judge-url   http://127.0.0.1:8010/judge \
  --suite beans_zero_core \
  --output-dir results/with_judge
```

Results are written to `judge_outputs.jsonl` alongside the normal prediction artifacts.

---

## Model evaluation instructions

---

### NatureLM-audio v1.0

**Hardware**: 1× GPU, ≥ 24 GB VRAM (A100-40 or better recommended)
**CPU (inference job)**: 4 cores, no GPU

#### One-time setup

```bash
# Download weights to shared cache (login node, needs internet)
HF_HOME=/scratch/shared/hf_cache uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('EarthSpeciesProject/naturelm-audio-1.0')
"

# Install launcher deps
cd examples/servers/naturelm-v1.0
uv venv && uv pip install -r requirements.txt
```

#### Local (single machine)

```bash
# Terminal 1 — start server
cd examples/servers/naturelm-v1.0
HF_HOME=/scratch/shared/hf_cache PORT=8000 ./serve.sh

# Wait for "Launcher ready" message, then terminal 2:
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --output-dir results/naturelm_v1_0
```

#### SLURM (two jobs)

```bash
# 1. Submit serving job
SERVE_JOB=$(sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh)
echo "Serving job: $SERVE_JOB"

# 2. Submit inference job (starts when serving job is running)
BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_OUT_DIR=$SCRATCH/results/naturelm_v1_0_$(date +%Y%m%d) \
sbatch --dependency=after:$SERVE_JOB examples/slurm/run_inference.sh
```

**Typical timing**: model loads in ~3 min; full suite ~2–4 h depending on dataset size.

---

### NatureLM-audio v1.1

**Hardware**: 1× GPU, ≥ 24 GB VRAM
**CPU (inference job)**: 4 cores, no GPU
**Requires**: gated HuggingFace access (`HF_TOKEN`)

#### One-time setup

```bash
# Check gated access before submitting a job
cd examples/servers/naturelm-v1.1
HF_TOKEN=hf_... ./serve.sh --check-access

# Download weights
HF_HOME=/scratch/shared/hf_cache HF_TOKEN=hf_... uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('EarthSpeciesProject/naturelm-audio-1.1.00-private', token='hf_...')
"

cd examples/servers/naturelm-v1.1
uv venv && uv pip install -r requirements.txt
```

#### Local

```bash
cd examples/servers/naturelm-v1.1
HF_TOKEN=hf_... HF_HOME=/scratch/shared/hf_cache PORT=8001 ./serve.sh

uv run beans-next run \
  --predict-url http://127.0.0.1:8001/predict \
  --suite beans_zero_core \
  --output-dir results/naturelm_v1_1
```

#### SLURM

```bash
SERVE_JOB=$(HF_TOKEN=hf_... sbatch --parsable examples/slurm/serve_naturelm_v1_1.sh)

BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_OUT_DIR=$SCRATCH/results/naturelm_v1_1_$(date +%Y%m%d) \
sbatch --dependency=after:$SERVE_JOB examples/slurm/run_inference.sh
```

---

### Audio Flamingo Next (`nvidia/audio-flamingo-next-hf`)

**Hardware**: 1× GPU with ≥ 20 GB VRAM (BF16, 8B params)
**CPU (inference job)**: 4 cores, no GPU
**License**: NVIDIA OneWay Noncommercial License — **non-commercial research use only**

#### One-time setup

```bash
HF_HOME=/scratch/shared/hf_cache uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/audio-flamingo-next-hf')
"

cd examples/servers/af3
uv venv && uv pip install -r requirements.txt
```

#### Local

```bash
cd examples/servers/af3
HF_HOME=/scratch/shared/hf_cache PORT=8000 ./serve.sh

uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --output-dir results/af3
```

#### SLURM

```bash
SERVE_JOB=$(sbatch --parsable examples/slurm/serve_af3.sh)

BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_OUT_DIR=$SCRATCH/results/af3_$(date +%Y%m%d) \
sbatch --dependency=after:$SERVE_JOB examples/slurm/run_inference.sh
```

**Note**: AF3 takes up to 15 min to load weights. The serve script waits up to 15 min for `/health`; the inference script polls for the URL file with a 30-min timeout.

---

### Qwen3-Omni (via vLLM)

**Hardware**: 1× GPU ≥ 24 GB VRAM for 7B; multi-GPU via tensor parallelism for larger variants
**CPU (inference job)**: 4 cores, no GPU
**Requires**: `vllm` installed on the GPU node (`pip install vllm`)

#### One-time setup

```bash
# On GPU node or login node with internet access:
HF_HOME=/scratch/shared/hf_cache uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-7B')
"

cd examples/servers/vllm
uv venv && uv pip install -r requirements.txt
# vLLM must be installed separately (GPU-node specific build):
# pip install vllm
```

#### Local

```bash
# Terminal 1 — vLLM model server
HF_HOME=/scratch/shared/hf_cache \
  vllm serve Qwen/Qwen3-Omni-7B --host 127.0.0.1 --port 8001

# Terminal 2 — BEANS-Next adapter (once vLLM is healthy)
cd examples/servers/vllm
VLLM_ADAPTER_STUB=0 \
  VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B \
  PORT=8000 ./serve.sh

# Terminal 3 — inference
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --output-dir results/qwen3_omni
```

#### SLURM

```bash
SERVE_JOB=$(VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B sbatch --parsable examples/slurm/serve_vllm.sh)

BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_OUT_DIR=$SCRATCH/results/qwen3_omni_$(date +%Y%m%d) \
sbatch --dependency=after:$SERVE_JOB examples/slurm/run_inference.sh
```

**Multi-GPU** (e.g. 2 GPUs for a larger model):
```bash
# Edit serve_vllm.sh header: #SBATCH --gpus=2
VLLM_MODEL_ID=Qwen/Qwen3-Omni-72B \
VLLM_TENSOR_PARALLEL_SIZE=2 \
sbatch --parsable examples/slurm/serve_vllm.sh
```

**Typical timing**: vLLM loads 7B in ~5 min; full suite ~3–6 h.

---

### ESC-50 official via OpenAI proxy (stub-first, CPU-only)

This is a **minimal wiring check** for the official ESC-50 evaluation protocol:
`beans_zero_esc50_official`. It runs the `openai_compatible_proxy` launcher in
**stub mode** (no API keys) and executes **`--limit 1`**.

From the repo root:

```bash
uv sync

uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/openai_compatible_proxy \
  --cmd-cwd . \
  --app serve:app \
  --host 127.0.0.1 \
  --port 19085 \
  --ready-url http://127.0.0.1:19085/health \
  --env OPENAI_PROXY_STUB=1 \
  -- uv run bash scripts/check_launcher.sh http://127.0.0.1:19085

uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/openai_compatible_proxy \
  --cmd-cwd . \
  --app serve:app \
  --host 127.0.0.1 \
  --port 19085 \
  --ready-url http://127.0.0.1:19085/health \
  --env OPENAI_PROXY_STUB=1 \
  -- uv run beans-next run \
    --config configs/benchmarks/esc50_official_openai_proxy_stub.yaml
```

### ESC-50 official via OpenAI proxy (real OpenAI, minimal reproducer)

This runs the **same official ESC-50 eval task** (`beans_zero_esc50_official`) but points the
`openai_compatible_proxy` launcher at the real OpenAI API. This **incurs cost**; start with
`limit: 1` (the default in the config).

#### Prereqs

- Recommended: store your OpenAI API key in a protected file:

```bash
mkdir -p "$HOME/.config/openai"
chmod 700 "$HOME/.config/openai"
printf "OPENAI_API_KEY=sk-...\n" >"$HOME/.config/openai/cfg"
chmod 600 "$HOME/.config/openai/cfg"
```

The `openai_compatible_proxy` launcher auto-loads `OPENAI_API_KEY` from
`~/.config/openai/cfg` if the env var is not set.

#### Run (safe server lifecycle; health-polled; no backgrounding)

From the repo root:

```bash
uv sync

uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/openai_compatible_proxy \
  --cmd-cwd . \
  --app serve:app \
  --host 127.0.0.1 \
  --port 19085 \
  --ready-url http://127.0.0.1:19085/health \
  --env OPENAI_PROXY_STUB=0 \
  --env OPENAI_BASE_URL=https://api.openai.com \
  --env OPENAI_MODEL=gpt-4o-audio-preview \
  -- uv run bash scripts/check_launcher.sh http://127.0.0.1:19085

uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/openai_compatible_proxy \
  --cmd-cwd . \
  --app serve:app \
  --host 127.0.0.1 \
  --port 19085 \
  --ready-url http://127.0.0.1:19085/health \
  --env OPENAI_PROXY_STUB=0 \
  --env OPENAI_BASE_URL=https://api.openai.com \
  --env OPENAI_MODEL=gpt-4o-audio-preview \
  -- uv run beans-next run \
    --config configs/benchmarks/esc50_official_openai_proxy_openai_real.yaml
```

To expand beyond 1 sample, edit `configs/benchmarks/esc50_official_openai_proxy_openai_real.yaml`
and set `limit: 5` (recommended for smoke tests).

### GPT-4o-audio-preview (OpenAI API)

**Hardware**: No GPU. Runs entirely on a CPU node or login node.
**Requires**: OpenAI API key with access to `gpt-4o-audio-preview`
**Cost**: billed per token by OpenAI. Audio tokens are expensive — use `--limit` for initial tests.

#### Setup

```bash
cd examples/servers/openai_compatible_proxy
uv venv && uv pip install -r requirements.txt
```

#### Local

```bash
OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://api.openai.com \
  OPENAI_MODEL=gpt-4o-audio-preview \
  PORT=8000 ./serve.sh

uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_smoke \
  --limit 20 \
  --output-dir results/gpt4o_audio_test
```

#### SLURM (CPU partition)

No serving job needed — the proxy is CPU-only and can run inside the inference job directly. For
server lifecycle discipline, use `scripts/with_uvicorn.py` in your Slurm script so the proxy is
health-polled (`/health`) and always cleaned up without backgrounding tricks.

**Note**: Check whether your cluster's GPU/CPU nodes have outbound HTTPS. Some HPC clusters block internet from compute nodes — you may need `--export=https_proxy=...` or to use a login node.

**Recommended first run**: `--suite beans_zero_smoke --limit 5` to verify auth and check cost before a full run.

---

### Gemini (Google AI Studio)

**Hardware**: No GPU. CPU-only.
**Requires**: [Google AI Studio](https://aistudio.google.com) API key
**Models**: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.5-flash-lite`, etc.

#### Gemini OpenAI-compatible endpoint notes (2026)

BEANS-Next talks to Gemini through the `openai_compatible_proxy` launcher, which calls an
OpenAI-compatible Chat Completions endpoint (`POST /v1/chat/completions`).

- **Auth**: use `Authorization: Bearer <API_KEY>` (the launcher handles `Bearer` prefix for the
  default `OPENAI_AUTH_HEADER=Authorization`).
- **Base URL (AI Studio)**: `https://generativelanguage.googleapis.com/v1beta/openai/`
  (a trailing slash is recommended by common client examples).
- **Alternative (Vertex AI)**: Vertex AI also exposes an OpenAI-compatible surface, but uses Google
  Cloud auth tokens and a different base URL. If you use Vertex, keep the same launcher and set
  `OPENAI_BASE_URL` accordingly to the Vertex docs for your project/location.
- **Conservative constraints**:
  - Expect `429` rate limits; start with `limit: 1` or `limit: 5` and increase slowly.
  - The proxy forwards audio only as `base64_wav` (avoid `file_path` / `file_url` here).

#### Setup

Same launcher as GPT-4o-audio — `openai_compatible_proxy`:

```bash
cd examples/servers/openai_compatible_proxy
uv venv && uv pip install -r requirements.txt
```

#### Local

```bash
mkdir -p "$HOME/.config/gemini"
chmod 700 "$HOME/.config/gemini"
printf "AIza...\n" >"$HOME/.config/gemini/cfg"
chmod 600 "$HOME/.config/gemini/cfg"

OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
  OPENAI_MODEL=gemini-2.5-flash \
  PORT=8000 ./serve.sh

uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --output-dir results/gemini_2_0_flash
```

#### Minimal ESC-50 official run config (fixed port 19085)

This follows the same `scripts/with_uvicorn.py` lifecycle pattern as the stub run, but uses real
Gemini credentials from `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or a
bare token / key-value entry in `~/.config/gemini/cfg`, and runs
`beans_zero_esc50_official` with `limit: 1` by default.

From the repo root:

```bash
uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/openai_compatible_proxy \
  --cmd-cwd . \
  --app serve:app \
  --host 127.0.0.1 \
  --port 19085 \
  --ready-url http://127.0.0.1:19085/health \
  --env OPENAI_PROXY_STUB=0 \
  --env OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
  --env OPENAI_MODEL=gemini-2.5-flash \
  -- uv run beans-next run \
    --config configs/benchmarks/esc50_official_openai_proxy_gemini_real.yaml
```

#### SLURM

Same pattern as GPT-4o-audio above — single CPU job with proxy + inference.

---

## Running a small test before committing to a full suite

Always validate with a smoke run first:

```bash
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_smoke \
  --limit 5 \
  --output-dir results/smoke_test
```

Check `results/smoke_test/metrics.json` — if scores are non-zero and no errors appear in `predictions.jsonl`, the pipeline is working correctly.

---

## Side-by-side comparison (multiple models)

Submit one serving job per model, then one inference job per model. All jobs can run concurrently if you have sufficient GPU quota:

```bash
# Submit all serving jobs
JOB_NLM=$(sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh)
JOB_AF3=$(sbatch --parsable examples/slurm/serve_af3.sh)
JOB_QWN=$(VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B sbatch --parsable examples/slurm/serve_vllm.sh)

# Submit inference jobs for each
for JOB_ID in $JOB_NLM $JOB_AF3 $JOB_QWN; do
  BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$JOB_ID.url \
  BEANS_PRO_SUITE=beans_zero_core \
  BEANS_PRO_OUT_DIR=$SCRATCH/results/run_${JOB_ID} \
  sbatch --dependency=after:$JOB_ID examples/slurm/run_inference.sh
done
```

Each run produces its own `run_summary.json`. Compare with:

```bash
for d in $SCRATCH/results/run_*/; do
  echo "=== $d ==="
  python -c "import json,sys; s=json.load(open('$d/run_summary.json')); print(json.dumps(s.get('metrics',{}), indent=2))"
done
```

---

## Re-scoring without re-running inference

If you want to apply different metrics or a judge scorer to an existing run:

```bash
uv run beans-next score-from-file \
  results/naturelm_v1_0/processed_predictions.jsonl \
  -o results/naturelm_v1_0_rescored
```

---

## Resource summary

| Model | GPU VRAM | GPU count | CPU (inference) | Internet needed |
|---|---|---|---|---|
| NatureLM-audio v1.0 | ≥ 24 GB | 1 | 4 cores | HuggingFace (setup only) |
| NatureLM-audio v1.1 | ≥ 24 GB | 1 | 4 cores | HF + gated token (setup only) |
| Audio Flamingo Next | ≥ 20 GB | 1 | 4 cores | HuggingFace (setup only) |
| Qwen3-Omni-7B | ≥ 24 GB | 1 | 4 cores | HuggingFace (setup only) |
| Qwen3-Omni-72B | ≥ 160 GB | 4–8× | 4 cores | HuggingFace (setup only) |
| GPT-4o-audio-preview | None | 0 | 4 cores | Always (API calls) |
| Gemini | None | 0 | 4 cores | Always (API calls) |

**SLURM partition guidance**:
- Serving jobs: GPU partition (e.g. `a100-40`)
- Inference jobs: CPU partition is sufficient; GPU partition also works if that's all you have

## Troubleshooting

**`launcher did not become healthy`**: model weights are still loading or failed. Check the serving job log (`/home/$USER/logs/<job_id>.log`) for OOM errors or missing weights.

**`--predict-url-file not found`**: the serving job hasn't started or failed immediately. Check the SLURM job state with `squeue -j <job_id>`.

**Empty predictions / all errors in `predictions.jsonl`**: check the `error` field on individual items. Common causes: audio format mismatch, GPU OOM mid-run, API rate limits.

**`No dataset examples were loaded`**: HuggingFace dataset download failed. Set `HF_HOME` to a writable shared path and check network access from the compute node.

**API rate limits (OpenAI/Gemini)**: reduce concurrency or add `OPENAI_PROXY_RETRIES` / `OPENAI_PROXY_TIMEOUT_SEC`. The proxy retries on 429 and 5xx automatically up to `OPENAI_PROXY_RETRIES` times.
