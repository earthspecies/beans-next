# BEANS-Next

Earth Species Project's bioacoustics benchmark library for audio language models.

The core package (`beans_next`) is **dependency-light**: no torch, transformers, or vLLM. Models are always reached over **HTTP** via the `predictions_v1` contract. Heavy inference lives in per-launcher virtual environments under `examples/servers/`.

## Documentation

| Document | What it covers |
|---|---|
| **[Full evaluation pipeline](docs/full_evaluation_pipeline.md)** | Complete end-to-end pipeline: rsync, key setup, weight download, serving, SLURM submit scripts, results retrieval, rescoring |
| **[Evaluation guide](docs/evaluation_guide.md)** | Per-model serving reference — hardware, local and SLURM instructions, metrics |
| **[LLM-as-judge guide](docs/llm_judge.md)** | All three judge modes (rubric, YES/NO, extractor); built-in templates; retroactive vs inline judging |
| **[Gemma 4 judge serving](docs/judge_model_gemma4.md)** | Serving `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` as a `predictions_v1` judge via vLLM |
| **[Launcher serving kit](examples/servers/README.md)** | All launchers, quick-start per model, conformance checks |
| **[SLURM scripts](examples/slurm/README.md)** | Two-job pattern (serving + inference), multi-model side-by-side |
| **[HTTP contract](docs/http_contract.md)** | `predictions_v1` wire schema, batching rules, endpoint spec |
| **[Paper workflow](docs/paper_workflow_iteration1.md)** | NatureLM side-by-side reproduction workflow |

> **Qwen3-Omni (Slurm) runbook**: The most up-to-date operational notes live in
> `examples/servers/af3/README.md` under **“Qwen3-Omni serving notes”** (single-stage vLLM-Omni YAML,
> known-bad nodes, `$USER` path expansion gotcha, `/tmp/voice_samples` workaround, and recommended
> 10-second audio cap).

---

## Installation

**Requirements**: Python ≥ 3.11, [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the library and dev dependencies:

```bash
git clone <repo>
cd beans-next
uv sync
```

This creates `.venv/` and installs `beans_next` plus all dev deps. Use `uv run` to execute anything inside that environment.

To use `esp_data` datasets (internal ESP data access), also install the `esp` group:

```bash
uv sync --group esp
```

> **Launchers** (model servers under `examples/servers/`) have their own isolated venvs and are set up separately — see the [launcher guide](examples/servers/README.md).

---

## Quick start

### 1. Smoke test — no GPU, no API key

Verify the full pipeline works on CPU using the deterministic `dummy` launcher:

```bash
uv run bash scripts/smoke_test.sh
```

This starts the dummy server, checks contract conformance, runs a small capped suite, and writes artifacts under `results/`. Takes ~30 seconds.

### 2. Run against a real model

Pick a model and start its launcher. For full per-model instructions (weights download, SLURM scripts, GPU requirements) see the **[Evaluation guide](docs/evaluation_guide.md)**.

> For **NatureLM-audio v1.1** real inference, see `examples/servers/naturelm-v1.1/README.md`
> for full setup instructions. Weights are gated — `HF_TOKEN` required.

**GPT-4o-audio-preview** (no GPU, OpenAI API key):

```bash
cd examples/servers/openai_compatible_proxy
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate

# Recommended: store your key in a protected file:
#   mkdir -p ~/.config/openai && chmod 700 ~/.config/openai
#   printf "sk-...\n" > ~/.config/openai/cfg && chmod 600 ~/.config/openai/cfg
# For Gemini via the same proxy:
#   mkdir -p ~/.config/gemini && chmod 700 ~/.config/gemini
#   printf "AIza...\n" > ~/.config/gemini/cfg && chmod 600 ~/.config/gemini/cfg

OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://api.openai.com \
  OPENAI_MODEL=gpt-4o-audio-preview \
  PORT=8000 ./serve.sh
```

**NatureLM-audio v1.0 / Audio Flamingo Next / Qwen3-Omni** (GPU required):

```bash
# NatureLM v1.0 — start server
cd examples/servers/naturelm-v1.0
uv venv && uv pip install -r requirements.txt && . .venv/bin/activate
PORT=8000 ./serve.sh
```

### 3. Run the benchmark

With any launcher running at `http://127.0.0.1:8000`:

```bash
# Quick smoke check (3 tasks, 5 examples each):
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_smoke \
  --limit 5

# Full evaluation (22 tasks, all examples):
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_core \
  --output-dir results/my_run
```

Results are written to `--output-dir` (default `results/<run_id>/`):

| File | Content |
|---|---|
| `predictions.jsonl` | Raw launcher responses |
| `processed_predictions.jsonl` | Post-processed predictions with ground truth |
| `scored_predictions.jsonl` | Post-processed predictions with computed scores |

### 4. Generate prompt/answer pairs (BeansPro)

To collect analysis artifacts (rendered prompts, **raw** model answers, **post-processed**
answers, and ground truth) for **BeansPro** subsets:

```bash
uv run beans-next pairs \
  --predict-url http://127.0.0.1:8000/predict \
  --model-tag naturelm_v1_0 \
  --k 100 \
  --subsets crow-description,alarm-call-presence
```

Outputs are written under:

- `results/prompt_answer_pairs/beans_next_v0_1_0/<run_id>/pairs.jsonl`
- `results/prompt_answer_pairs/beans_next_v0_1_0/<run_id>/manifest.json`
- `results/prompt_answer_pairs/beans_next_v0_1_0/<run_id>/sample_ids/<subset>.jsonl`

For convenience, you can also concatenate multiple model runs into a single analysis file:

- `results/prompt_answer_pairs/beans_next_v0_1_0/beans_next_pairs_ALL6_concat_20260428.jsonl`

### 4. YAML run config (reproducible runs)

```yaml
# my_run.yaml
model: gpt4o_audio_openai_api   # registry preset from beans_next/registry/model/
suite: beans_zero_core
limit: 50
out_dir: results/my_run
```

```bash
uv run beans-next run --config my_run.yaml
```

### NatureLM 1.1 checkpoint-specific configs

You can benchmark different NatureLM checkpoints by creating dedicated run config files
that point to the launcher URL you are serving. We include a ready-to-run example for:

- `gs://foundation-models/naturelm-audio-1.5/all_backup/merged_variations_f0_v5`
- config file: `configs/benchmarks/beans_zero_core_naturelm_v1_1_ckpt_merged_variations_f0_v5.yaml`

Start the launcher with your target checkpoint URI first:

```bash
NATURELM_GCS_CHECKPOINT_URI=gs://foundation-models/naturelm-audio-1.5/all_backup/merged_variations_f0_v5 \
  sbatch examples/slurm/serve_naturelm_v1_1.sh
```

Then run the matching config:

```bash
uv run beans-next run --config \
  configs/benchmarks/beans_zero_core_naturelm_v1_1_ckpt_merged_variations_f0_v5.yaml \
  -o results/naturelm_v1_1_ckpt_merged_variations_f0_v5_$(date +%Y%m%d)
```

To test your own checkpoints, duplicate that YAML and update:

- `models[0].inline.name` (unique identifier in outputs)
- `models[0].inline.description` (human-readable checkpoint note)
- launcher startup command (`NATURELM_GCS_CHECKPOINT_URI=<your gs://...>`)

You can also start from the generic template:

- `configs/benchmarks/beans_zero_core_naturelm_v1_1_checkpoint_template.yaml`

Available registry model presets:

| Preset | Model |
|---|---|
| `dummy_local_8000` | Deterministic stub (CPU) |
| `naturelm_v1_0_local_8000` | NatureLM-audio v1.0 |
| `naturelm_v1_1_local_8001` | NatureLM-audio v1.1 |
| `gpt4o_audio_openai_api` | GPT-4o-audio-preview |
| `gemini_openai_api` | Gemini (any version) |
| `qwen3_omni_vllm_local_8000` | Qwen3-Omni-7B via vLLM |
| `af_next_local_8000` | Audio Flamingo Next |

### 5. Re-score without re-running inference

Every run saves three JSONL files to `--output-dir`:

| File | Content |
|---|---|
| `predictions.jsonl` | Raw model responses — unprocessed `predictions[0]` strings |
| `processed_predictions.jsonl` | After post-processing (fuzzy match, comma split, etc.) |
| `scored_predictions.jsonl` | After scoring (accuracy, F1, mAP, CIDEr, …) |

Use `predictions.jsonl` as input to `score-from-file` — it contains the original raw text so all three rescoring paths (normal pipeline, YES/NO judge, extractor judge) can be applied retroactively without re-running inference.

```bash
# Normal pipeline (post-process + metrics)
uv run beans-next score-from-file results/my_run/predictions.jsonl \
  -o results/my_run_rescored

# Add YES/NO judge pass
uv run beans-next score-from-file results/my_run/predictions.jsonl \
  --judge-url http://127.0.0.1:8010/predict \
  -o results/my_run_rescored

# Add extractor judge pass (judge converts raw output → clean label → full metrics)
uv run beans-next score-from-file results/my_run/predictions.jsonl \
  --judge-extract-url http://127.0.0.1:8010/predict \
  --task-type classification \
  -o results/my_run_rescored

# All three at once — output files never overlap (judge_* and judge_extracted_* prefixes)
uv run beans-next score-from-file results/my_run/predictions.jsonl \
  --judge-url http://127.0.0.1:8010/predict \
  --judge-extract-url http://127.0.0.1:8010/predict \
  --task-type classification \
  -o results/my_run_rescored
```

---

## Running on a SLURM cluster

See **[examples/slurm/README.md](examples/slurm/README.md)** for the two-job pattern (GPU serving job + CPU inference job) and per-model SLURM scripts.

Quick example:

```bash
# Submit serving job (GPU node), then inference job (CPU node)
SERVE_JOB=$(sbatch --parsable examples/slurm/serve_af3.sh)

BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB.url \
BEANS_PRO_SUITE=beans_zero_core \
sbatch --dependency=after:$SERVE_JOB examples/slurm/run_inference.sh
```

---

## Metrics

BEANS-Next applies deterministic scorers automatically based on task type. All scorers are in `beans_next.metrics`.

### Scoring by task type

| Task type | Post-processing | Scored by |
|---|---|---|
| `classification` | comma-split + Levenshtein fuzzy match to label vocab | `top1_accuracy`, `accuracy`, `precision`, `recall`, `f1` |
| `detection` | comma-split + fuzzy match | `average_precision`, `precision`, `recall`, `f1` |
| `captioning` | whitespace normalise | corpus **`cider`** in `summary.json` (per-sample scores empty); optional **`spider`** (CIDEr + SPICE) needs Java |
| `open_ended`, `counting`, `qa` | whitespace normalise | LLM judge (see below) |

### Classification

| Scorer | Description |
|---|---|
| `top1_accuracy` | Correct if prediction matches any option in a comma-separated target string (`"cat, feline"` → both `"cat"` and `"feline"` are correct) |
| `accuracy` | Exact-string match or multilabel exact-row match |
| `precision`, `recall`, `f1` | Support `average=` in `{"macro", "micro", "weighted", "binary"}` |

Label matching uses **Levenshtein fuzzy matching** (`max_distance=5`): a predicted string within 5 edit operations of a known label is snapped to that label before scoring.

Per-dataset fixed label vocabularies are loaded from `beans_next/registry/beans_zero_labels.json` (21 datasets, e.g. ESC-50 → 50 labels, `unseen-species-cmn` → 202). Inline `labels` in an eval-task YAML takes priority over the registry.

### Detection / multi-label

| Scorer | Description |
|---|---|
| `average_precision` | Per-label PR-curve integral, averaged across labels (macro) or pooled (micro) |
| `precision`, `recall`, `f1` | Multi-label, all `average=` modes supported |

### Captioning — CIDEr (default)

**CIDEr** — TF-IDF n-gram (1–4) cosine similarity with Gaussian length penalty, computed once over the full test split (corpus IDF). Pure Python / NumPy. Reported as `metrics.mean.cider` in `summary.json` (normalized to `[0, 1]` from the internal ×10 scale).

### Captioning — SPIDEr (optional, Java)

```
SPIDEr = (CIDEr / 10 + SPICE) / 2
```

**SPICE** — scene-graph F1 via Java subprocess. Requires Java ≥ 8 and Stanford CoreNLP 3.6.0 JARs. Download once:

```bash
uv run beans-next setup-spice
```

JARs are cached to `~/.cache/beans-next/spice/lib/`. The registered `spider()` scorer uses SPICE when available; otherwise SPICE is treated as `0.0` and a warning is logged.

### LLM-as-judge

For open-ended tasks (descriptions, counting, QA), deterministic metrics are insufficient because the same correct answer can be phrased many different ways — and models may refuse, hedge, or give structured vs. prose responses unpredictably.

beans-next supports three judge modes. Pick based on your task:

| Mode | Class | Flag | Output |
|---|---|---|---|
| **1 — Rubric judge** | `JudgeScorer` | `--judge-url` | Structured score (0–1) via `judge_scores_v1` endpoint |
| **2 — YES/NO judge** | `PredictV1Judge` | `--judge-url` | Binary `judge_accuracy` via `predictions_v1` endpoint |
| **3 — Extractor judge** | `PredictV1Extractor` | `--judge-extract-url` | Structured prediction → full metrics via `predictions_v1` endpoint |

**Mode 1** — dedicated `judge_scores_v1` endpoint (existing rubric-based judge):

```bash
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --judge-url   http://127.0.0.1:8010/judge \
  --suite beans_zero_core
```

Built-in rubric templates:

| Template id | Best for |
|---|---|
| `bioacoustic_open_qa_v1` | Soundscape descriptions, call-type explanations (default) |
| `bioacoustic_counting_v1` | "Count vocalizations per species" tasks; handles refusals, common names, flexible formatting |

Per-task template override: add `judge: bioacoustic_counting_v1` to the eval-task YAML.

**Mode 2** — YES/NO binary scoring via any `predictions_v1` model (e.g. Gemma 4):

```bash
# Run inline — judge fires after inference completes
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --judge-url   http://127.0.0.1:8010/predict \
  --suite beans_zero_core

# Or retroactively on saved predictions
uv run beans-next score-from-file results/my_run/processed_predictions.jsonl \
  --judge-url http://127.0.0.1:8010/predict \
  -o results/my_run_judge
```

Output artifacts: `judge_outputs.jsonl`, `judge_scored_predictions.jsonl`, `judge_summary.json`.

**Mode 3** — structured extraction: judge converts verbose model output to a clean label/description, then full metrics run:

```bash
uv run beans-next score-from-file results/my_run/processed_predictions.jsonl \
  --judge-extract-url http://127.0.0.1:8010/predict \
  --task-type classification \
  -o results/my_run_extracted
```

`--task-type` selects the extraction template (`classification`, `detection`, `captioning`). Output: `judge_extracted_scored_predictions.jsonl`, `judge_extracted_summary.json`.

Full details: **[docs/llm_judge.md](docs/llm_judge.md)** · Gemma 4 serving guide: **[docs/judge_model_gemma4.md](docs/judge_model_gemma4.md)**

---

## Adding a new launcher

A launcher is a self-contained FastAPI server. Minimal contract:

```
POST /predict  — accepts predictions_v1 request, returns predictions_v1 response
GET  /info     — capability document (name, model, audio_payload_types, …)
GET  /health   — readiness probe
```

Start from `examples/servers/hf_transformers/` (generic Tier-2 template).
Full contract spec: **[docs/http_contract.md](docs/http_contract.md)**.

Key rules:
- One response item per request `sample_id`; match by id, not array position
- HTTP 413 when `len(requests) > max_batch_size`
- Per-item errors go in `responses[i].error`; HTTP status stays 200
- Isolated venv — do **not** import `beans_next`

---

## Launcher conformance check

```bash
uv run bash scripts/check_launcher.sh http://127.0.0.1:<port>
```

---

## Development

```bash
uv run ruff check --fix .
uv run python -c "import beans_next"
uv run pytest -q
```

---

## Architecture

- **HTTP-only inference** — no in-process model execution in core.
- **Wire schema** — `predictions_v1` only.
- **No heavy deps in core** — `beans_next` does not depend on torch, transformers, or vLLM.

Full spec: `DESIGN.md`, `AGENT_SPEC.md`, `INCREMENTS.md`.
