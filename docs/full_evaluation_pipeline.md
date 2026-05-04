# Full Evaluation Pipeline

End-to-end instructions for running the complete BEANS-Zero benchmark (`beans_zero_core`, 22 tasks) against every supported model. Covers remote development setup, model serving, inference, results retrieval, and rescoring.

---

## Overview

Every evaluation follows the same two-phase pattern:

```
Phase 1 — Serve          Phase 2 — Infer + Score
────────────────          ────────────────────────────────────────────────
model server (GPU)  ───►  beans-next run (CPU) → predictions.jsonl
                                               → scored_predictions.jsonl
                                               → summary.json
```

The core library never loads model weights. All inference is over HTTP (`predictions_v1`). Serving and inference are fully decoupled — you can re-score any completed run without touching the model.

**Suite used here**: `beans_zero_core` — 22 tasks (16 classification + 5 detection + 1 captioning), full test split, no sample cap.

---

## Quick reference

| Model | GPU VRAM | Serve script | Submit script | Run config |
|---|---|---|---|---|
| NatureLM v1.0 | ≥24 GB | `serve_naturelm_v1_0.sh` | `submit_beans_zero_core_naturelm_v1_0.sh` | `beans_zero_core_naturelm_v1_0.yaml` |
| NatureLM v1.1 | ≥24 GB | `serve_naturelm_v1_1.sh` | `submit_beans_zero_core_naturelm_v1_1.sh` | `beans_zero_core_naturelm_v1_1.yaml` |
| Audio Flamingo Next | ≥20 GB | `serve_af3.sh` | `submit_beans_zero_core_af3.sh` | `beans_zero_core_af3.yaml` |
| Qwen3-Omni-7B | ≥24 GB | `serve_qwen3_omni.sh` | `submit_beans_zero_core_qwen3_omni.sh` | `beans_zero_core_qwen3_omni.yaml` |
| GPT-4o-audio | None | `openai_compatible_proxy` | — (CPU only) | `beans_zero_core_gpt4o.yaml` |
| Gemini | None | `openai_compatible_proxy` | — (CPU only) | `beans_zero_core_gemini.yaml` |

All scripts and configs are under `examples/slurm/` and `configs/benchmarks/` respectively.

---

## 1. Remote development setup

If developing locally and running inference on a remote cluster, keep code in sync with rsync.

### Sync repo to cluster

```bash
# From your local machine — run from the parent of your local beans-next checkout.
# Replace <cluster> with your SSH alias (e.g. entry in ~/.ssh/config).

rsync -avz --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.git/' --exclude='results/' \
  beans-next/ <cluster>:~/code/beans-next/
```

One-liner alias to add to your local `~/.bashrc`:

```bash
alias sync-beans='rsync -avz --exclude=".venv/" --exclude="__pycache__/" --exclude="*.pyc" --exclude=".git/" --exclude="results/" ~/code/beans-next/ <cluster>:~/code/beans-next/'
```

### Sync results back to local

```bash
# Pull a specific run's artifacts from cluster to local.
rsync -avz <cluster>:~/beans-next-results/naturelm_v1_0_20260501/ \
  ~/code/beans-next/results/naturelm_v1_0_20260501/

# Or use SCRATCH if results are there:
rsync -avz <cluster>:/scratch/$USER/beans-next-results/ \
  ~/code/beans-next/results/
```

### Run locally (no cluster)

For API models (GPT-4o, Gemini) or local GPU machines, the cluster is not needed at all. See the per-model sections below for local instructions.

---

## 2. One-time setup (cluster)

### Install `uv` (if not present)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Clone repo and install

```bash
cd ~/code
git clone <repo-url> beans-next
cd beans-next
uv sync
```

### Create log directory

SLURM jobs write logs to `~/logs/`. Create it once:

```bash
mkdir -p ~/logs ~/beans-next-launchers ~/beans-next-results
```

### SPICE (optional, SPIDEr only) — one-time, CPU node

`beans_zero_captioning` uses **CIDEr** by default (no Java). Download the JARs only if you need **SPIDEr** (`spider`):

```bash
uv run beans-next setup-spice
```

JARs go to `~/.cache/beans-next/spice/lib/`. Java ≥ 8 must be available on the inference node.

---

## 3. Store API keys and tokens

Keys are auto-loaded from config files so they don't appear in command history or job environment exports.

### HuggingFace token (NatureLM v1.1 + gated weights)

```bash
mkdir -p ~/.config/huggingface && chmod 700 ~/.config/huggingface
printf 'hf_...\n' > ~/.config/huggingface/hf_token
chmod 600 ~/.config/huggingface/hf_token
```

Both serve scripts auto-read from this file. You can also pass `HF_TOKEN=hf_...` inline.

### OpenAI API key (GPT-4o-audio)

```bash
mkdir -p ~/.config/openai && chmod 700 ~/.config/openai
printf 'OPENAI_API_KEY=sk-...\n' > ~/.config/openai/cfg
chmod 600 ~/.config/openai/cfg
```

### Google AI Studio API key (Gemini)

```bash
mkdir -p ~/.config/gemini && chmod 700 ~/.config/gemini
printf 'AIza...\n' > ~/.config/gemini/cfg
chmod 600 ~/.config/gemini/cfg
```

---

## 4. Download model weights (one-time, cluster)

Run on a login node (has internet access). All weights go to the shared HuggingFace cache.

```bash
export HF_HOME=/scratch/shared/hf_cache

# NatureLM-audio v1.0
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('EarthSpeciesProject/naturelm-audio-1.0')
"

# NatureLM-audio v1.1 (gated — requires approved HF access)
HF_TOKEN=$(cat ~/.config/huggingface/hf_token) \
uv run python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('EarthSpeciesProject/naturelm-audio-1.1.00-private', token=os.environ['HF_TOKEN'])
"

# Audio Flamingo Next
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/audio-flamingo-next-hf')
"

# Qwen3-Omni-7B
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-7B')
"
```

**NatureLM v1.0 extra step**: the launcher requires the NatureLM-audio source code:

```bash
cd ~/code
git clone https://github.com/earthspecies/NatureLM-audio.git
# The serve script expects it at ~/code/NatureLM-audio (or set NATURELM_CODE_DIR).
```

---

## 5. Smoke test before full run

Always validate the full pipeline with a tiny run first. Use the dummy launcher (no GPU, no keys):

```bash
# Terminal 1 — start dummy server
cd examples/servers/dummy
PORT=8000 ./serve.sh

# Terminal 2 — smoke run (3 tasks, 5 examples each, ~30 seconds)
uv run beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --suite beans_zero_smoke \
  --limit 5 \
  -o results/smoke_test

# Check: scores should be non-zero, no errors in predictions.jsonl
cat results/smoke_test/summary.json | python -m json.tool
```

Once the pipeline is confirmed, proceed to the real model.

---

## 6. Running a full evaluation

### Option A: SLURM (recommended for GPU models)

Use the per-model submit scripts. Each script submits two SLURM jobs — a serving job (GPU) and an inference job (CPU) — and wires them together automatically.

**Submit from repo root** (all submit scripts must be run from repo root):

```bash
# NatureLM v1.0
bash examples/slurm/submit_beans_zero_core_naturelm_v1_0.sh

# NatureLM v1.1 (reads HF_TOKEN from ~/.config/huggingface/hf_token)
bash examples/slurm/submit_beans_zero_core_naturelm_v1_1.sh

# Audio Flamingo Next
bash examples/slurm/submit_beans_zero_core_af3.sh

# Qwen3-Omni-7B
bash examples/slurm/submit_beans_zero_core_qwen3_omni.sh

# Qwen3-Omni-72B (4 GPUs)
VLLM_MODEL_ID=Qwen/Qwen3-Omni-72B \
VLLM_TENSOR_PARALLEL_SIZE=4 \
bash examples/slurm/submit_beans_zero_core_qwen3_omni.sh
```

Each submit script prints both job IDs, the log paths, and the output directory:

```
Submitting serving job...
  Serving job: 56789
  Log: ~/logs/56789.log
Submitting inference job (depends on serve job 56789)...
  Inference job: 56790
  Log: ~/logs/56790.log
  Output: /scratch/$USER/beans-next-results/naturelm_v1_0_20260501_143022
```

Monitor progress:

```bash
squeue --me
tail -f ~/logs/56789.log   # serving job
tail -f ~/logs/56790.log   # inference job
```

Cancel a run:

```bash
scancel 56789 56790
```

**Output directory**: defaults to `$SCRATCH/beans-next-results/<model>_<timestamp>`. Override with:

```bash
BEANS_PRO_OUT_DIR=/my/path bash examples/slurm/submit_beans_zero_core_naturelm_v1_0.sh
```

#### Partition notes

- Serving jobs: GPU partition (`a100-40`, `h100-80`). To use h100-80:
  ```bash
  sbatch --partition=h100-80 examples/slurm/serve_naturelm_v1_0.sh
  ```
- Inference jobs: CPU partition (`cpu`). The inference script already sets `#SBATCH --partition=cpu`.

---

### Option B: Manual SLURM (two separate sbatch calls)

The submit scripts are thin wrappers around the underlying `serve_*.sh` + `run_inference.sh`. You can also run them manually for more control:

```bash
# 1. Submit serving job
SERVE_JOB=$(sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh)

# 2. Submit inference job with full suite, custom output dir
BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_RUN_ID=naturelm_v1_0_20260501 \
BEANS_PRO_OUT_DIR=$SCRATCH/results/naturelm_v1_0_20260501 \
sbatch --dependency=after:$SERVE_JOB examples/slurm/run_inference.sh
```

Available `run_inference.sh` environment variables:

| Variable | Default | Description |
|---|---|---|
| `BEANS_PRO_URL_FILE` | — | Path to URL file written by serving job (required) |
| `BEANS_PRO_SUITE` | `beans_zero_core` | Suite to run |
| `BEANS_PRO_LIMIT` | (none) | Cap examples per task; omit for full suite |
| `BEANS_PRO_OUT_DIR` | `~/beans-next-results/run_<job_id>` | Output directory |
| `BEANS_PRO_RUN_ID` | `slurm_<job_id>` | Run identifier in artifacts |
| `BEANS_PRO_DATA_SOURCE` | auto (`esp_data` for beans_zero_*) | `hf` or `esp_data` |
| `BEANS_PRO_DEBUG` | `0` | Set to `1` for verbose logging |

---

### Option C: Local (single machine with GPU)

Use the per-model run configs. Start the launcher in one terminal, run inference in another.

**NatureLM v1.0:**

```bash
# Terminal 1
cd examples/servers/naturelm-v1.0
HF_HOME=/scratch/shared/hf_cache PORT=8000 ./serve.sh

# Terminal 2 (wait for "Launcher ready")
uv run beans-next run \
  --config configs/benchmarks/beans_zero_core_naturelm_v1_0.yaml \
  -o results/naturelm_v1_0_$(date +%Y%m%d)
```

**NatureLM v1.1:**

```bash
# Terminal 1
cd examples/servers/naturelm-v1.1
HF_TOKEN=$(cat ~/.config/huggingface/hf_token) \
  HF_HOME=/scratch/shared/hf_cache PORT=8001 ./serve.sh

# Terminal 2
uv run beans-next run \
  --config configs/benchmarks/beans_zero_core_naturelm_v1_1.yaml \
  -o results/naturelm_v1_1_$(date +%Y%m%d)
```

**Audio Flamingo Next:**

```bash
# Terminal 1
cd examples/servers/af3
HF_HOME=/scratch/shared/hf_cache PORT=8000 ./serve.sh

# Terminal 2
uv run beans-next run \
  --config configs/benchmarks/beans_zero_core_af3.yaml \
  -o results/af3_$(date +%Y%m%d)
```

**Qwen3-Omni (two processes):**

```bash
# Terminal 1 — vLLM backend
HF_HOME=/scratch/shared/hf_cache \
  vllm serve Qwen/Qwen3-Omni-7B --host 127.0.0.1 --port 8001

# Terminal 2 — BEANS-Next adapter sidecar
cd examples/servers/vllm
VLLM_ADAPTER_STUB=0 \
  VLLM_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B \
  PORT=8000 ./serve.sh

# Terminal 3
uv run beans-next run \
  --config configs/benchmarks/beans_zero_core_qwen3_omni.yaml \
  -o results/qwen3_omni_$(date +%Y%m%d)
```

---

### Option D: API models (no GPU needed)

For GPT-4o and Gemini, no serving job is needed. The proxy runs on the same CPU as inference.

**GPT-4o-audio-preview:**

```bash
# Terminal 1 — proxy (reads key from ~/.config/openai/cfg)
cd examples/servers/openai_compatible_proxy
uv venv && uv pip install -r requirements.txt
OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://api.openai.com \
  OPENAI_MODEL=gpt-4o-audio-preview \
  PORT=8000 ./serve.sh

# Terminal 2
uv run beans-next run \
  --config configs/benchmarks/beans_zero_core_gpt4o.yaml \
  -o results/gpt4o_$(date +%Y%m%d)
```

**Gemini:**

```bash
# Terminal 1 — proxy (reads key from ~/.config/gemini/cfg)
cd examples/servers/openai_compatible_proxy
OPENAI_PROXY_STUB=0 \
  OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
  OPENAI_MODEL=gemini-2.5-flash \
  PORT=8000 ./serve.sh

# Terminal 2
uv run beans-next run \
  --config configs/benchmarks/beans_zero_core_gemini.yaml \
  -o results/gemini_$(date +%Y%m%d)
```

> **Cost note**: Run `--limit 5` first to verify auth and estimate per-task cost before committing to all 22 tasks.

---

### Option E: Side-by-side (all models at once)

Submit all serving jobs, then all inference jobs. They run concurrently if GPU quota allows:

```bash
JOB_NLM=$(sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh)
JOB_AF3=$(sbatch --parsable examples/slurm/serve_af3.sh)
JOB_QWN=$(VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B sbatch --parsable examples/slurm/serve_qwen3_omni.sh)

for JOB_ID in $JOB_NLM $JOB_AF3 $JOB_QWN; do
  BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$JOB_ID.url \
  BEANS_PRO_SUITE=beans_zero_core \
  BEANS_PRO_OUT_DIR=${SCRATCH}/beans-next-results/run_${JOB_ID} \
  sbatch --dependency=after:$JOB_ID examples/slurm/run_inference.sh
done

squeue --me
```

---

## 7. Output artifacts

Every run writes these files to `--output-dir` (or `BEANS_PRO_OUT_DIR` in SLURM):

| File | Content |
|---|---|
| `predictions.jsonl` | **Raw** model responses — `predictions[0]` strings, unprocessed |
| `processed_predictions.jsonl` | After post-processing (fuzzy match, comma split, whitespace normalisation) |
| `scored_predictions.jsonl` | Per-sample scores (accuracy, F1, mAP, SPIDEr) |
| `summary.json` | Aggregate metrics per task + model identity |
| `model_identity.json` | `/info` payload from the launcher |
| `checkpoint.json` | Resume state (completed sample IDs) |

For multi-task suites, each task's artifacts are in a subdirectory, with a top-level `run_summary.json` aggregating across tasks.

### Retrieve results from cluster

```bash
# Sync a specific run back to local
rsync -avz <cluster>:${SCRATCH}/beans-next-results/naturelm_v1_0_20260501/ \
  ~/code/beans-next/results/naturelm_v1_0_20260501/

# Or pull everything
rsync -avz <cluster>:${SCRATCH}/beans-next-results/ \
  ~/code/beans-next/results/
```

### Read summary metrics

```bash
python -c "
import json, sys
s = json.load(open('results/naturelm_v1_0_20260501/run_summary.json'))
for task, metrics in s.get('tasks', {}).items():
    print(f'{task}: {metrics}')
"
```

---

## 8. Rescoring without re-running inference

`predictions.jsonl` contains the raw model outputs — every rescoring path reads from it so you never need to re-run inference.

### Normal pipeline (re-apply post-processing + metrics)

```bash
uv run beans-next score-from-file \
  results/naturelm_v1_0_20260501/predictions.jsonl \
  -o results/naturelm_v1_0_rescored
```

### YES/NO judge pass (Mode 2)

Adds a `judge_accuracy` metric alongside normal scoring. Requires a judge model serving at `--judge-url` (e.g. Gemma 4 via vLLM, see `docs/judge_model_gemma4.md`):

```bash
uv run beans-next score-from-file \
  results/naturelm_v1_0_20260501/predictions.jsonl \
  --judge-url http://127.0.0.1:8010/predict \
  -o results/naturelm_v1_0_rescored
```

Additional output files: `judge_outputs.jsonl`, `judge_scored_predictions.jsonl`, `judge_summary.json`.

### Extractor judge pass (Mode 3)

Judge converts verbose raw output to a structured label, then scores with the full metrics pipeline. Use when the model's output format doesn't match what the parsers expect:

```bash
uv run beans-next score-from-file \
  results/naturelm_v1_0_20260501/predictions.jsonl \
  --judge-extract-url http://127.0.0.1:8010/predict \
  --task-type classification \
  -o results/naturelm_v1_0_extracted
```

Additional output files: `judge_extracted_scored_predictions.jsonl`, `judge_extracted_summary.json`.

### All three at once

Normal scoring + both judge passes run in one call, writing to non-overlapping files:

```bash
uv run beans-next score-from-file \
  results/naturelm_v1_0_20260501/predictions.jsonl \
  --judge-url http://127.0.0.1:8010/predict \
  --judge-extract-url http://127.0.0.1:8010/predict \
  --task-type classification \
  -o results/naturelm_v1_0_full_rescore
```

Full judge documentation: `docs/llm_judge.md`.

---

## 9. Comparing results across models

```bash
# Print accuracy / F1 for each completed run
for d in results/*/; do
  echo "=== $d ==="
  python -c "
import json, pathlib, sys
p = pathlib.Path('$d/run_summary.json')
if not p.exists():
    print('  (no run_summary.json)')
    sys.exit(0)
s = json.load(open(p))
metrics = s.get('metrics', s.get('summary', {}))
print(json.dumps(metrics, indent=2))
"
done
```

---

## 10. Resuming an interrupted run

If a SLURM job was killed mid-run (time limit, preemption), resume from the checkpoint:

```bash
# SLURM: add --resume to the existing output dir
BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/<new_serve_job>.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_OUT_DIR=/scratch/$USER/beans-next-results/naturelm_v1_0_20260501 \
sbatch --dependency=after:<new_serve_job> examples/slurm/run_inference.sh
```

The `BenchmarkArtifactWriter` detects the existing `checkpoint.json` and skips already-completed samples automatically — no flags needed.

For local dev:

```bash
uv run beans-next run \
  --config configs/benchmarks/beans_zero_core_naturelm_v1_0.yaml \
  --resume-from results/naturelm_v1_0_20260501 \
  -o results/naturelm_v1_0_20260501
```

---

## 11. Troubleshooting

**`SLURM_SUBMIT_DIR is not set`**: always submit from the repo root, not from inside `examples/slurm/`.

**`launcher did not become healthy`**: model is still loading or OOM. Check the serve log (`~/logs/<serve_job_id>.log`). AF3 takes up to 15 min; NatureLM v1.0 ~3 min; Qwen3-Omni-7B ~5 min.

**`BEANS_PRO_URL_FILE not found`**: serving job failed before writing the URL file. Check serve log for errors (missing weights, OOM).

**`esp_data not importable`**: the cluster's `esp_data` package is not in the inference venv. Force HuggingFace loading: `BEANS_PRO_DATA_SOURCE=hf sbatch ...`.

**`No dataset examples were loaded`**: HuggingFace download failed. Compute nodes may lack outbound internet — switch to `esp_data` or set `HF_HUB_OFFLINE=1` with a pre-populated `HF_HOME`.

**Empty `predictions.jsonl` / all errors**: check the `error` field on individual items. Common causes: audio format mismatch, GPU OOM mid-run, API rate limits, network timeout.

**API rate limits (OpenAI/Gemini)**: the proxy auto-retries on 429/5xx. Lower concurrency or add `OPENAI_PROXY_RETRIES=10`. Start with `--limit 5` before full suite.

**`HF_TOKEN required` for NatureLM v1.0**: v1.0 requires `HF_TOKEN` only in real mode because the base model (`meta-llama/Meta-Llama-3.1-8B-Instruct`) is gated. Ensure your HF account has approved access.

**Scratch disk full**: `scratch_guard.sh` is sourced by all serve scripts and prunes HF/wheel caches automatically. If it fails loudly, free space manually before resubmitting.
