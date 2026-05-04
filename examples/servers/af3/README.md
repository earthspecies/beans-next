## Tier-2 reference launcher: `af3`

Serves `nvidia/audio-flamingo-next-hf` (Audio Flamingo Next, 8B, BF16) via the BEANS-Next HTTP contract (`predictions_v1`).

**License**: `nvidia/audio-flamingo-next-hf` is released under the NVIDIA OneWay Noncommercial License.
Only non-commercial research use is permitted.

Upstream references agents must check before changing this launcher:

- GitHub project: `https://github.com/NVIDIA/audio-flamingo`
- Hugging Face model card: `https://huggingface.co/nvidia/audio-flamingo-next-hf`

### Qwen3-Omni serving notes

This README is also the current reference for Qwen3-Omni serving via the `vllm` launcher.
Anyone working on `examples/servers/vllm/**` or `examples/slurm/serve_vllm.sh` should read
this section before launching Qwen jobs.

Authoritative Qwen references:

- GitHub project: `https://github.com/QwenLM/Qwen3-Omni`
- Hugging Face model card: `https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Hugging Face captioner card: `https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner`

Current target checkpoint:

- Default model: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (Apache-2.0; about 35B BF16 params).
- It supports text, audio, image, and video input, with text/audio output for the Instruct model.
- For BEANS-Next ESC-50, request text output only. Do not use audio generation.
- Qwen's evaluation notes say to use no system prompt for benchmarks, put multimodal content first,
  then the textual task instruction. For ESC-50, keep the official BEANS-Next prompt text and attach
  audio before that text in the user message.
- `Qwen/Qwen3-Omni-30B-A3B-Thinking` is a poor fit for label-only ESC-50 scoring because it emits
  `<think>` reasoning before the final answer. Use the non-Thinking Instruct checkpoint for benchmark
  runs unless the task explicitly needs chain-of-thought style output.
- `Qwen/Qwen3-Omni-30B-A3B-Captioner` is fast and useful for dense audio descriptions, but its model
  card states that it accepts audio input only and no text prompt. It returns captions rather than
  label-shaped answers, so it is not the primary ESC-50 classification path.

#### Validated path: lean Instruct text-only serving

Best known ESC-50 path as of 2026-04-26:

- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Runtime: `vllm==0.18.0` + `vllm-omni==0.18.0` + `qwen-omni-utils`
- Hardware: one H100 80 GB
- Output: text only
- vLLM-Omni stages: one understanding/generation stage only; omit talker and code2wav stages
- Deploy config: `examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml`
- Adapter format: `VLLM_AUDIO_CONTENT_FORMAT=audio_url_data`
- Adapter batch size: `VLLM_MAX_BATCH_SIZE=1` for stable per-sample scoring and simple debugging
- Context: `VLLM_MAX_MODEL_LEN=8192` (ESC-50); use `16384`+ for Watkins (some items exceeded 8k)

Additional operational notes (2026-04-28):

- **Known-bad node(s)**: `slurm-8x-h100-1` frequently fails due to low scratch space and/or a full `/tmp`.
  Prefer excluding it for Qwen runs: `sbatch --exclude=slurm-8x-h100-1 ...`.
- **vLLM-Omni Speech API init touches `/tmp/voice_samples`** even for text-only pipelines. The Slurm
  launcher (`examples/slurm/serve_vllm.sh`) now works around this by pointing temp usage at scratch
  when possible and by ensuring `/tmp/voice_samples` is usable.
- **Avoid literal `/home/$USER` paths**: if `$USER` is not expanded, vLLM will try to write under
  `/home/$USER/...` and fail with `PermissionError: [Errno 13] Permission denied: '/home/$USER'`.
  The Slurm launcher now expands `$USER` inside HF cache env vars and `VLLM_EXTRA_ARGS`, but you should
  still prefer passing real paths (or keep quoting rules below).
- **Audio cap (recommended)**: for Qwen Omni classification runs, cap audio to 10s to reduce multimodal
  token pressure and improve stability (mirrors the OpenAI proxy path). The Slurm launcher applies this
  automatically for `Qwen/Qwen3-Omni-*` via:
  - `VLLM_ADAPTER_MAX_AUDIO_SECONDS=30`
  - `VLLM_ADAPTER_CANONICALIZE_WAV=1`

The vLLM-Omni YAML still uses `model_stage: thinker`; this is vLLM-Omni's internal name for the
audio/text understanding stage. It does **not** mean the checkpoint is the Thinking model.

Submit from the Slurm-visible repo root after syncing local edits:

```bash
# Important: keep the remote command in single quotes so $USER expands on the compute node
# (not on your local machine), and avoid single-quoting VLLM_EXTRA_ARGS so --download-dir
# does NOT end up as the literal path /scratch/$USER.
ssh slurm "cd /home/$USER/code/beans-next && \
  BEANS_PRO_HF_HOME=/scratch/$USER/hf_cache_instruct_lean \
  HF_HUB_DISABLE_XET=1 \
  VLLM_OMNI_INSTALL=1 \
  VLLM_INSTALL_VERSION=0.18.0 \
  VLLM_OMNI_VERSION=0.18.0 \
  VLLM_OMNI=1 \
  VLLM_AUDIO_CONTENT_FORMAT=audio_url_data \
  VLLM_OUTPUT_MODALITIES=text \
  VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  VLLM_TENSOR_PARALLEL_SIZE=1 \
  VLLM_MAX_MODEL_LEN=8192 \
  VLLM_MAX_BATCH_SIZE=1 \
  BEANS_PRO_PORT=8216 \
  VLLM_PORT=8116 \
  VLLM_EXTRA_ARGS="--stage-configs-path /home/$USER/code/beans-next/examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml --max-num-seqs 1 --download-dir /scratch/$USER/hf_cache_instruct_lean/hub" \
  sbatch --partition=h100-80 --gpus=1 --exclude=slurm-8x-h100-1 examples/slurm/serve_vllm.sh'
```

Read the serving URL from the URL file written by the job:

```bash
/mnt/home/$USER/beans-next-launchers/<job_id>.url  # NFS-mounted locally; or: ssh slurm cat $HOME/beans-next-launchers/<job_id>.url
```

#### Copy/paste canonical Slurm command (lean single-stage, 1×H100)

Run from the repo root on a machine that can `ssh slurm`:

```bash
ssh slurm "cd /home/\$USER/code/beans-next && \
  BEANS_PRO_HF_HOME=/home/\$USER/hf_cache_qwen3_omni_instruct_lean \
  HF_HUB_DISABLE_XET=1 \
  VLLM_OMNI_INSTALL=1 \
  VLLM_INSTALL_VERSION=0.18.0 \
  VLLM_OMNI_VERSION=0.18.0 \
  VLLM_OMNI=1 \
  VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  VLLM_AUDIO_CONTENT_FORMAT=audio_url_data \
  VLLM_MAX_BATCH_SIZE=1 \
  VLLM_TENSOR_PARALLEL_SIZE=1 \
  VLLM_MAX_MODEL_LEN=16384 \
  VLLM_EXTRA_ARGS=\"--stage-configs-path /home/\$USER/code/beans-next/examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml --download-dir /home/\$USER/hf_cache_qwen3_omni_instruct_lean\" \
  sbatch --partition=h100-80 --gpus=1 --exclude=slurm-8x-h100-1 examples/slurm/serve_vllm.sh"
```

Then read the URL file:

```bash
ssh slurm "cat \$HOME/beans-next-launchers/<JOB_ID>.url"
```

Run a smoke sample:

```bash
uv run python examples/servers/naturelm-v1.1/smoke_real_one.py \
  --predict-url "$PREDICT_URL" \
  --split esc50 \
  --timeout-sec 300
```

The validated smoke response for job `56707` was a label-shaped answer (`frog`) with no `<think>` block.

Run official ESC-50:

```bash
uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --suite beans_zero_esc50_official \
  --limit 400 \
  --run-id qwen3_omni_instruct_lean_esc50_full400_<job_id> \
  --output-dir results/ingested/qwen3_omni_instruct_lean_esc50_full400_<job_id> \
  --workers 1
```

Validated result for job `56707`:

- Output dir: `results/ingested/i13/qwen3_omni_instruct_lean_esc50_full400_56707`
- Samples: `400/400`
- Errors: `0`
- Accuracy / top-1: `0.7975`
- Wall time: `105s`
- Wall throughput: `3.81 samples/sec`
- Mean request latency: `0.163s`
- Median request latency: `0.152s`
- p95 request latency: `0.268s`
- H100 memory: about `73.2 GiB / 81.6 GiB`

Comparison notes from the same debugging session:

- `Qwen/Qwen3-Omni-30B-A3B-Thinking` with a single text-output stage did run, but produced `<think>`
  rationales and scored `0.0` with the official single-label postprocessor despite successful HTTP
  inference.
- `Qwen/Qwen3-Omni-30B-A3B-Captioner` served through regular `vllm serve` and was much faster than
  the Thinking run, but it emits rich captions. It also scored `0.0` with the current ESC-50
  postprocessor even when captions described the correct source.
- The lean Instruct config is the only path in this session that produced label-shaped outputs and a
  strong official ESC-50 score.

#### Current solution paths for Qwen3-Omni

Do not treat the generic `vllm==0.11.0` adapter environment as the Qwen3-Omni solution. The upstream
Qwen docs and current vLLM issue tracker now point to two viable paths:

1. **vLLM-Omni / Qwen vLLM support branch** for OpenAI-compatible serving.
2. **Direct Transformers** for a custom FastAPI launcher.

For the vLLM-Omni path, build a fresh launcher environment with Python 3.12 and install with `uv`.
The vLLM-Omni quickstart uses `vllm==0.19.0` plus the `vllm-omni` package/source tree:

```bash
cd examples/servers/vllm
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/venvs/qwen3_omni_vllm_omni"
uv venv --python 3.12 --seed "$UV_PROJECT_ENVIRONMENT"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  vllm==0.19.0 --torch-backend=auto
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  vllm-omni qwen-omni-utils
```

If the PyPI `vllm-omni` package is behind the docs, install the source tree instead:

```bash
git clone https://github.com/vllm-project/vllm-omni.git /scratch/$USER/src/vllm-omni
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e /scratch/$USER/src/vllm-omni
```

Then serve Qwen3-Omni through the `--omni` entrypoint:

```bash
UV_PROJECT_ENVIRONMENT="/scratch/$USER/venvs/qwen3_omni_vllm_omni" \
uv run vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --host 127.0.0.1 \
  --port 8103 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --allowed-local-media-path /
```

For BEANS-Next Slurm runs with a prebuilt vLLM-Omni environment, set:

```bash
BEANS_PRO_SKIP_UV_SYNC=1 \
UV_PROJECT_ENVIRONMENT="/scratch/$USER/venvs/qwen3_omni_vllm_omni" \
VLLM_OMNI=1 \
VLLM_AUDIO_CONTENT_FORMAT=audio_url_data \
VLLM_OUTPUT_MODALITIES=text \
VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct \
sbatch examples/slurm/serve_vllm.sh
```

`VLLM_AUDIO_CONTENT_FORMAT=audio_url_data` makes the adapter send Qwen-compatible
`audio_url` data URLs instead of `input_audio`; `VLLM_OUTPUT_MODALITIES=text` asks vLLM-Omni to skip
audio generation for ESC-50.

For the Transformers path, use a separate environment because it intentionally conflicts with the
stock-vLLM adapter's Transformers 4.x pin:

```bash
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/venvs/qwen3_omni_transformers"
uv venv --python 3.12 --seed "$UV_PROJECT_ENVIRONMENT"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  "transformers>=5.2.0" accelerate qwen-omni-utils soundfile
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  -U flash-attn --no-build-isolation
```

A Transformers-backed BEANS-Next launcher should load
`Qwen3OmniMoeForConditionalGeneration` and `Qwen3OmniMoeProcessor`, prepare audio with
`qwen_omni_utils.process_mm_info`, call `model.disable_talker()` for the Instruct checkpoint, and
always generate with `return_audio=False`. This path is slower on MoE than vLLM-Omni but is the most
direct fallback when serving support changes.

#### Generic vLLM serve + BEANS-Next adapter

Use `examples/slurm/serve_vllm.sh` for cluster runs. That script now starts vLLM through
`uv run vllm serve`, not `python -m vllm`.

The reported failure:

```text
No module named vllm.__main__; 'vllm' is a package and cannot be directly executed
```

Root cause: the installed `vllm` package exposes a console script (`vllm serve`) but no importable
module entrypoint (`python -m vllm`). The actionable fix is to invoke the console script through
the launcher-local `uv` environment:

```bash
cd examples/servers/vllm
uv sync --group upstream
uv run vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 127.0.0.1 \
  --port 8103 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --allowed-local-media-path / \
  --tensor-parallel-size 1
```

Slurm submission pattern:

```bash
rsync -av --delete --exclude ".venv/" \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

ssh slurm "cd /home/$USER/code/beans-next && \
  VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  VLLM_TENSOR_PARALLEL_SIZE=1 \
  BEANS_PRO_PORT=8203 \
  sbatch examples/slurm/serve_vllm.sh"
```

After the job is `R` and the URL file exists, run local inference:

```bash
URL_FILE="/mnt/home/$USER/beans-next-launchers/<job_id>.url"
PREDICT_URL="$(sed -n '1p' "$URL_FILE" | tr -d '[:space:]')"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUTDIR="results/qwen3_omni_esc50_official/limit5_${TS}"

uv run beans-next run \
  --task-id beans_zero_esc50_official \
  --predict-url "$PREDICT_URL" \
  --limit 5 \
  --output-dir "$OUTDIR" \
  --run-id "qwen3_omni_limit5_<job_id>_${TS}"

bash scripts/validate_run_dir.sh "$OUTDIR"
```

Only after the `--limit 5` run passes should agents run `--limit 400`.

#### Qwen-specific adapter warning

Qwen's vLLM serve examples send audio with OpenAI-compatible content items shaped like:

```json
{"type": "audio_url", "audio_url": {"url": "https://.../audio.wav"}}
```

The current BEANS-Next vLLM adapter accepts BEANS-Next `base64_wav` requests and has historically emitted
`input_audio` content items. If vLLM starts successfully but rejects the chat request schema, update
`examples/servers/vllm/adapter.py` to emit `audio_url` items instead, either as a `data:audio/wav;base64,...`
URL or as a job-local temporary WAV file under `--allowed-local-media-path /`. Keep that change launcher-only.

#### Non-vLLM fallback paths

If vLLM remains incompatible on the cluster, do not keep relaunching the same failing job. Use one of
these fallbacks:

- **Transformers launcher fallback**: create or extend a launcher under `examples/servers/**` that loads
  `Qwen3OmniMoeForConditionalGeneration` and `Qwen3OmniMoeProcessor`, declares `qwen-omni-utils`
  and related GPU deps in the launcher-local `pyproject.toml`, installs with `uv sync`, runs with
  `uv run`, uses `return_audio=False`, and exposes the same `predictions_v1` endpoints. For batch
  ESC-50, call `model.disable_talker()` to save about 10 GB VRAM, as recommended by the model card.
- **Qwen Thinking fallback**: serve `Qwen/Qwen3-Omni-30B-A3B-Thinking` if the Instruct thinker/talker
  path blocks text-only evaluation. This is not the same model; record it explicitly.
- **DashScope/API fallback**: use a provider API only if credentials and an OpenAI-compatible audio
  request shape are available. Otherwise a small provider-specific proxy is required.

Observed Slurm note: vLLM `0.11.0` API server startup fails if `VLLM_USE_V1=0` is forced
(`AssertionError` in `build_async_engine_client_from_engine_args`). Do not set `VLLM_USE_V1=0` for
`vllm serve` on this cluster unless a later vLLM/Qwen release explicitly requires it.

Observed dependency note: vLLM `0.11.0` with Transformers 5.x fails during Qwen tokenizer caching:

```text
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

The launcher-local vLLM upstream group pins `transformers==4.57.6` because vLLM `0.11.0` does not yet
fully support Transformers 5.x tokenizers. Keep this pin isolated to `examples/servers/vllm`; do not
add Transformers to BEANS-Next core.

For ESC-50 audio-only retries, set the Slurm wrapper env var
`VLLM_LIMIT_MM_PER_PROMPT='{"audio":1,"image":0,"video":0}'`. The wrapper passes this JSON as one
argument to `uv run vllm serve`; avoid putting this JSON inside `VLLM_EXTRA_ARGS`, where shell quoting
can strip the double quotes before vLLM parses it.

Observed stock-vLLM blocker: after the fixes above, PyPI vLLM `0.11.0` still does not serve
`Qwen/Qwen3-Omni-30B-A3B-Instruct` or `Qwen/Qwen3-Omni-30B-A3B-Thinking` here. It logs
`TransformersForMultimodalLM has no vLLM implementation`, then fails in the generic Transformers
loader:

```text
ValueError: Unrecognized configuration class
<class 'transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe.Qwen3OmniMoeConfig'>
for this kind of AutoModel: AutoModel.
```

Do not keep retrying stock `vllm==0.11.0` for Qwen3-Omni ESC-50. The next serving path should be one
of:

- `vllm-omni` / the upstream Qwen3-Omni vLLM support branch, installed with `uv` in the launcher-local
  environment.
- A Transformers-backed FastAPI launcher using `Qwen3OmniMoeForConditionalGeneration` directly with
  `return_audio=False`.
- A provider/API proxy if credentials and an audio request shape are available.

### Endpoints

- **`GET /health`**: readiness probe (HTTP 200 when ready)
- **`GET /info`**: capability document; advertises `predictions_v1`
- **`POST /predict`**: batched inference; one response per `sample_id`; HTTP 413 if batch exceeds `max_batch_size`

### Stub mode (CPU-only conformance)

`AF3_STUB=1` returns deterministic hashed outputs without loading any weights.
Use this to run `scripts/check_launcher.sh` on CPU machines.

```bash
PORT=19084 AF3_STUB=1 ./serve.sh
```

### Audio Flamingo Next model notes

Treat the Hugging Face model card as the source of truth for runtime behavior:

- Default checkpoint: `nvidia/audio-flamingo-next-hf` (AF-Next-Instruct).
- Best for general audio QA, ASR/AST-style prompts, multi-turn chat, and direct assistant-style answers.
- Input audio should be mono `16 kHz`.
- The released processor is configured for long audio up to `1800` seconds (`30` minutes), internally handled in `30`-second windows.
- Prompting matters. For ESC-50 and other classification runs, make the task and output format explicit and keep the BEANS-Next official prompt unchanged unless the increment explicitly owns prompt changes.
- The model card also lists `nvidia/audio-flamingo-next-think-hf` for explicit reasoning and `nvidia/audio-flamingo-next-captioner-hf` for dense captions. Do not swap checkpoints for benchmark runs without recording the model id and revision.

### Real inference (GPU required)

Requires a CUDA GPU with enough VRAM for an 8B BF16 model. The current Slurm recipe has validated startup on a single A100 40 GB node.

Use the launcher-local `uv` environment; do not add `transformers`, `torch`, or other GPU dependencies to BEANS-Next core:

```bash
cd examples/servers/af3
uv sync --group gpu
PORT=19084 uv run ./serve.sh
```

The launcher intentionally pins a Transformers build with `audioflamingonext` support in both `pyproject.toml` and `requirements.txt`. Slurm uses `uv sync --group gpu`, so `pyproject.toml` is the authoritative dependency source for cluster jobs.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `AF3_BIND_HOST` | `127.0.0.1` | Bind address |
| `AF3_STUB` | unset | Set to `1` for stub mode |
| `AF3_MODEL` | `nvidia/audio-flamingo-next-hf` | HuggingFace model id |
| `AF3_MODEL_REVISION` | main | HuggingFace revision / git ref |
| `AF3_MAX_BATCH_SIZE` | `4` | Max items per `/predict` call |
| `PORT` | `19084` | Listen port |

### Slurm / hybrid ESC-50 runbook

Recommended pattern: run the AF-Next server on a Slurm GPU node, then run `beans-next` locally
pointing at the cluster endpoint (hybrid pattern). This avoids cluster CPU job stalls during audio
materialization and makes debugging faster (artifacts land locally).

1. Sync local edits into the NFS-visible repo before `sbatch`:

```bash
rsync -av --delete --exclude ".venv/" \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/
```

2. Submit the serve job:

```bash
ssh slurm "cd /home/$USER/code/beans-next && BEANS_PRO_PORT=8202 sbatch examples/slurm/serve_af3.sh"
```

3. Poll until the job is running, then wait for the URL file at `/mnt/home/$USER/beans-next-launchers/<job_id>.url`.

4. Run a minimal ESC-50 official check first (`--limit 5`):

```bash
URL_FILE="/mnt/home/$USER/beans-next-launchers/<job_id>.url"
PREDICT_URL="$(sed -n '1p' "$URL_FILE" | tr -d '[:space:]')"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUTDIR="results/af3_esc50_official/limit5_${TS}"

uv run beans-next run \
  --task-id beans_zero_esc50_official \
  --predict-url "$PREDICT_URL" \
  --limit 5 \
  --output-dir "$OUTDIR" \
  --run-id "af3_limit5_<job_id>_${TS}"

bash scripts/validate_run_dir.sh "$OUTDIR"
```

Only after `--limit 5` produces complete artifacts should you run `--limit 400`.

### AF3 Slurm troubleshooting notes

- Earlier AF-Next Slurm attempts failed before `/health` because the installed Transformers did not recognize the `audioflamingonext` architecture (`KeyError: 'audioflamingonext'`). The fix is the launcher-local git-pinned Transformers dependency in `examples/servers/af3/pyproject.toml`.
- Later Slurm job `56640` successfully installed the pinned Transformers build, loaded weights, reached `/health`, and wrote its URL file.
- Job `56640` then idled until the Slurm `12:00:00` time limit. Its server log shows the startup health probe but no `/predict` requests, and no AF3 ESC-50 artifacts were produced.
- Strongest current conclusion: the AF-Next server did not crash during inference. The benchmark/inference side was not launched, failed before reaching HTTP, or was lost with the start-only agent execution. The next agent should reuse the runbook above and start with `--limit 5`.
- Manager status: AF-Next server startup is successful. The only missing AF3 work is local inference against the Slurm URL file, followed by artifact validation and then the full `--limit 400` run.

#### Fixed blockers (April 2026)

- **CUDA driver / PyTorch build mismatch (critical)**: earlier AF3 jobs installed `torch==2.11.0+cu130`, but the A100 nodes expose an older NVIDIA driver (CUDA 12.4 era). This produced `torch.cuda.is_available() == False`, silently forcing CPU/offload and making ESC-50 inference extremely slow. Fix:
  - Pin Torch in the launcher (`examples/servers/af3/pyproject.toml`) and resolve it from the official PyTorch CUDA 12.4 wheel index (`https://download.pytorch.org/whl/cu124`) using `tool.uv` sources.
  - Regenerate `examples/servers/af3/uv.lock` and ensure Slurm runs `uv sync --group gpu`.
- **GPU sanity check in Slurm logs**: `examples/slurm/serve_af3.sh` now prints torch/CUDA details (and forces a tiny CUDA allocation) before starting Uvicorn. If CUDA is not usable, the job log shows why immediately.
- **Missing audio loader dependency (`librosa`)**: AF-Next's Transformers path calls an audio loader that requires `librosa`. Without it, `/predict` returns an error about `load_audio_librosa`. Fix: add `librosa` to the launcher's GPU dependency group.
- **Prompt/audio wiring for BEANS prompts**: BEANS prompts may omit the `<Audio><AudioHere></Audio>` placeholder. AF-Next expects `content` to be a list of typed items; the launcher now always converts messages to the expected typed format and attaches any remaining audio to a user message.
- **Generation config incompatibility (`temperature=0.0`)**: BEANS uses `temperature=0.0` to express greedy decoding. This Transformers build rejects `temperature=0.0` unless expressed as `do_sample=False`. The launcher normalizes non-positive temperatures to greedy decoding.

### Audio input

Audio is accepted as `base64_wav`, `file_path`, or `file_url` (see `predictions_v1` contract).
Prompts reference audio via the `<Audio><AudioHere></Audio>` placeholder.

### Conformance check (repo root)

```bash
uv run python scripts/with_uvicorn.py \
  --cwd examples/servers/af3 --cmd-cwd . \
  --app serve:app --host 127.0.0.1 --port 19084 \
  --env AF3_STUB=1 \
  -- uv run bash scripts/check_launcher.sh "http://127.0.0.1:19084"
```
