# Agent prompts: serve + full eval pipeline on Slurm / proxy

Use this file to prompt other agents with a checklist and exact commands.
It includes (a) serve bring-up, (b) health + URL-file verification, and (c) end-to-end
eval execution (smoke + full runs), plus the “hybrid pattern” used for ESC-50/Watkins.

Notes
- If an agent claims they can’t submit `sbatch` “from this environment”, it usually means Slurm client tools aren’t installed locally. In that case, the agent must run commands on the Slurm login node via `ssh slurm` and execute `sbatch` there.
- Do not print or echo API tokens / HF tokens in logs. Use the token files and/or env vars quietly.
- URL coordination: Slurm serve scripts write `$HOME/beans-next-launchers/<job_id>.url` once `/health` passes.

Dataset policy (important)
- **Always use `esp_data`** for dataset loading in evaluations. Do **not** use HuggingFace dataset loading in these workflows.
- Slurm inference jobs: set `BEANS_NEXT_DATA_SOURCE=esp_data`.
  - Local inference (`uv run beans-next run ...`): pass `--data-source esp_data`.

BirdSet policy (important)
- BirdSet is effectively **open-set** for many models (free-form scientific names with extra prose/timestamps).
- The correct BirdSet prompt is **scientific-name**:
  - Prompt registry id: `birdset_species_v1` (file: `beans_next/registry/prompt/birdset_species_v1.yaml`)
- For later canonicalization / label matching (without forcing a closed-set prompt), BirdSet eval-task YAMLs include:
  - `scientific_labels:` (auto-synced from `esp_data.BirdSet` metadata)
  - Script to refresh those vocabularies: `uv run python scripts/birdset_sync_scientific_labels.py`
- Inference entrypoints:
  - Full suite: `BEANS_NEXT_SUITE=birdset_core` (8 subsets)
  - Single subset task id: `BEANS_NEXT_TASK_ID=birdset_nbp_test_5s` (and the other `birdset_*_test_5s`)

GCS prediction upload (important)
- **Always upload run artifacts to GCS** after a successful full run so colleagues can reparse results.
- Destination bucket: `gs://foundation-model-data/synthetic/predictions/<run_id>/`
- For `beans-next run` invocations: add `--upload-gcs` (uses default bucket automatically, appends run-id).
- For post-run uploads (e.g. after a Slurm inference job wrote artifacts to disk):
  ```bash
  uv run python -c "
  from beans_next.results import upload_run_artifacts
  from pathlib import Path
  upload_run_artifacts(Path('${OUT_DIR}'), 'gs://foundation-model-data/synthetic/predictions/${RUN_ID}')
  "
  ```
- Requires `google-cloud-storage`: `uv sync --extra gcs` (already installed on the cluster).

## Always do this first (before any `sbatch`): sync local repo → NFS

Slurm jobs run from the NFS-visible repo under `/home/$USER/code/beans-next`. If your active
workspace is on local disk, sync it to `/mnt/home` so the job sees your latest changes.

```bash
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/
```

## Agent prompt: NatureLM v1.0 (real) on Slurm

Prompt:
You are on the Slurm login node. Serve NatureLM-audio v1.0 (real mode) using `examples/slurm/serve_naturelm_v1_0.sh`.
Requirements:
- Submit from the repo root: `/home/$USER/code/beans-next`.
- Use port default `8000` unless the increment requires otherwise.
- Ensure the gated HF token is available: set `HF_TOKEN="$(< ~/.config/huggingface/hf_token)"` (and set `HUGGINGFACE_HUB_TOKEN` to the same value) before calling `sbatch`.
- If UV connectivity might be blocked, set `BEANS_NEXT_SKIP_UV_SYNC=1`.
- If you submit via `ssh slurm "..."`, be careful with `$(...)` expansion:
  - `ssh slurm "export HF_TOKEN=\"$(< ~/.config/huggingface/hf_token)\" ..."` will try to read the token file on the *local* machine first.
  - Fix: escape the substitution so it runs remotely (`\\$(< ...)`), or run exports as separate remote commands (examples below).
Commands (run on login node):
```bash
# Ensure Slurm sees latest code (local → NFS):
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

cd /home/$USER/code/beans-next
export HF_TOKEN="$(< ~/.config/huggingface/hf_token)"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
# Optional: BEANS_NEXT_SKIP_UV_SYNC=1
SERVE_JOB_ID="$(sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```
If you must submit from a machine without Slurm client tools, use `ssh slurm` safely:
```bash
SERVE_JOB_ID="$(
  ssh slurm "cd /home/\$USER/code/beans-next \
    && export HF_TOKEN=\"\$(< ~/.config/huggingface/hf_token)\" \
    && export HUGGINGFACE_HUB_TOKEN=\"\$HF_TOKEN\" \
    && sbatch --parsable examples/slurm/serve_naturelm_v1_0.sh"
)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```
Verification:
1. Poll `squeue -j $SERVE_JOB_ID` until state is `R`.
2. Wait for `$HOME/beans-next-launchers/$SERVE_JOB_ID.url` to appear and be non-empty.
   - This can take several minutes on first boot: NatureLM v1.0 downloads multiple checkpoint shards and only writes the URL file after `/health` returns 200.
   - If it’s slow, check `/home/$USER/logs/$SERVE_JOB_ID.log` for download / load progress and for uvicorn startup errors.
3. Read the endpoint from the URL file (it ends with `/predict`).
4. From the machine where you run `curl`, call:
   - `curl -sf "$BASE_URL/health"`
   - `curl -sf "$BASE_URL/info"`
Output back to me:
- `job_id`, Slurm node/partition, URL file path, `base_url`, and the JSON from `/info`.

## Agent prompt: NatureLM v1.1 (real) on Slurm

Prompt:
You are on the Slurm login node. Serve NatureLM-audio v1.1 (real mode) on Slurm using `examples/slurm/serve_naturelm_v1_1.sh`.
Critical detail:
- NatureLM v1.1 weights are gated on HuggingFace.
- Other agents reported that `~/.config/huggingface/hf_token` exists but the launcher might not auto-load it.
- Therefore: explicitly export `HF_TOKEN="$(< ~/.config/huggingface/hf_token)"` into the environment before submission (and also set `HUGGINGFACE_HUB_TOKEN`).
Choose one GCS checkpoint (real mode requires `NATURELM_GCS_CHECKPOINT_URI`):
1) Default / normal v1.1 checkpoint:
- `NATURELM_GCS_CHECKPOINT_URI=gs://foundation-models/naturelm-audio-1.1/base_model/1290000`
2) Multi-audio checkpoint (merged variations):
- `NATURELM_GCS_CHECKPOINT_URI=gs://foundation-models/naturelm-audio-1.5/all_backup/merged_variations_f0_v5`
Commands (run on login node):
```bash
# Ensure Slurm sees latest code (local → NFS):
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

cd /home/$USER/code/beans-next
export HF_TOKEN="$(< ~/.config/huggingface/hf_token)"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
# Pick ONE:
export NATURELM_GCS_CHECKPOINT_URI="gs://foundation-models/naturelm-audio-1.1/base_model/1290000"
# export NATURELM_GCS_CHECKPOINT_URI="gs://foundation-models/naturelm-audio-1.5/all_backup/merged_variations_f0_v5"
# Optional:
# - BEANS_NEXT_SKIP_UV_SYNC=1
# - BEANS_NEXT_DEBUG=1
SERVE_JOB_ID="$(sbatch --parsable examples/slurm/serve_naturelm_v1_1.sh)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```
Verification:
1. Poll `squeue -j $SERVE_JOB_ID` until state is `R`.
2. Wait for `$HOME/beans-next-launchers/$SERVE_JOB_ID.url` to appear and be non-empty.
3. Read endpoint from URL file and verify:
   - `curl -sf "$BASE_URL/health"`
   - `curl -sf "$BASE_URL/info"`
If the URL file never appears:
- Tail the Slurm log for the first authentication / HF gated access failure.
- If you see 401/403, confirm `HF_TOKEN` was set in the serve job environment.
Output back to me:
- `job_id`, node/partition, URL file path, base endpoint, `/health` and `/info`.

## Agent prompt: AF3 (Audio Flamingo Next) on Slurm

Prompt:
You are on the Slurm login node. Serve Audio Flamingo Next using `examples/slurm/serve_af3.sh`.
Known failure modes / fixes:
- **Slow/stuck startup with `/health=503` for a long time**: if the serve log shows
  “unauthenticated requests to the HF Hub”, the model download can be very slow or
  rate-limited. Fix: **export `HF_TOKEN` + `HUGGINGFACE_HUB_TOKEN` before `sbatch`**.

Commands:
```bash
# Ensure Slurm sees latest code (local → NFS):
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

cd /home/$USER/code/beans-next
# Strongly recommended (avoids slow HF downloads / rate limits):
export HF_TOKEN="$(< ~/.config/huggingface/hf_token)"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
# Default AF3_MODEL is already set in the script; override only if needed:
# export AF3_MODEL="nvidia/audio-flamingo-next-hf"
# Optional:
# - BEANS_NEXT_SKIP_UV_SYNC=1
# - BEANS_NEXT_DEBUG=1
SERVE_JOB_ID="$(sbatch --parsable examples/slurm/serve_af3.sh)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```
Verification:
1. Wait for job `R`.
2. Wait for `$HOME/beans-next-launchers/$SERVE_JOB_ID.url` and read the endpoint.
3. Verify:
   - `curl -sf "$BASE_URL/health"`
   - `curl -sf "$BASE_URL/info"`
Output back to me:
- `job_id`, endpoint, `/health` + `/info`.

## Agent prompt: Qwen3-Omni Instruct on Slurm (vLLM)

Prompt:
You are on the Slurm login node. Serve Qwen3-Omni Instruct via vLLM using `examples/slurm/serve_qwen3_omni.sh` (which execs `examples/slurm/serve_vllm.sh`).
Defaults:
- vLLM adapter sidecar port is fixed to `19082` by the script.
- vLLM upstream port is `19083`.
- Script defaults `VLLM_MODEL_ID` to `Qwen/Qwen3-Omni-7B`; override to the desired model.
If you want the known-good “lean” single-stage config from prior runs:
- Use `--exclude=slurm-8x-h100-1`
- Use `--partition=h100-80`
- Set `VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Use single-stage YAML: `examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml`
Commands (pick one):
1) Simple serve (7B default):
```bash
# Ensure Slurm sees latest code (local → NFS):
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

cd /home/$USER/code/beans-next
SERVE_JOB_ID="$(sbatch --parsable examples/slurm/serve_qwen3_omni.sh)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```
2) Lean 30B known-good style:
```bash
# Ensure Slurm sees latest code (local → NFS):
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

cd /home/$USER/code/beans-next
export VLLM_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
export VLLM_MAX_BATCH_SIZE="1"
export VLLM_TENSOR_PARALLEL_SIZE="1"
export VLLM_TENSOR_PARALLEL_SIZE="1"
export VLLM_OMNI_INSTALL="1"
export VLLM_INSTALL_VERSION="0.18.0"
export VLLM_OMNI_VERSION="0.18.0"
export VLLM_OMNI="1"
export VLLM_MAX_MODEL_LEN="16384"
export VLLM_AUDIO_CONTENT_FORMAT="audio_url_data"
export HF_HUB_DISABLE_XET="1"
export BEANS_NEXT_HF_HOME="/home/$USER/hf_cache_qwen3_omni_instruct_lean"
export VLLM_EXTRA_ARGS="--stage-configs-path /home/$USER/code/beans-next/examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml --download-dir /home/$USER/hf_cache_qwen3_omni_instruct_lean"
SERVE_JOB_ID="$(sbatch --parsable \
  --partition=h100-80 --gpus=1 --exclude=slurm-8x-h100-1 \
  examples/slurm/serve_vllm.sh)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```
Verification:
1. Wait for job `R`.
2. Wait for `$HOME/beans-next-launchers/$SERVE_JOB_ID.url` and read endpoint.
3. Verify:
   - `curl -sf "$BASE_URL/health"`
   - `curl -sf "$BASE_URL/info"`
Output back to me:
- `job_id`, endpoint, `/info` JSON.

## Agent prompt: OpenAI / Gemini via OpenAI-compatible proxy (Slurm CPU serve required for this sweep)

Prompt:
You are on the Slurm login node. Start the OpenAI-compatible proxy on a **Slurm CPU node** so the
server is not tied to this host.

Important:
- This launcher has no GPU dependency, but it **must** run where there is network egress to the
  upstream API.
- Do **not** print or echo keys. Keys should be loaded from protected cfg files:
  - `~/.config/openai/cfg` for OpenAI
  - `~/.config/gemini/cfg` for Gemini

For this sweep, **do not** start the proxy locally via `examples/servers/openai_compatible_proxy/serve.sh`.
Known gotchas (Gemini proxy):
- If the proxy fails to start with `address already in use`, the chosen `PORT` is taken.
  Pick a new port (e.g. `19086`) and retry.
- Some Gemini 2.5 models may intermittently return an upstream response that lacks
  assistant text; BEANS-Next surfaces this as:
  `upstream error: upstream response missing assistant content text`
  If you see this repeatedly:
  - Stay on `OPENAI_MODEL="gemini-2.5-pro"` (preferred) if it’s mostly working, but
    be aware you may see intermittent missing-text upstream errors.
  - If the missing-text errors are frequent enough to spoil the run, fall back to
    `OPENAI_MODEL="gemini-2.5-flash-lite"` (known to behave better under tight caps in
    prior runs) or `gemini-2.5-flash`.
  - Keep `OPENAI_PROXY_MAX_CONCURRENCY=1` and a modest `OPENAI_PROXY_MAX_BATCH_SIZE` (e.g. `8`) to reduce
    timeouts / partial responses.
Verification:
- After the Slurm job writes its `.url` file, verify:
  - `curl -sf "$BASE_URL/health"`
  - `curl -sf "$BASE_URL/info"`
Output back to me:
- health/info responses (no secrets).

### Deprecated (do not use in this sweep): local proxy bring-up

If you are debugging locally only (not part of this sweep), you can still run:
`examples/servers/openai_compatible_proxy/serve.sh` with the environment variables shown in the
launcher help. For sweep runs, always use the Slurm CPU serve scripts below.

### Slurm CPU: serve OpenAI + Gemini proxies (writes `.url` + validates `/predict`)

Prompt:
You are on the Slurm login node. Start one of the CPU-only proxy servers on a Slurm CPU node using the dedicated serve scripts:
- OpenAI (ChatGPT via OpenAI API): `examples/slurm/serve_openai_proxy_gpt4o.sh`
- Gemini (Google AI Studio OpenAI-compatible endpoint): `examples/slurm/serve_openai_proxy_gemini.sh`

Important:
- These jobs require **network egress** from the compute node to the upstream API.
- Do **not** print or echo keys. Keys should be loaded from protected cfg files:
  - `~/.config/openai/cfg` for OpenAI
  - `~/.config/gemini/cfg` for Gemini
- The scripts only write `$HOME/beans-next-launchers/<job_id>.url` after:
  1) `/health` returns 200, and
  2) a `/predict` smoke-check succeeds (ensures `/predict` returns valid `predictions_v1` JSON with non-empty predictions).

Commands (run on login node):
```bash
cd /home/$USER/code/beans-next

# OpenAI GPT proxy (default port 19085; override with BEANS_NEXT_PORT if needed)
OPENAI_PROXY_STUB=0 \
  SERVE_JOB_ID="$(sbatch --parsable examples/slurm/serve_openai_proxy_gpt4o.sh)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"

# OR Gemini proxy (default port 19086; override with BEANS_NEXT_PORT if needed)
OPENAI_PROXY_STUB=0 \
  SERVE_JOB_ID="$(sbatch --parsable examples/slurm/serve_openai_proxy_gemini.sh)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```

Verification:
1. Wait for job `R`.
2. Wait for `$HOME/beans-next-launchers/$SERVE_JOB_ID.url` to appear and be non-empty.
3. Read endpoint from the URL file and verify:
   - `curl -sf "$BASE_URL/health"`
   - `curl -sf "$BASE_URL/info"`
Output back to me:
- `job_id`, partition, node, URL file path, `PREDICT_URL`, and `/info` JSON.

## Agent prompt: Full eval pipeline (serve → health → url → eval slurm)

Prompt:
You are orchestrating a full BEANS-Next evaluation pipeline on Slurm for a GPU-backed model
server launched via `examples/slurm/serve_<model>.sh`.
Execution constraint:
- `sbatch` must be issued on the Slurm login node (use `ssh slurm "sbatch ..."` if your agent
  cannot run Slurm commands locally).
Pipeline (two-job pattern):
1. Submit the serve job (GPU) and capture `SERVE_JOB_ID`.
2. Wait until the server is healthy by waiting for the `.url` file written by the serve
   launcher (the serve scripts only write the URL file after `GET /health` passes).
3. As an extra safety check, fetch `/health` and `/info` from the `BASE_URL`.
4. Submit a smoke evaluation job (`test_run_inference.sh`, limit 1–5).
5. If smoke passes, submit the full evaluation job (`run_inference.sh`).
6. Optionally validate outputs with `scripts/validate_run_dir.sh`.
7. Shut down the serve job with `scancel`.
Template (fill `<PLACEHOLDERS>`):
```bash
# 0) Sync local repo → NFS (required before any sbatch)
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

cd /home/$USER/code/beans-next
# A) Serve job: model-specific submit command from earlier prompt blocks.
# Example placeholders:
#   <SERVE_SUBMIT_ENV_VARS> sbatch --parsable <SERVE_SCRIPT>
# Run on login node:
SERVE_JOB_ID="$(
  ssh slurm "cd /home/\$USER/code/beans-next && <SERVE_SUBMIT_ENV_VARS> sbatch --parsable <SERVE_SCRIPT>"
)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
# B) Wait for URL file to exist and be non-empty.
#    On this host, /home is NFS-mounted at /mnt/home, so we can wait locally.
LOCAL_URL_FILE="/mnt/home/$USER/beans-next-launchers/${SERVE_JOB_ID}.url"
until [ -s "${LOCAL_URL_FILE}" ]; do
  sleep 5
done
# C) Read endpoint from URL file and verify health/info.
PREDICT_URL="$(head -1 "${LOCAL_URL_FILE}" | tr -d '[:space:]')"
BASE_URL="${PREDICT_URL%/predict}"
curl -sf "${BASE_URL}/health"
curl -sf "${BASE_URL}/info"
# D) Smoke eval on cluster CPU/general partition
export BEANS_NEXT_SUITE="<SUITE>"   # or use BEANS_NEXT_TASK_ID + BEANS_NEXT_DATASET_NAME
export BEANS_NEXT_LIMIT="5"
export BEANS_NEXT_RUN_ID="smoke_${SERVE_JOB_ID}"
export BEANS_NEXT_COPY_RESULTS_TO_HOME="1"
export BEANS_NEXT_OUT_DIR="/scratch/$USER/beans-next-results/smoke_${SERVE_JOB_ID}"
ssh slurm "BEANS_NEXT_URL_FILE=\"\$HOME/beans-next-launchers/${SERVE_JOB_ID}.url\" \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_SUITE='${BEANS_NEXT_SUITE}' \
  BEANS_NEXT_LIMIT='${BEANS_NEXT_LIMIT}' \
  BEANS_NEXT_RUN_ID='${BEANS_NEXT_RUN_ID}' \
  BEANS_NEXT_COPY_RESULTS_TO_HOME='${BEANS_NEXT_COPY_RESULTS_TO_HOME}' \
  BEANS_NEXT_OUT_DIR='${BEANS_NEXT_OUT_DIR}' \
  sbatch examples/slurm/test_run_inference.sh"
# E) Full eval on cluster CPU/general partition
unset BEANS_NEXT_LIMIT
export BEANS_NEXT_RUN_ID="full_${SERVE_JOB_ID}"
export BEANS_NEXT_OUT_DIR="/scratch/$USER/beans-next-results/full_${SERVE_JOB_ID}"
ssh slurm "BEANS_NEXT_URL_FILE=\"\$HOME/beans-next-launchers/${SERVE_JOB_ID}.url\" \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_SUITE='${BEANS_NEXT_SUITE}' \
  BEANS_NEXT_RUN_ID='${BEANS_NEXT_RUN_ID}' \
  BEANS_NEXT_COPY_RESULTS_TO_HOME='${BEANS_NEXT_COPY_RESULTS_TO_HOME}' \
  BEANS_NEXT_OUT_DIR='${BEANS_NEXT_OUT_DIR}' \
  sbatch examples/slurm/run_inference.sh"
# F) (Optional) validate artifacts
# When BEANS_NEXT_COPY_RESULTS_TO_HOME=1, run_inference.sh copies to:
#   $HOME/beans-next-results/ingested/${RUN_ID}
# So you can validate:
#   bash scripts/validate_run_dir.sh "$HOME/beans-next-results/ingested/${BEANS_NEXT_RUN_ID}"
# G) Full artifacts on GCS (required for rescoring suite tasks)
# run_inference.sh already runs `gsutil -m rsync -r` over the *entire* ingested tree when
# BEANS_NEXT_COPY_RESULTS_TO_HOME=1 (default) and BEANS_NEXT_UPLOAD_GCS=1 (default).
# Do NOT use `upload_run_artifacts()` for suite runs — it only uploads flat files in one
# directory, not suite/<suite_id>/<task_id>/*.jsonl.
# Backfill from a complete local tree (scratch or NFS):
#   LOCAL_SRC="$HOME/beans-next-results/ingested/${BEANS_NEXT_RUN_ID}" \
#     BEANS_NEXT_GCS_REL_PATH="<same suffix as under beans-next-results/>" \
#     bash scripts/sync_beans_next_results_to_gcs.sh
# H) Shutdown serve job after eval completion
ssh slurm "scancel ${SERVE_JOB_ID}"
```
Output back to me:
- `SERVE_JOB_ID`
- `BASE_URL`
- smoke eval Slurm job id (if available) + full eval Slurm job id (if available)
- validation result (pass/fail) if you ran `validate_run_dir.sh`
- GCS URIs uploaded (list from step G)

## Agent prompt: BirdSet core (all 8 subsets) via Slurm serve + inference job

Prompt:
You are running the full BirdSet core suite (`birdset_core`) end-to-end on Slurm:
server on Slurm GPU (or CPU proxy) + inference on Slurm CPU.

Requirements:
- Always use `esp_data`: set `BEANS_NEXT_DATA_SOURCE=esp_data`.
- Use scientific-name prompt wiring (BirdSet tasks use `birdset_species_v1`).
- Refresh BirdSet scientific label vocabularies before a big run:
  - `uv run python scripts/birdset_sync_scientific_labels.py`

Execution template (two-job pattern; adapt `<SERVE_SCRIPT>`):
```bash
# 0) Sync repo → NFS (required before any sbatch)
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

cd /home/$USER/code/beans-next

# 1) Submit serve job (examples: serve_af3.sh, serve_naturelm_v1_1.sh, serve_openai_proxy_gpt4o.sh)
SERVE_JOB_ID="$(sbatch --parsable <SERVE_SCRIPT>)"
echo "SERVE_JOB_ID=$SERVE_JOB_ID"

# 2) Submit inference job (depends on serve job)
RUN_ID="birdset_core_${SERVE_JOB_ID}_$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="/scratch/$USER/beans-next-results/${RUN_ID}"
INFER_JOB_ID="$(
  BEANS_NEXT_URL_FILE=\"$HOME/beans-next-launchers/${SERVE_JOB_ID}.url\" \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_SUITE=birdset_core \
  BEANS_NEXT_RUN_ID=\"$RUN_ID\" \
  BEANS_NEXT_OUT_DIR=\"$OUT_DIR\" \
  sbatch --parsable --dependency=after:\"$SERVE_JOB_ID\" examples/slurm/run_inference.sh
)"
echo "INFER_JOB_ID=$INFER_JOB_ID"
echo "OUT_DIR=$OUT_DIR"
```

Optional smoke run:
- Set `BEANS_NEXT_LIMIT=5` (or `1`) on the inference job submission to validate wiring.

Outputs to report back:
- serve job id + URL file path
- inference job id + OUT_DIR
- `summary.json` paths for each BirdSet eval task under the suite

## Agent prompt: What inference scripts do (so agents don’t duplicate logic)

Prompt:
When launching evaluation:
- `examples/slurm/test_run_inference.sh` is a tiny smoke run (defaults: suite `beans_zero_smoke`, limit `1`).
- `examples/slurm/run_inference.sh` is the full inference job.
  It *already*:
  1) waits for `BEANS_NEXT_URL_FILE` to exist
  2) polls `GET /health` from the server
  3) then runs `uv run beans-next ...` to materialize predictions + summaries
Therefore, the orchestration logic is:
- Wait for the serve job URL file + verify health externally once (as requested).
- Then submit `test_run_inference.sh` and `run_inference.sh`.
- After `run_inference.sh` completes, **upload artifacts to GCS** (step G in the full pipeline above).
  The Slurm inference scripts do **not** upload automatically; the upload is a manual post-run step.

## Agent prompt: Full BEANS-Zero core suite (all subsets) via run-config (recommended)

Prompt:
You are running the **full** `beans_zero_core` suite (all BEANS-Zero subsets) against an already
running launcher discovered via the URL-file protocol.

Requirements:
- Dataset backend policy: **always** `esp_data` (`BEANS_NEXT_DATA_SOURCE=esp_data`).
- Smoke-first: **submit smoke** (limit 1–5) before any uncapped run.
- Use a run-config YAML (do not enumerate tasks by hand).

Run-configs (esp_data, full `beans_zero_core`):
- `configs/benchmarks/beans_zero_core_naturelm_v1_0_esp_data.yaml`
- `configs/benchmarks/beans_zero_core_naturelm_v1_1_esp_data.yaml`
- `configs/benchmarks/beans_zero_core_af3_esp_data.yaml`
- `configs/benchmarks/beans_zero_core_qwen3_omni_esp_data.yaml`
- `configs/benchmarks/beans_zero_core_gpt4o_esp_data.yaml`
- `configs/benchmarks/beans_zero_core_gemini_esp_data.yaml`

Smoke + full inference job pattern (run on Slurm login node; fill placeholders):
```bash
cd /home/$USER/code/beans-next

SERVE_JOB_ID="<SERVE_JOB_ID>"
URL_FILE="$HOME/beans-next-launchers/${SERVE_JOB_ID}.url"

CONFIG_PATH="<CONFIG_PATH_FROM_LIST_ABOVE>"
INC="<INC>"                 # e.g. i18
MODEL_DIR="<MODEL_DIR>"     # e.g. naturelm_v1_0, naturelm_v1_1, af3, qwen3_omni, gpt4o, gemini
SUBSET_DIR="beans_zero_core"
TS="$(date -u +%Y%m%d_%H%M%S)"

SMOKE_RUN_ID="smoke_${MODEL_DIR}_${SUBSET_DIR}_${TS}"
FULL_RUN_ID="full_${MODEL_DIR}_${SUBSET_DIR}_${TS}"

SMOKE_OUT_DIR="/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${SMOKE_RUN_ID}"
FULL_OUT_DIR="/scratch/${USER}/.cache/beans-next-results/${INC}/${MODEL_DIR}/${SUBSET_DIR}/${FULL_RUN_ID}"

# Smoke first (limit 5 recommended)
SMOKE_JOB_ID="$(
  BEANS_NEXT_URL_FILE="$URL_FILE" \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_CONFIG="$CONFIG_PATH" \
  BEANS_NEXT_LIMIT=5 \
  BEANS_NEXT_RUN_ID="$SMOKE_RUN_ID" \
  BEANS_NEXT_OUT_DIR="$SMOKE_OUT_DIR" \
  sbatch --parsable examples/slurm/test_run_inference.sh
)"
echo "SMOKE_JOB_ID=$SMOKE_JOB_ID"

# Full run (uncapped) only after smoke passes
FULL_JOB_ID="$(
  BEANS_NEXT_URL_FILE="$URL_FILE" \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_CONFIG="$CONFIG_PATH" \
  BEANS_NEXT_RUN_ID="$FULL_RUN_ID" \
  BEANS_NEXT_OUT_DIR="$FULL_OUT_DIR" \
  sbatch --parsable --dependency=afterok:"$SMOKE_JOB_ID" examples/slurm/run_inference.sh
)"
echo "FULL_JOB_ID=$FULL_JOB_ID"
```

Notes (serve env vars; set before `sbatch` of serve scripts):
- NatureLM v1.0: export `HF_TOKEN="$(< ~/.config/huggingface/hf_token)"` and `HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"`.
- NatureLM v1.1: export `HF_TOKEN` + `HUGGINGFACE_HUB_TOKEN`, and
  `NATURELM_GCS_CHECKPOINT_URI="gs://foundation-models/naturelm-audio-1.1/base_model/1290000"` (**non-multiaudio**).
- AF3: strongly recommended to export `HF_TOKEN` + `HUGGINGFACE_HUB_TOKEN` to avoid slow HF downloads / rate limits.
- OpenAI/Gemini proxies: do not echo keys; use `~/.config/openai/cfg` / `~/.config/gemini/cfg` as described above.

## Agent prompt: Hybrid eval workflow (Slurm serve + local `beans-next run`) — Watkins/Lifestage/Calltype subsets of BEANS-Zero

Prompt:
You are running the BEANS-Zero benchmarks on Watkins/Lifestage/Calltype subsets with the hybrid pattern:
`examples/slurm/serve_<model>.sh` on Slurm GPU + local `uv run beans-next run` on this host
using the server URL from `$HOME/beans-next-launchers/<job>.url`.
This matches how we executed ESC-50 official / Watkins / Beans-Pro tasks in our earlier
runs: serve lifecycle is owned by you, inference runs locally (so artifacts land locally).
Execution template (NatureLM v1.0 shown as example; adapt `<SERVE_SCRIPT>`, `<PORT>`, and `<MODEL_DIR>`):
```bash
# 0) Sync local repo → NFS path used by Slurm jobs
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

# Always use esp_data for dataset loading in BEANS-Next runs.
export BEANS_NEXT_DATA_SOURCE="esp_data"

# 1) Submit serve job on login node (fixed port avoids collisions)
ssh slurm "cd /home/$USER/code/beans-next && BEANS_NEXT_PORT=<PORT> BEANS_NEXT_DEBUG=1 sbatch <SERVE_SCRIPT>"
# → capture <SERVE_JOB_ID>

# 2) Poll until Slurm job is RUNNING
ssh slurm "squeue --me --noheader -j <SERVE_JOB_ID>"

# 3) Read URL file + derive URLs
URL_FILE="/mnt/home/$USER/beans-next-launchers/<SERVE_JOB_ID>.url"
PREDICT_URL="$(head -1 "$URL_FILE")"
BASE_URL="${PREDICT_URL%/predict}"
# 4) Verify health + identity
curl -fsS "${BASE_URL}/health"
curl -fsS "${BASE_URL}/info"
# 5) Smoke + (optionally) full run per task
for TASK_ID in beans_zero_watkins beans_zero_lifestage beans_zero_call_type; do
  TS="$(date -u +%Y%m%d_%H%M%S)"
  OUT_DIR="results/ingested/i14/<MODEL_DIR>/${TASK_ID}/limit5_<SERVE_JOB_ID>_${TS}"
  # Smoke (limit 5) — no upload for smoke runs
  uv run beans-next run \
    --task-id "${TASK_ID}" \
    --predict-url "${PREDICT_URL}" \
    --data-source esp_data \
    --limit 5 \
    --output-dir "${OUT_DIR}" \
    --run-id "$(basename "${OUT_DIR}")"
  bash scripts/validate_run_dir.sh "${OUT_DIR}"
  # Confirm official prompt version (avoid wrong prompt like classification_bioacoustic_v1)
  python - <<'PY' "${OUT_DIR}"
import json,sys
p=sys.argv[1]+"/summary.json"
d=json.load(open(p))
print("prompt_version=", d.get("prompt_version"))
PY
  # If smoke looks good, full run (no --limit) — upload artifacts to GCS
  OUT_DIR_FULL="results/ingested/i14/<MODEL_DIR>/${TASK_ID}/full_<SERVE_JOB_ID>_${TS}"
  uv run beans-next run \
    --task-id "${TASK_ID}" \
    --predict-url "${PREDICT_URL}" \
    --data-source esp_data \
    --output-dir "${OUT_DIR_FULL}" \
    --run-id "$(basename "${OUT_DIR_FULL}")" \
    --upload-gcs
  bash scripts/validate_run_dir.sh "${OUT_DIR_FULL}"
done
# 6) Shutdown serve job
ssh slurm "scancel <SERVE_JOB_ID>"
```
Output back to me:
- serve job id + URL file path
- `/health` and `/info` results
- for each TASK_ID: smoke and (if run) full `summary.json` prompt_version + primary metric

## Agent prompt: ESC-50 subset of BEANS-Zero official evaluation workflow (Slurm serve + inference job)

Prompt:
You are running `beans_zero_esc50_official` (ESC-50 official) end-to-end using the Slurm
2-job pattern:
server on Slurm GPU + inference on Slurm CPU.
Use `examples/slurm/run_esc50_official_inference.sh` for the inference job.
Exact template (Slurm GPU serve + Slurm CPU inference + local validation):
```bash
# 0) Sync repo → NFS
rsync -av --delete --exclude '.venv/' \
  /home/$USER/code/beans-next/ \
  /mnt/home/$USER/code/beans-next/

# Always use esp_data for dataset loading in evaluation jobs.
export BEANS_NEXT_DATA_SOURCE="esp_data"

# 1) Submit serve job (example: NatureLM v1.0)
cd /home/$USER/code/beans-next
BEANS_NEXT_PORT=<PORT> BEANS_NEXT_DEBUG=1 sbatch examples/slurm/serve_naturelm_v1_0.sh
# → capture <SERVE_JOB_ID>

# 2) Wait for URL file on NFS
test -s "$HOME/beans-next-launchers/<SERVE_JOB_ID>.url"
# 3) Read URL file + verify
BASE_PREDICT_URL="$(head -1 "$HOME/beans-next-launchers/<SERVE_JOB_ID>.url")"
BASE_URL="${BASE_PREDICT_URL%/predict}"
curl -fsS "${BASE_URL}/health"
curl -fsS "${BASE_URL}/info"
# 4) Submit ESC-50 official inference job (limit 5 or 1 while debugging)
RUN_ID="esc50_official_limit5_<SERVE_JOB_ID>_$(date -u +%Y%m%d_%H%M%S)"
BEANS_NEXT_URL_FILE="$HOME/beans-next-launchers/<SERVE_JOB_ID>.url" \
  BEANS_NEXT_DATA_SOURCE=esp_data \
  BEANS_NEXT_DEBUG=1 \
  BEANS_NEXT_LIMIT=5 \
  BEANS_NEXT_TASK_ID=beans_zero_esc50_official \
  BEANS_NEXT_DATASET_NAME=esc50 \
  BEANS_NEXT_OUT_DIR="/home/$USER/code/beans-next/results/ingested/<INCREMENT>/naturelm_v1_0_esc50_official_esp_data/${RUN_ID}" \
  BEANS_NEXT_RUN_ID="${RUN_ID}" \
  sbatch examples/slurm/run_esc50_official_inference.sh
# 5) Ingest + validate (if needed)
# If BEANS_NEXT_COPY_RESULTS_TO_HOME=1 was used, artifacts already exist locally.
# Otherwise rsync from /home on Slurm to local checkout and validate:
bash scripts/validate_run_dir.sh results/ingested/<INCREMENT>/<...>/<RUN_ID>
```
Notes:
- `scripts/validate_run_dir.sh` is strict about `processed_predictions.jsonl` shape for classification;
  if it fails (e.g., targets is a string), stop and fix artifacts before scaling up.
- If you want a direct CPU-only inference against a running NatureLM v1.1 server URL,
  there are dedicated exception scripts:
  - `examples/slurm/run_esc50_official_cpu_inference_naturelm_v1_1.sh`
  - `examples/slurm/run_esc50_official_cpu_inference_naturelm_v1_0.sh`
Output back to me:
- serve job id + URL file path
- inference job id
- validate_run_dir.sh pass/fail

