# BEANS-Next SLURM scripts

Two patterns are supported:

- **Hybrid (primary)**: serve job runs on cluster GPU; `beans-next run` runs **locally on this host**
  pointing at the cluster endpoint. Artifacts land locally — no copy-back step.
- **Two-job**: serve job (GPU node) + inference job (CPU/GPU node), both on the cluster. Artifacts
  land on the cluster filesystem and must be rsynced back for local inspection.

**Recommended:** Prefer the **Hybrid** pattern. Running `beans-next run` locally while only serving
on Slurm makes debugging faster (artifacts land locally) and avoids potential cluster CPU job stalls
during audio materialization with remote storage backends.

```
serve_*.sh        GPU node — loads model, starts launcher, writes URL file
run_inference.sh  cluster CPU node (no GPU) — polls URL file, runs beans-next (two-job pattern only)
test_run_inference.sh  cluster CPU node — tiny run (defaults: beans_zero_core, limit 3)
```

## Cluster access

Login: `ssh slurm`. URL files written by Slurm jobs land at `$HOME/beans-next-launchers/<job_id>.url`.
On this machine, `/home` is NFS-mounted at `/mnt/home`, so URL files are readable locally at
`/mnt/home/$USER/beans-next-launchers/<job_id>.url` without rsync.

GPU partitions (in preference order): **`a100-40`** → **`h100-80`**.  
Check availability before submitting: `sinfo`, `squeue --me`.  
Override partition at submit time: `sbatch --partition=h100-80 examples/slurm/serve_naturelm_v1_0.sh`

## Quick start

If you only copy/paste one thing, use `examples/slurm/CLUSTER_MIN_REPRO.md`. It documents both
patterns with the minimal, execution-policy-safe recipe (fixed ports, no `&`, readiness via
`GET /health` polling).

### 1. Pre-download model weights (once, on a login node)

Most GPU partitions block outbound internet. Download weights to a shared filesystem first:

```bash
HF_HOME=/scratch/shared/hf_cache uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/audio-flamingo-next-hf')   # AF3
snapshot_download('EarthSpeciesProject/naturelm-audio-1.0')  # NatureLM v1.0
"
```

Set `HF_HOME=/scratch/shared/hf_cache` in all job scripts (already the default).

### 2. Submit a serving job

Check availability first: `sinfo`, `squeue --me`. Default partition is `a100-40`; use
`sbatch --partition=h100-80 <script>` if a100-40 is full.

```bash
# NatureLM v1.0 (a100-40 by default):
sbatch examples/slurm/serve_naturelm_v1_0.sh

# NatureLM v1.0 on h100-80:
sbatch --partition=h100-80 examples/slurm/serve_naturelm_v1_0.sh

# NatureLM v1.1 (gated weights):
HF_TOKEN=hf_... sbatch examples/slurm/serve_naturelm_v1_1.sh

# Audio Flamingo Next:
sbatch examples/slurm/serve_af3.sh

# Qwen3-Omni via vLLM:
VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B sbatch examples/slurm/serve_vllm.sh
```

Note the serving job id (shown by `sbatch`). Monitor with `squeue --me`.

`BEANS_PRO_URL_DIR` defaults to `$HOME/beans-next-launchers`. Since `/home` is NFS-shared, URL
files written by cluster jobs are readable on this local host without any rsync or SSH.

### Ports (fixed by default; overrideable)

These scripts default to **fixed ports** (many clusters require this for firewalling / security policy):

- `serve_naturelm_v1_0.sh`: `BEANS_PRO_PORT=8000`
- `serve_naturelm_v1_1.sh`: `BEANS_PRO_PORT=8001`
- `serve_af3.sh`: `BEANS_PRO_PORT=8002`
- `serve_vllm.sh`:
  - adapter sidecar: `BEANS_PRO_PORT=8003`
  - upstream vLLM: `VLLM_PORT=8103` (bound to `127.0.0.1`)

If your cluster allows it and you need multiple servers on the same node, override ports when submitting:

```bash
BEANS_PRO_PORT=8200 sbatch examples/slurm/serve_naturelm_v1_0.sh
```

### 3. Submit the inference job (two-job pattern only)

**Hybrid pattern**: skip this — run `beans-next run` locally instead (see `CLUSTER_MIN_REPRO.md`).

**Two-job pattern**: start with the **minimal reproducer** (`BEANS_PRO_LIMIT=1`) to validate
end-to-end wiring before any larger run.

```bash
SERVE_JOB_ID=12345

BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_LIMIT=1 \
BEANS_PRO_OUT_DIR=/scratch/$USER/results/af3_run_$SERVE_JOB_ID \
BEANS_PRO_COPY_RESULTS_TO_HOME=1 \
sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_inference.sh
```

Convenience tiny run (defaults to `beans_zero_core`, `--limit 3`, and copy-to-home enabled):

```bash
SERVE_JOB_ID=12345
BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/test_run_inference.sh
```

Copy-back note:

- If you set `BEANS_PRO_COPY_RESULTS_TO_HOME=1`, the inference job will copy artifacts into
  `$HOME/beans-next-results/ingested/<run_id>/` at the end. If `/home` is NFS-shared to your
  local machine, they will be visible locally without rsync.
- If you do not set it and you write `BEANS_PRO_OUT_DIR` under `$SCRATCH`, you must manually rsync
  the results back for local inspection (see “Results inspection” below).

`--dependency=after:$SERVE_JOB_ID` starts the inference job as soon as the serving job begins running (not after it ends). The inference script then polls for the URL file and polls `/health`, so it waits for the model to finish loading without any fixed “sleep N minutes then hope” delay.

Important: `examples/slurm/run_inference.sh` does **not** request GPUs, but its `#SBATCH --partition=...` line is a site-specific placeholder. On most clusters you should set it to a CPU (or general) partition before running.

If your site prefers “only start inference if the serve job succeeds”, use your preferred Slurm
dependency type (commonly `afterok:<job_id>`). The scripts still rely on URL-file + `/health` polling
for real readiness, so no fixed sleeps are required either way.

### Minimal cluster validation (REQUIRED: start with `--limit 1`)

Before attempting any large run, validate the end-to-end wiring with the smallest possible scope.

1. Submit a serving job (one of `serve_*.sh`) and capture the job id.
2. Submit an inference job with **`BEANS_PRO_LIMIT=1`** and a dedicated output directory:

```bash
SERVE_JOB_ID=12345

BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_LIMIT=1 \
BEANS_PRO_OUT_DIR=$SCRATCH/beans-next-results/validate_limit1_$SERVE_JOB_ID \
sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_inference.sh
```

If this fails, **stop** and debug using that 1-sample run output and logs. Do not broaden scope until this is green.

### Note on `uv sync` and offline clusters

These scripts run `uv sync` inside the Slurm job by default. On clusters where compute nodes cannot reach PyPI, you have two safe options:

- **Pre-build the environment on a login node** that has internet access, on a shared filesystem (recommended).
- **Skip `uv sync` inside the job** if your cluster environment is already populated (or you vendored wheels) (consult `uv` docs for your site policy).

All `examples/slurm/*.sh` scripts support:

- `BEANS_PRO_SKIP_UV_SYNC=1` to skip `uv sync` inside the Slurm job

### 4. Multiple inference runs against the same server

Submit the serving job once and reuse it for multiple inference jobs — each pointing at the same URL file:

```bash
SERVE_JOB_ID=12345

# Run 1 — minimal cap (recommended while validating / debugging)
BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_LIMIT=1 \
sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_inference.sh

# Run 2 — different suite (scale only after the minimal run is green)
BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/$SERVE_JOB_ID.url \
BEANS_PRO_SUITE=beans_zero_smoke \
BEANS_PRO_LIMIT=50 \
sbatch --dependency=after:$SERVE_JOB_ID examples/slurm/run_inference.sh
```

## Environment variables

### Serving jobs (all)

| Variable | Default | Description |
|---|---|---|
| `BEANS_PRO_URL_DIR` | `$HOME/beans-next-launchers` | Directory for URL files |
| `BEANS_PRO_PORT` | (see above) | Fixed port the launcher listens on |
| `BEANS_PRO_HOSTNAME` | (auto) | Optional hostname to write into the URL file (use if `socket.gethostname()` is not routable from the inference job) |
| `HF_HOME` | `$SCRATCH/hf_cache` | HuggingFace model cache — update to your cluster's shared cache path |

### `serve_naturelm_v1_1.sh`

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace token for gated weights |

### `serve_vllm.sh`

| Variable | Required | Description |
|---|---|---|
| `VLLM_MODEL_ID` | Yes | HuggingFace model id (e.g. `Qwen/Qwen3-Omni-7B`) |
| `VLLM_TENSOR_PARALLEL_SIZE` | No (default `1`) | GPUs for tensor parallelism; increase `--gpus` to match |
| `VLLM_PORT` | No (default `8103`) | Upstream vLLM port (internal, `127.0.0.1`) |

### `serve_af3.sh`

| Variable | Default | Description |
|---|---|---|
| `AF3_MODEL` | `nvidia/audio-flamingo-next-hf` | Model id override |

### `run_inference.sh`

| Variable | Required | Description |
|---|---|---|
| `BEANS_PRO_URL_FILE` | Yes | Path to URL file from serving job |
| `BEANS_PRO_SUITE` | No (default `beans_zero_core`) | Suite id |
| `BEANS_PRO_LIMIT` | No | Cap on examples per task |
| `BEANS_PRO_OUT_DIR` | No (default `$HOME/beans-next-results/run_<job_id>`) | Artifact output directory |
| `BEANS_PRO_CONFIG` | No | Path to a run config YAML (overrides `--suite`) |
| `BEANS_PRO_RUN_ID` | No (default `slurm_<job_id>`) | Run id label in artifacts |

## ESC-50 official (single-task) — Slurm inference entrypoint

For “paper comparable” ESC-50 runs, use the **official** eval task:
`beans_zero_esc50_official` (dataset instruction prompt + `top1_accuracy`).

This repo provides an ESC-50-only inference wrapper:

- `examples/slurm/run_esc50_official_inference.sh`

It uses the standard URL-file pattern (`$HOME/beans-next-launchers/<serve_job_id>.url`) and
delegates all URL-file + `/health` polling to `examples/slurm/run_inference.sh`.

Minimal reproducible Slurm submission (two-job pattern):

```bash
SERVE_JOB_ID=12345

BEANS_PRO_URL_FILE="$HOME/beans-next-launchers/${SERVE_JOB_ID}.url" \
BEANS_PRO_LIMIT=1 \
BEANS_PRO_OUT_DIR="/scratch/$USER/beans-next-results/esc50_official_limit1_${SERVE_JOB_ID}" \
BEANS_PRO_RUN_ID="esc50_official_${SERVE_JOB_ID}" \
BEANS_PRO_COPY_RESULTS_TO_HOME=1 \
sbatch --dependency=after:${SERVE_JOB_ID} examples/slurm/run_esc50_official_inference.sh
```

Readiness discipline (serve job):

- poll `squeue --me` until the serve job state is **`R` (Running)**
- then wait for the URL file to appear
- then poll `GET /health` until OK

For the hybrid pattern (serve on cluster, infer locally), see `examples/slurm/CLUSTER_MIN_REPRO.md`.

## Adapting to your cluster

- **Partition names**: replace `a100-40` with your cluster's GPU partition. For CPU inference jobs, change the partition and remove GPU lines.
- **CPU-only inference**: in `run_inference.sh`, change `#SBATCH --partition=a100-40` to your CPU partition and optionally remove `--cpus-per-gpu` in favour of `--cpus-per-task`.
- **Internet access**: if GPU nodes can't reach HuggingFace, download weights in advance (see above) and set `TRANSFORMERS_OFFLINE=1` in the serve scripts.
- **Time limits**: increase `--time` for large models (vLLM with a 70B model may need 2–3h just to load).

## URL file protocol

The serve scripts write a single line to `$BEANS_PRO_URL_DIR/<job_id>.url` (this is what `examples/slurm/run_inference.sh` reads via `BEANS_PRO_URL_FILE`):

```
http://<hostname>:<port>/predict
```

If your cluster’s node hostnames are not directly routable between partitions (common with split CPU/GPU fabrics), set `BEANS_PRO_HOSTNAME` when submitting the serve job to force a reachable name (e.g. an FQDN or an IP your site permits).

The serve scripts write this file **atomically** (write to `*.tmp` then rename), so the inference job never reads a partially-written URL.

The inference script polls for this file (default timeout: 30 minutes), normalizes it to a `/predict` URL (it also accepts a base URL like `http://<hostname>:<port>`), then **polls `GET /health`** at the base URL before starting `beans-next run` (default health timeout: 15 minutes).

You can override URL-file polling via:

- `BEANS_PRO_URL_WAIT_TIMEOUT_SEC` (default `1800`)
- `BEANS_PRO_URL_WAIT_INTERVAL_SEC` (default `5`)

You can override health polling via:

- `BEANS_PRO_HEALTH_TIMEOUT_SEC` (default `900`)
- `BEANS_PRO_HEALTH_INTERVAL_SEC` (default `5`)
- `BEANS_PRO_HEALTH_CONNECT_TIMEOUT_SEC` (default `2`)
- `BEANS_PRO_HEALTH_MAX_TIME_SEC` (default `5`)

You can also pass the URL directly to `beans-next run` via `--predict-url`, or pass a **path to a URL file**
via `--predict-url-file` (a local text file containing a single `http(s)://...` URL).

## Results inspection

**Hybrid pattern**: artifacts land locally in the directory you passed to `-o`. No copy-back needed.
Run `scripts/validate_run_dir.sh` directly on that directory.

**Two-job pattern**: artifacts land on the cluster filesystem. After the inference job completes,
copy `BEANS_PRO_OUT_DIR` back to `results/ingested/<run_id>/` on this host. That copied directory
is the input to the CPU-only ingestion loop (no launcher required locally).

### Copy-back: rsync patterns (recommended)

Run these on your local machine (this repo). Replace placeholders.

```bash
cd /home/$USER/code/beans-next

RUN_ID="<run_id_from_cluster>"  # recommended: the BEANS_PRO_RUN_ID you used on cluster
CLUSTER_RUN_DIR="<absolute_path_to_BEANS_PRO_OUT_DIR_on_cluster>"

mkdir -p "results/ingested/${RUN_ID}"

# Trailing slash on CLUSTER_RUN_DIR copies directory contents into RUN_ID/.
rsync -av --info=progress2 \
  "$USER@slurm:${CLUSTER_RUN_DIR}/" \
  "results/ingested/${RUN_ID}/"
```

If your site requires jumping through a login node, use your normal SSH jump configuration:

```bash
rsync -av --info=progress2 -e "ssh -J $USER@slurm" \
  "$USER@<compute-hostname>:${CLUSTER_RUN_DIR}/" \
  "results/ingested/${RUN_ID}/"
```

Minimal local loop (repo root):

```bash
bash scripts/validate_run_dir.sh "results/ingested/<run_id>"
uv run beans-next score-from-file "results/ingested/<run_id>/predictions.jsonl" -o "results/ingested/<run_id>__rescored"
```

For the deeper checklist and common failure modes, see `docs/results_ingestion.md`.
