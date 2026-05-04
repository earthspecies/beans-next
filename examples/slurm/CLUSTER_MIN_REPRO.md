# BEANS-Next on Slurm — cluster minimal reproducer

> **For agents:** read `examples/slurm/AGENT_RUNBOOK.md` instead of this doc. It contains
> the same execution steps plus the failure recovery ladder and finish log format in a single
> compact reference. This file is retained for human reference and architectural context.

Two patterns are documented here:

- **Hybrid (primary for I7)**: serve job runs on cluster GPU; `beans-next run` runs **locally on this
  host** pointing at the cluster endpoint. Artifacts land locally — no copy-back required.
- **Two-job (full-scale / production)**: serve job + Slurm inference job, both on the cluster.
  Artifacts land on the cluster filesystem and must be rsynced back.

Start with the **hybrid pattern** and `BEANS_PRO_LIMIT=1` to validate end-to-end wiring.

---

## Cluster access

Login node alias: `slurm` (configured in `~/.ssh/config`).

```bash
ssh slurm   # opens a shell on the login node
```

`/home` is **NFS-shared** across the whole cluster. On Slurm login/compute nodes it is mounted at
`/home`. On this local host, the same shared filesystem is mounted at **`/mnt/home`**. This means:

- URL files written by a Slurm job to `$HOME/beans-next-launchers/` (i.e. `/home/marius_miron_earthspecies_org/...` on Slurm)
  are readable on this host under `/mnt/home/marius_miron_earthspecies_org/...` without rsync.
- The repo under `/home/marius_miron_earthspecies_org/code/beans-next` on Slurm is the same content as
  `/mnt/home/marius_miron_earthspecies_org/code/beans-next` on this host.

If `/home` is not already mounted on this machine, see `slurm.md § Mount /home locally`.

---

## Hybrid pattern: serve on cluster, infer locally

### Assumptions (read once)

- **Shared home is mounted**: on Slurm nodes this is `/home`; on this host it is `/mnt/home`.
  The URL dir `$HOME/beans-next-launchers/` corresponds to `/mnt/home/marius_miron_earthspecies_org/beans-next-launchers/`
  on this machine.
- **Repo sync is required on this host**: this IDE workspace lives under local `/home/...` (ext4).
  Before submitting any Slurm job that runs from the NFS-visible repo path, sync your working tree
  into `/mnt/home`:

```bash
rsync -av --delete --exclude '.venv/' \
  "$HOME/code/beans-next/" \
  "/mnt/home/marius_miron_earthspecies_org/code/beans-next/"
```
- **Endpoint reachability (GCP)**: serve scripts auto-detect the node's **internal IP** via the GCP
  instance metadata API (`http://metadata.google.internal/...`) and write it to the URL file. Since
  this machine is on the same GCP VPC, the internal IP is directly routable — **no SSH tunnel
  needed** in normal operation. If metadata detection fails, the script falls back to
  `socket.gethostname()`; override with `BEANS_PRO_HOSTNAME=<ip>` if needed.
- **No fixed sleeps**: URL-file coordination + `GET /health` polling; no `&`, no sleep delays.
- **Partitions**: prefer `a100-40`; fall back to `h100-80`. Check before submitting:

```bash
sinfo                    # see all partitions + node states
squeue --me              # see your running/pending jobs
squeue --state=RUNNING   # see all running jobs (gauge queue load)
```

To use h100-80, pass `--partition=h100-80` at submit time (overrides the script's `#SBATCH` line):

```bash
sbatch --partition=h100-80 examples/slurm/serve_naturelm_v1_0.sh
```

### 0) Choose URL directory + local results directory

`BEANS_PRO_URL_DIR` defaults to `$HOME/beans-next-launchers` (Slurm-side path). On this host the
same directory is visible at `/mnt/home/marius_miron_earthspecies_org/beans-next-launchers`.

```bash
export BEANS_PRO_URL_DIR="$HOME/beans-next-launchers"
export BEANS_PRO_RESULTS_LOCAL="$(pwd)/results/local_inference"
mkdir -p "$BEANS_PRO_RESULTS_LOCAL"
```

### 1) Check node availability + submit the serve job (GPU)

```bash
# Check what's free before submitting:
sinfo
squeue --me

cd /home/marius_miron_earthspecies_org/code/beans-next

# NatureLM v1.0 (default partition: a100-40, fixed port: 8000)
SERVE_JOB_ID=$(sbatch examples/slurm/serve_naturelm_v1_0.sh | awk '{print $NF}')
echo "SERVE_JOB_ID=$SERVE_JOB_ID"

# Use h100-80 if a100-40 is full:
# SERVE_JOB_ID=$(sbatch --partition=h100-80 examples/slurm/serve_naturelm_v1_0.sh | awk '{print $NF}')

# NatureLM v1.1 (gated weights, port 8001)
# HF_TOKEN=hf_... SERVE_JOB_ID=$(sbatch examples/slurm/serve_naturelm_v1_1.sh | awk '{print $NF}')

# AF3 (port 8002)
# SERVE_JOB_ID=$(sbatch examples/slurm/serve_af3.sh | awk '{print $NF}')

# vLLM (adapter port 8003)
# VLLM_MODEL_ID=Qwen/Qwen3-Omni-7B SERVE_JOB_ID=$(sbatch examples/slurm/serve_vllm.sh | awk '{print $NF}')
```

If compute nodes can't reach PyPI, add `BEANS_PRO_SKIP_UV_SYNC=1`:

```bash
SERVE_JOB_ID=$(BEANS_PRO_SKIP_UV_SYNC=1 sbatch examples/slurm/serve_naturelm_v1_0.sh | awk '{print $NF}')
```

The serve job writes once healthy:

- URL file: `${BEANS_PRO_URL_DIR}/${SERVE_JOB_ID}.url`
- Contents: one line like `http://<hostname>:<port>/predict`

### 2) Monitor queue until job is RUNNING

**Do not proceed to step 3 until the job state is `R` (Running).** After `sbatch` the job enters
`PD` (Pending) state — it has not started and will not write the URL file yet. Queue wait time
depends on cluster load and can range from seconds to many minutes.

Poll the queue:

```bash
# One-shot check:
squeue --me

# Poll until no longer pending (repeat manually every ~30s, or use watch):
watch -n 30 squeue --me
# Exit watch with Ctrl+C once you see state = R.
```

Non-interactive loop (useful for scripted/agent use):

```bash
echo "Waiting for job ${SERVE_JOB_ID} to start..."
while squeue --me --noheader -j "${SERVE_JOB_ID}" 2>/dev/null | grep -q ' PD '; do
  echo "  still pending... $(date '+%H:%M:%S')"
  sleep 30
done
echo "Job ${SERVE_JOB_ID} is now running (or completed/failed). Check: squeue --me"
```

Once state is `R`, the job is loading model weights. The URL file will appear after `/health` passes
(typically a few minutes after the job starts). Only then does step 3 make sense.

### 3) Wait for URL file + read the endpoint

On this host, poll the URL file via the NFS mount at `/mnt/home`:

```bash
URL_FILE="${BEANS_PRO_URL_DIR}/${SERVE_JOB_ID}.url"

echo "Polling for URL file: $URL_FILE"
until [ -s "/mnt${URL_FILE}" ]; do sleep 5; done

PREDICT_URL=$(head -1 "/mnt${URL_FILE}" | tr -d '[:space:]')
echo "Endpoint: $PREDICT_URL"
```

If the NFS mount is not set up yet, poll via SSH instead:

```bash
until ssh slurm "test -s '$URL_FILE'" 2>/dev/null; do sleep 5; done
PREDICT_URL=$(ssh slurm "head -1 '$URL_FILE'" | tr -d '[:space:]')
echo "Endpoint: $PREDICT_URL"
```

### 4) (Fallback only) SSH port forward if GCP internal IP is not reachable

**Skip this step on GCP** — the serve scripts write the node's internal IP to the URL file, and
VMs on the same GCP VPC can reach it directly. Only needed if metadata auto-detection failed and
`PREDICT_URL` contains an unroutable hostname. In that case:

```bash
# Parse host + port from PREDICT_URL (e.g. http://gpu-node-42:8000/predict)
REMOTE_HOST=$(echo "$PREDICT_URL" | sed -E 's|https?://([^:/]+).*|\1|')
REMOTE_PORT=$(echo "$PREDICT_URL" | sed -E 's|.*:([0-9]+).*|\1|')

# Keep this running in a separate terminal (or add -f to background it).
ssh -N -L "${REMOTE_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" slurm
PREDICT_URL="http://127.0.0.1:${REMOTE_PORT}/predict"
echo "Using local tunnel: $PREDICT_URL"
```

### 5) Conformance check (from this host)

```bash
cd "$HOME/code/beans-next"
uv run bash scripts/check_launcher.sh "${PREDICT_URL%/predict}"
```

Expect: **PASS**.

### 6) Run beans-next locally (minimal reproducer: `--limit 1`)

Start with limit 1 to validate wiring before a larger run.

```bash
RUN_ID="hybrid_min_repro_${SERVE_JOB_ID}"
OUT_DIR="${BEANS_PRO_RESULTS_LOCAL}/${RUN_ID}"

uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --suite beans_zero_core \
  --limit 1 \
  --run-id "$RUN_ID" \
  -o "$OUT_DIR"

echo "Artifacts in: $OUT_DIR"
ls -la "$OUT_DIR"
```

If this fails, **stop** and debug using the single-sample output. Do not increase scope until green.

### 6b) Run ESC-50 official only (minimal reproducer: `--limit 1`)

For ESC-50 comparisons, run the official eval task:
`beans_zero_esc50_official` (dataset instruction prompt + top1 metric).

```bash
RUN_ID="hybrid_esc50_official_limit1_${SERVE_JOB_ID}"
OUT_DIR="${BEANS_PRO_RESULTS_LOCAL}/${RUN_ID}"

uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --task-id beans_zero_esc50_official \
  --dataset-name esc50 \
  --limit 1 \
  --run-id "$RUN_ID" \
  -o "$OUT_DIR"

echo "Artifacts in: $OUT_DIR"
ls -la "$OUT_DIR"
```

### 7) Validate artifacts locally (no launcher required)

```bash
bash scripts/validate_run_dir.sh "$OUT_DIR"

# Optional offline rescore:
uv run beans-next score-from-file "$OUT_DIR/predictions.jsonl" \
  -o "${OUT_DIR}__rescored"
bash scripts/validate_run_dir.sh "${OUT_DIR}__rescored"
```

---

## Troubleshooting (quick checks)

When a run fails, do not scale up; diagnose with `--limit 1` first.

### URL file exists but inference never starts

- Ensure the serve job is actually **RUNNING** (`squeue --me`). Pending jobs do not write URL files.
- Confirm `PREDICT_URL` is reachable from this host:

```bash
curl -fsS "${PREDICT_URL%/predict}/health"
curl -fsS "${PREDICT_URL%/predict}/info"
```

### Server can’t read audio paths (hybrid gotcha)

If your prompt uses `payload_type="file_path"`, the server must be able to read those paths.
In hybrid mode, local host paths usually do not exist on the GPU node. Prefer base64 WAV payloads.

### “Paper comparable” ESC-50 protocol

For comparisons, run the official ESC-50 eval task (dataset instruction prompt + top1 metric):
`beans_zero_esc50_official`. Avoid candidate-label list prompts unless you explicitly want
closed-set multiple-choice.

### Audio preprocessing mismatches

Different models require different sample rates and clip lengths. Before implementing a new
launcher or adapter, consult upstream docs/code (and web docs) for:
- expected sample rate
- clip length / chunking strategy
- padding/truncation

### Weird import errors (“partially initialized module …”)

On the Slurm checkout, ensure there are no stray repo-root files like `types.py` that can shadow
stdlib modules and break unrelated imports.

### 8) Scale up (~100 samples, multiple splits)

Once limit-1 is green:

```bash
RUN_ID="hybrid_100s_${SERVE_JOB_ID}"
OUT_DIR="${BEANS_PRO_RESULTS_LOCAL}/${RUN_ID}"

uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --suite beans_zero_core \
  --limit 100 \
  --run-id "$RUN_ID" \
  -o "$OUT_DIR"

bash scripts/validate_run_dir.sh "$OUT_DIR"
```

---

## Two-job pattern: both serve and inference on cluster (full-scale / production)

Use this when inference throughput requires a cluster node (large suites, no local network path).

### 0) Choose shared paths

```bash
export BEANS_PRO_URL_DIR="$HOME/beans-next-launchers"
export BEANS_PRO_RESULTS_ROOT="$SCRATCH/beans-next-results"
```

### 1) Submit serve job

```bash
SERVE_JOB_ID=$(sbatch examples/slurm/serve_naturelm_v1_0.sh | awk '{print $NF}')
echo "SERVE_JOB_ID=$SERVE_JOB_ID"
```

### 2) Submit inference job (minimal reproducer: `BEANS_PRO_LIMIT=1`)

```bash
BEANS_PRO_URL_FILE="${BEANS_PRO_URL_DIR}/${SERVE_JOB_ID}.url" \
BEANS_PRO_SUITE=beans_zero_core \
BEANS_PRO_LIMIT=1 \
BEANS_PRO_OUT_DIR="${BEANS_PRO_RESULTS_ROOT}/min_repro_limit1_${SERVE_JOB_ID}" \
BEANS_PRO_RUN_ID="slurm_min_repro_${SERVE_JOB_ID}" \
BEANS_PRO_COPY_RESULTS_TO_HOME=1 \
sbatch --dependency=after:${SERVE_JOB_ID} examples/slurm/run_inference.sh
```

### 3) Copy back + offline inspection

If you set `BEANS_PRO_COPY_RESULTS_TO_HOME=1`, the inference job will copy results into:

- Slurm view: `$HOME/beans-next-results/ingested/$BEANS_PRO_RUN_ID/`
- Local host view: `/mnt/home/marius_miron_earthspecies_org/beans-next-results/ingested/$BEANS_PRO_RUN_ID/`

…so you can skip manual rsync and go straight to validation below.

```bash
RUN_ID="slurm_min_repro_${SERVE_JOB_ID}"
mkdir -p "results/ingested/${RUN_ID}"
rsync -av --info=progress2 \
  "marius_miron_earthspecies_org@slurm:${BEANS_PRO_RESULTS_ROOT}/min_repro_limit1_${SERVE_JOB_ID}/" \
  "results/ingested/${RUN_ID}/"

bash scripts/validate_run_dir.sh "results/ingested/${RUN_ID}"
uv run beans-next score-from-file "results/ingested/${RUN_ID}/predictions.jsonl" \
  -o "results/ingested/${RUN_ID}__rescored"
```

See `docs/results_ingestion.md` for the deeper checklist.

---

## Slightly-less-minimal templates (hybrid pattern)

### Run a larger cap

```bash
uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --suite beans_zero_core \
  --limit 50 \
  --run-id "hybrid_limit50_${SERVE_JOB_ID}" \
  -o "${BEANS_PRO_RESULTS_LOCAL}/hybrid_limit50_${SERVE_JOB_ID}"
```

### Run a config YAML instead of a suite

```bash
uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --config configs/paper/beans_zero_naturelm_side_by_side.yaml \
  --limit 1 \
  --run-id "hybrid_config_${SERVE_JOB_ID}" \
  -o "${BEANS_PRO_RESULTS_LOCAL}/hybrid_config_${SERVE_JOB_ID}"
```
