# SLURM Usage Guide

## Overview

Slurm provides:

* **Resource abstraction** (you don’t manage nodes directly)
* **Resource isolation** (jobs only use allocated resources)
* **Scheduling** (queues + allocation)
* **Usage tracking**

### Job Flow

1. Submit job with resource requirements
2. Job enters queue
3. Slurm allocates resources
4. Script runs on assigned node from submission directory

---

## Cluster Basics

* `/home` → persistent, shared across cluster (backed up)
* `/scratch/$USER`, `/tmp/$USER` → **ephemeral local disk (~250GB)**
* Data → use cloud buckets (GCS / R2)

Note: on this particular local host, the cluster `/home` share is mounted at **`/mnt/home`**
(NFS). On the Slurm login/compute nodes it is mounted at `/home`.

⚠️ Never store important data on compute nodes.

---

## Access

All interaction happens via the **login node (`slurm-login`)**.

### Connect

```bash
ssh slurm
```

---

## Agent quickstart (connect → check resources → launch jobs)

### 0. Connect

```bash
ssh slurm
```

### 0.5 Sync repo into the NFS mount (REQUIRED on this host)

On this machine, the Slurm-shared `/home` filesystem is mounted at `/mnt/home` (see note in
“Cluster Basics”). The repo workspace you edit in this IDE is under local `/home/...` (ext4), so
Slurm will not see your latest changes unless you sync them to the NFS path.

Run this **before each `sbatch`**:

```bash
rsync -av --delete --exclude '.venv/' \
  /home/marius_miron_earthspecies_org/code/beans-next/ \
  /mnt/home/marius_miron_earthspecies_org/code/beans-next/
```

### 1. Check free resources / queue state

Fast overview:

```bash
sinfo
squeue --me
```

Useful “where are the free GPUs?” views (format varies by cluster config):

```bash
# Show nodes/partitions with GPU + state (idle/mix/alloc)
sinfo --Node --long

# Compact partition summary (nodes by state)
sinfo --summarize

# Jobs in a specific partition (example partition names in this repo: a100-40, h100-80)
squeue --partition=a100-40
squeue --partition=h100-80
```

#### Picking the “most free” GPU option (exclude `t4`)

For BEANS-Next GPU work, prefer **`a100-40` first**, then **`h100-80`**, and **exclude `t4`**
unless explicitly requested for debugging.

1) **Check partition pressure (who’s queued where):**

```bash
squeue --partition=a100-40
squeue --partition=h100-80
```

2) **Check per-node free GPUs (allocated vs total):**

Slurm reports GPU totals in `CfgTRES` and allocations in `AllocTRES`. “Free GPUs” is:
\( \text{free} = \text{CfgTRES gres/gpu} - \text{AllocTRES gres/gpu} \)

```bash
# Inspect one node (shows totals + allocations)
scontrol show node slurm-8x-h100-2 | tr " " "\n" | egrep "^(NodeName=|State=|Partitions=|Gres=|CfgTRES=|AllocTRES=)"
```

If you want a quick cluster snapshot across the non-T4 GPU nodes, run:

```bash
for n in $(sinfo -N -h -o "%N %P %G" | egrep "a100-40|h100-80" | awk "{print $1}"); do
  echo "--- $n ---"
  scontrol show node "$n" | tr " " "\n" | egrep "^(NodeName=|State=|Partitions=|Gres=|CfgTRES=|AllocTRES=)" || true
done
```

3) **Decide which partition is “more likely to be free soon”:**

- Prefer a partition with **more free GPUs right now** (higher free count on at least one node).
- If both have free GPUs, prefer the partition with **fewer queued jobs** (smaller `squeue` output),
  because it tends to start faster.
- If a node is `DRAIN` / `DRAINED`, ignore it.

If you have a job id and want to understand why it’s pending:

```bash
scontrol show job <job_id>
```

### 2. Launch a job (preferred: `sbatch`)

Run one of this repo’s Slurm scripts from the repo root:

```bash
cd /home/marius_miron_earthspecies_org/code/beans-next
sbatch examples/slurm/serve_naturelm_v1_0.sh
```

Notes for agents:

* After `sbatch`, the job typically enters **PD** (Pending). The server is **not** running yet.
* Wait until the job is **R** (Running) before expecting a URL file, logs, or a working `/health`.

---

## Field notes / common failure modes (BEANS-Next model bring-up)

These are high-frequency issues encountered when moving from **stub** → **real inference** on
Slurm (NatureLM v1.0, and expected for other models). Keep this list in mind when adding
new serve scripts (AF3, vLLM/Qwen, NatureLM v1.1, API-backed models).

### Tokens / gated weights

- **HuggingFace gated access**: many model servers require `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`)
  for gated repos (e.g. Llama base models, private ESP model repos).
- **GitHub rate limiting / auth**: if a serve script installs a model package via
  `git+https://github.com/...`, you may need `GITHUB_TOKEN` or a credential helper on the node.

### Local `file_path` vs remote server

- If `payload_type="file_path"` is used, the **server must be able to read that path**.
  In hybrid setups (client on host, server on Slurm), local paths usually do not exist on the GPU
  node. Prefer `payload_type="base64_wav"` (runner can convert automatically in BEANS-Next).

### `esp_data` backend (cluster-first)

- Symptom: `ModuleNotFoundError: esp_data` or failures loading BEANS-Zero without HF access.
- Fix pattern: use `BEANS_PRO_DATA_SOURCE=esp_data` and ensure jobs run `uv sync --group esp`
  (requires `pyproject.toml` `tool.uv.index` + `tool.uv.sources` for `esp-pypi`).

### “Official” ESC-50 vs closed-set prompt

- **Always use the official ESC-50 eval-task** (`beans_zero_esc50_official`) when comparing to the
  BEANS-Zero / NatureLM paper.
- Avoid prompts that include the full candidate label list unless you explicitly want a
  **closed-set multiple-choice** variant (it will inflate scores).

### Audio preprocessing must be explicit and per-model

- Many models expect a specific **sample rate** (NatureLM: 16 kHz) and a specific **clip length**
  (ESC-50 official: 5 seconds).
- When implementing a launcher, consult upstream docs/code (and/or web docs) to match:
  resampling, padding/truncation, mono/stereo handling, normalization, and chunking.

### Repo sync + “shadowing” bugs

- If you rsync files into the Slurm-visible repo, **never** leave stray Python files at repo root
  (e.g. `types.py`) — they can shadow stdlib modules and break imports in surprising ways.
  If you see errors like “partially initialized module …”, check for shadowing.

### Scratch disk full (common on GPU nodes)

Symptom:

- Serve job fails during `uv sync`, wheel downloads, venv creation, or Hugging Face model download
- Errors like “No space left on device”, `/scratch` at ~100% usage, or launcher never reaches `/health`

Facts about this cluster:

- `/scratch/$USER` is **node-local ephemeral disk (~250GB)**.
- Nodes can be full **for reasons unrelated to your job** (other users, system state, etc.).

#### Cleanup: mistakenly-created literal `/scratch/$USER`

If a runbook passes a single-quoted path like `'/scratch/$USER'` to a tool that creates cache
directories, some nodes may end up with a *literal* directory named `/scratch/$USER` (with a dollar
sign in the directory name) rather than `/scratch/<your_username>`.

This repo includes a safe cleanup helper that deletes **only** that literal directory:

```bash
# Run on one node (interactive):
bash examples/slurm/cleanup_literal_scratch_user.sh

# Run across every node you allocate (recommended):
sbatch --partition=a100-40 --nodes=8 examples/slurm/cleanup_literal_scratch_user_job.sh
sbatch --partition=h100-80 --nodes=8 examples/slurm/cleanup_literal_scratch_user_job.sh
```

Note: Slurm can only run on nodes you allocate. To cover “all nodes”, run once per partition with a
node count large enough to span that partition, or have an admin run the job with a full node list
/ reservation.

Empirical cache sizes observed (April 2026, Qwen3-Omni on H100 nodes):

- `Qwen/Qwen3-Omni-30B-A3B-Thinking` HF hub cache: ~**60 GB**
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` HF hub cache: ~**66 GB**
- `Qwen/Qwen3-Omni-30B-A3B-Captioner` HF hub cache: ~**60 GB**
- Job-scoped serve venvs under `/scratch/$USER/venvs/*`: often **~5–6 GB** each
- `uv_managed_python` under `/scratch/$USER/uv_managed_python`: **~0.1–0.2 GB**
- `~/.cache/uv` is usually on shared `/home` (NFS) and was ~**3.1 GB** for this user; it is not the
  dominant consumer of node-local `/scratch`.

Operational policy:

- **Fail fast** if `/scratch` free space is below the model’s required headroom.
- **Try safe cleanup** first (old job venvs, stale HF locks, optionally old per-user HF caches).
- If caches are shared under `/scratch/.cache`, cleanup may **empty** known cache directories
  (e.g. `/scratch/.cache/huggingface`, `/scratch/.cache/uv`) but must **never delete** `.cache`
  itself nor delete those cache directories (tools expect the directories to exist).
- If still insufficient, exit with a clear error marker so the manager/agent can resubmit elsewhere.

Serve scripts in this repo implement this via `examples/slurm/scratch_guard.sh` and exit with
`BEANS_PRO_DISK_SPACE_BLOCKER` (exit code 86) when they cannot recover enough disk.

Default scratch headroom thresholds (override per job with `BEANS_PRO_SCRATCH_MIN_FREE_GB`):

- Qwen/vLLM: **100 GB**
- NatureLM: **50 GB**
- AF3: **45 GB**
- Fallback: **40 GB**

Cache-aware optimization:

- Serve scripts call `beans_next_scratch_guard` which can **lower** the required free space when the
  model is already present in the HF hub cache (heuristic: model hub dir exists and is ≥10 GB).
- Cached-mode defaults (also overrideable):
  - Qwen/vLLM cached: **10 GB**
  - NatureLM cached: **10 GB**
  - AF3 cached: **5 GB**
  - Fallback cached: **5 GB**

### Failure recovery ladder (quick reference)

For the full recovery ladder with exact commands, see
`examples/slurm/AGENT_RUNBOOK.md`. Summary:

| Failure | Action |
|---|---|
| Exit 86 (scratch full, node 1) | `scontrol show job <id>` → get node name → `sbatch --exclude=<node> <script>` |
| Exit 86 (weights likely cached) | Add `BEANS_PRO_SCRATCH_MIN_FREE_GB=20` to the sbatch env |
| Exit 86 (two nodes failed) | Write BLOCKED with both node names + `free_gb` evidence |
| Job stuck PD > 10 min | Check reason with `scontrol show job`; try `--partition=h100-80` |
| URL file missing after job is R | `tail -f /mnt/home/.../logs/<job_id>.log` for startup errors |
| HF gated access denied | Stop; operator must grant HF access |

### 3. Monitor until running, then follow logs

```bash
# Poll until the job transitions to R (Running)
squeue --me

# Once running, inspect detailed job info (node, reason, paths)
scontrol show job <job_id>
```

### 4. Cancel if needed

```bash
scancel <job_id>
scancel --me
```

### One-time setup (summary)

* Generate SSH key
* Add key to GCP OS Login
* Configure `~/.ssh/config`
* (Optional) add key to GitHub

---

## Running Jobs

### 1. `srun` (interactive / debugging)

```bash
srun ls
srun --gpus=a100:1 train.sh
srun --gpus=t4:2 --mem=2G --cpus-per-gpu=3 train.py
```

Interactive shell:

```bash
srun --partition=debug --gpus=t4:1 --pty bash
```

Characteristics:

* Blocks terminal
* Streams output live
* Dies if terminal closes

---

### 2. `sbatch` (production)

```bash
sbatch job.sh
```

* Non-interactive
* Continues after disconnect
* Preferred method

---

## Example Job Script

### Single GPU (uv project)

```bash
#!/usr/bin/env bash
#SBATCH --gpus=t4:1
#SBATCH --output="/home/$USER/logs/%A.log"
#SBATCH --job-name="my_job"

export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/myenv

cd ~/project
uv sync

srun uv run train.py
```

---

### Non-uv project

```bash
#!/usr/bin/env bash
#SBATCH --gpus=t4:1
#SBATCH --output="/home/$USER/logs/%A.log"

uv venv --python 3.12 /scratch/$USER/venvs/foo
source /scratch/$USER/venvs/foo/bin/activate

cd ~/project
uv pip install .

srun ./run.sh
```

---

## Multi-GPU / Multi-node

### Multi-GPU (same node)

```bash
#SBATCH --gpus-per-node=a100:4
```

⚠️ `--gpus=a100:4` does NOT guarantee same node.

---

### Multi-node

```bash
#SBATCH --nodes=8
#SBATCH --gpus-per-node=a100:4
```

---

### Using `srun` inside `sbatch`

```bash
srun foo.sh &
srun bar.sh &
wait
```

Benefits:

* Better monitoring
* Step-level control

---

## Job Arrays

```bash
#SBATCH --array=1-100%5
#SBATCH --gpus=a100:1

./run input_${SLURM_ARRAY_TASK_ID}.json
```

* `%5` → max concurrent jobs
* Use for parameter sweeps / batch runs

---

## Monitoring

### Jobs

```bash
squeue
squeue --me
squeue --state=RUNNING
```

Detailed:

```bash
scontrol show job <job_id>
```

---

### Nodes / partitions

```bash
sinfo
sinfo --long
```

---

### Logs

Default:

```
slurm-{jobid}.out
```

Custom:

```bash
#SBATCH --output="/home/$USER/logs/%j.log"
```

Follow logs:

```bash
tail -f file.log
```

---

### Attach to job

```bash
sattach <job_id>.0
```

⚠️ Requires `srun` inside script.

---

## Notifications

```bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=you@example.com
```

(Delivered via Slack in this cluster)

---

## Cancelling Jobs

```bash
scancel <job_id>
scancel --me
scancel --state=PENDING
```

---

## Best Practices / Etiquette

* Do NOT fill `/home` (shared storage)
* Use buckets for large data
* Do NOT run heavy jobs on login node
* Use `debug` partition for interactive work only
* Prefer job arrays over many small jobs
* Never SSH directly into compute nodes
* Avoid VSCode Remote SSH on login node
* Treat compute node storage as ephemeral

---

## Tips

### Mount `/home` locally (from another VM)

```bash
sudo apt install -y nfs-common
mkdir -p /mnt/home

echo 'esp-nfs-server:/home/YOUR_USER /mnt/home nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab

sudo mount -a
```

---

### Debugging with breakpoints

```bash
pip install web-pdb
export PYTHONBREAKPOINT="web_pdb.set_trace(-1)"
```

Then access:

```
http://<node-ip>:5555
```

---

## Key Takeaways

* Use `sbatch` for real workloads
* Use `/scratch` for performance-sensitive temp data
* Monitor with `squeue`, logs, and `scontrol`
* Scale with job arrays and multi-node configs

---
