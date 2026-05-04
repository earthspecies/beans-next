# Agent Runbook — Hybrid GPU Evaluation

Single authoritative reference for executing a hybrid evaluation increment (GPU server on
Slurm, inference runs locally on this host). **Read this doc + the increment-specific
section in INCREMENTS.md.** That is sufficient for execution. Do not spend context budget
reading AGENT_SPEC.md, DESIGN.md, or CLUSTER_MIN_REPRO.md unless debugging an architecture
question — the execution rules are embedded here.

---

## Required reading (2 docs, not 5)

1. **This file** — execution steps, recovery ladder, finish log format.
2. **INCREMENTS.md §I-X** — task ids, subsets, ports, output dirs, and any increment-specific constraints.

---

## Pre-flight (always, before any `sbatch`)

```bash
# Sync local workspace → NFS. Required even for small edits.
rsync -av --delete \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.ruff_cache/' --exclude '.pytest_cache/' \
  "$HOME/code/beans-next/" \
  "/mnt/home/marius_miron_earthspecies_org/code/beans-next/"
```

---

## Execution checklist

### Step 1 — Check GPU availability

```bash
ssh slurm "sinfo --summarize"
ssh slurm "squeue --partition=a100-40"
ssh slurm "squeue --partition=h100-80"
```

Prefer `a100-40` → `h100-80`. Avoid `t4` unless explicitly requested.

### Step 2 — Submit serve job

```bash
ssh slurm "cd /home/marius_miron_earthspecies_org/code/beans-next && \
  BEANS_PRO_PORT=<PORT> BEANS_PRO_DEBUG=1 \
  sbatch examples/slurm/serve_<model>.sh"
# → record <job_id>
```

### Step 3 — Poll until RUNNING (mandatory; PD means not started)

```bash
# Repeat every ~30s until state shows R
ssh slurm "squeue --me --noheader -j <job_id>"
```

Do not proceed to step 4 until the job state is `R`. Long waits in `PD` are normal.

### Step 4 — Read URL file and verify server

```bash
URL_FILE=/mnt/home/marius_miron_earthspecies_org/beans-next-launchers/<job_id>.url
# Wait until file appears (typically 2-5 min after R)
cat "$URL_FILE"

BASE_URL=$(cat "$URL_FILE" | sed 's|/predict$||')
curl -sf "$BASE_URL/health" | python3 -m json.tool
curl -sf "$BASE_URL/info"  | python3 -m json.tool
```

### Step 5 — Smoke run (--limit 5) for EACH task before any full run

```bash
PREDICT_URL=$(cat /mnt/home/marius_miron_earthspecies_org/beans-next-launchers/<job_id>.url)

uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --task-id <task_id> \
  --data-source esp_data \
  --limit 5 \
  --output-dir results/ingested/<increment>/<model>/<task_id>
```

### Step 6 — Validate smoke artifacts

```bash
bash scripts/validate_run_dir.sh results/ingested/<increment>/<model>/<task_id>
```

Only proceed to full run after both smoke run and validation pass cleanly.

### Step 7 — Full run (no --limit), repeat per task

```bash
uv run beans-next run \
  --predict-url "$PREDICT_URL" \
  --task-id <task_id> \
  --data-source esp_data \
  --output-dir results/ingested/<increment>/<model>/<task_id>

bash scripts/validate_run_dir.sh results/ingested/<increment>/<model>/<task_id>
```

### Step 8 — Shut down serve job

```bash
ssh slurm "scancel <job_id>"
```

---

## Failure recovery ladder

Recoverable failures **must be attempted** before writing a BLOCKED finish log.
Unrecoverable failures may be reported immediately.

---

### Exit 86 — scratch disk full (`BEANS_PRO_DISK_SPACE_BLOCKER`)

**RECOVERABLE. Must retry on a different node before writing BLOCKED.**

```bash
# 1. Identify the bad node
ssh slurm "scontrol show job <job_id> | tr ' ' '\n' | grep '^BatchHost='"
# → e.g. BatchHost=slurm-8x-a100-1

# 2. Resubmit excluding that node
ssh slurm "cd /home/marius_miron_earthspecies_org/code/beans-next && \
  BEANS_PRO_PORT=<PORT> BEANS_PRO_DEBUG=1 \
  sbatch --exclude=slurm-8x-a100-1 examples/slurm/serve_<model>.sh"
```

If this is a repeated run and the model weights are likely already cached on the cluster
(i.e. this model ran successfully in a prior increment), also try lowering the required
headroom. The scratch guard lowers it automatically when it detects cached weights, but
you can force it:

```bash
ssh slurm "cd /home/marius_miron_earthspecies_org/code/beans-next && \
  BEANS_PRO_PORT=<PORT> BEANS_PRO_DEBUG=1 \
  BEANS_PRO_SCRATCH_MIN_FREE_GB=20 \
  sbatch --exclude=slurm-8x-a100-1 examples/slurm/serve_<model>.sh"
```

**Write BLOCKED only after two distinct node attempts both fail with exit 86.** Include
both node names and their `free_gb` / `required_gb` lines from the serve logs.

---

### Job stuck in PD for > 10 minutes

**RECOVERABLE.**

```bash
# Check why pending
ssh slurm "scontrol show job <job_id>" | grep -E "Reason|NodeList|Partition"

# If partition is overloaded, cancel and resubmit to the other GPU partition
ssh slurm "scancel <job_id>"
ssh slurm "cd /home/marius_miron_earthspecies_org/code/beans-next && \
  BEANS_PRO_PORT=<PORT> BEANS_PRO_DEBUG=1 \
  sbatch --partition=h100-80 examples/slurm/serve_<model>.sh"
```

---

### HF gated access denied (`GatedRepoError` / HTTP 403/401)

**UNRECOVERABLE without operator action.** Stop and report immediately:
- exact error from serve log
- model repo name
- whether `HF_TOKEN` is set in the serve script env

---

### URL file not written after job reaches R

Check the live log for errors:

```bash
tail -f /mnt/home/marius_miron_earthspecies_org/logs/<job_id>.log
```

Common causes:
- CUDA not available (`torch.cuda.is_available()` = False) → try different node/partition
- Missing imports (`naturelm`, `esp_data`) → check serve script `uv sync` group
- Scratch space exhausted mid-startup (different from exit 86 — job did not exit) →
  check log for "No space left on device" after the guard passed

---

### `/health` returns error or model load failure

Read the live serve log. If CUDA unavailable, try a different partition. If model fails to
load (weights corrupt or incompatible), report BLOCKED with the exact exception from the log.

---

### `esp_data` backend stalls during inference

**RECOVERABLE.**

```bash
# Kill the inference run (Ctrl+C or kill the process)
# Retry with HF backend as documented fallback
uv run beans-next run ... --data-source hf --limit 5
```

If HF backend works but esp_data does not, record the failure in the finish log and continue
with HF backend for this increment.

---

### Prompt version mismatch (`prompt_version=classification_bioacoustic_v1`)

**RECOVERABLE.** This means the wrong prompt YAML was selected. Read the eval-task YAML for
the task id to find the correct `prompt_yaml` key. Check `summary.json` after the smoke run:
`prompt_version` must match the value specified in the increment's INCREMENTS.md entry.

---

## Finish log — mandatory in ALL cases

**A finish log is required even if the task ran for 30 seconds and hit a blocker.**

A missing finish log is worse than a BLOCKED finish log — it forces the next manager to
re-dispatch the entire increment from scratch without any evidence trail.

Minimum required content:

```markdown
## <TASK-ID> — <Model> × <Subsets> — FINISH

- **Status**: PASS | BLOCKED | PARTIAL
- **Date**: YYYY-MM-DD
- **Serve job**: <job_id> (node: <node_name>, partition: <partition>)

### Commands run (in order)

1. rsync → OK
2. sbatch → job <job_id>
3. poll squeue → state R after ~Xmin
4. /health → <status>
5. smoke --limit 5 → <outcome>
6. validate_run_dir → <outcome>
7. full run → <outcome or BLOCKED reason>
8. scancel → OK | skipped (job already ended)

### Artifacts written

- results/ingested/<increment>/<model>/<task_id>/summary.json (n_samples=X, top1_accuracy=Y)
- ... or "none" if blocked before any run completed

### If BLOCKED

- Recovery steps attempted: (list each attempt with node name + outcome)
- Evidence: (paste exact error lines, 5-20 lines, no secrets)
- Next step for operator: (single concrete action)
```

---

## Recoverable vs unrecoverable at a glance

| Failure | Recoverable? | First action |
|---|---|---|
| Exit 86 scratch full (first node) | Yes | `sbatch --exclude=<node>` |
| Exit 86 scratch full (second node) | Borderline | Try `BEANS_PRO_SCRATCH_MIN_FREE_GB=20` |
| Exit 86 scratch full (third node) | No | Write BLOCKED |
| Job stuck PD > 10 min | Yes | Switch partition |
| HF gated access denied | No | Report; operator action needed |
| URL file missing (job is R) | Yes | Read log, wait or fix env |
| `/health` error / CUDA unavailable | Yes | Try different node/partition |
| Missing `esp_data` module | Yes | Try `--data-source hf` |
| Prompt version wrong | Yes | Fix eval-task YAML reference |
| Model weights corrupted | No | Report; operator action needed |

---

## What must NOT appear in a finish log

- "PASS" when smoke run was not validated with `validate_run_dir.sh`
- Omitted `prompt_version` check (must match increment spec)
- "BLOCKED" without documenting which recovery steps were tried
- No finish log at all
