# Results ingestion + offline rescoring (Increment 7, CPU-only)

This guide is for **Increment 7** workflows where a **GPU cluster** runs `beans-next run`, then you **copy the run directory back** to this repo/host and **rescore offline** via `beans-next score-from-file` (no launcher required).

## End-to-end CPU-only loop (once a cluster run directory exists)

At this point the cluster has already produced a run directory (often shaped like `results/<model>/<run_id>/`).
Everything below is CPU-only and runs locally in this repo.

1. Copy the run directory into this repo (recommended: copy the whole directory).
2. Validate the copy is usable for offline rescoring:

```bash
bash scripts/validate_run_dir.sh results/ingested/<run_id>
```

If you copied a whole bundle directory containing nested per-model/per-run leaf directories, you can validate the bundle root too (the validator will discover leaf run dirs like `<model>/<run_id>/` and fail on the first invalid one):

```bash
bash scripts/validate_run_dir.sh results/ingested/<bundle_root>
```

3. Rescore offline into a fresh output directory (recommended; preserves the copied artifacts):

```bash
uv run beans-next score-from-file \
  results/ingested/<run_id>/predictions.jsonl \
  -o results/ingested/<run_id>__rescored
```

4. Inspect outputs (at minimum `summary.json`, plus `scored_predictions.jsonl` for per-sample details).

Optional: validate the rescored output directory too:

```bash
bash scripts/validate_run_dir.sh results/ingested/<run_id>__rescored
```

### One-command helper (recommended)

If you have already copied a leaf run directory back (one directory containing `predictions.jsonl`), you can run a single deterministic helper that validates + rescoring + validates output:

```bash
./scripts/ingest_and_rescore.sh results/ingested/<run_id>
```

This writes rescored artifacts to `results/ingested/<run_id>__rescored/`.

## What to copy from the cluster

Copy the entire run output directory (for example `results/<run_id>/`) into this repo under:

`results/ingested/<run_id>/`

If you ran via Slurm using `examples/slurm/run_inference.sh`, the output directory is whatever you set as `BEANS_PRO_OUT_DIR` (or its default from that script).

### Common directory shapes (all supported by the validator)

- **Leaf run dir (single run)**:
  - `results/ingested/<run_id>/predictions.jsonl`
- **Per-model leaf run dir** (common BEANS-Next layout):
  - `results/ingested/<model>/<run_id>/predictions.jsonl`
- **Copied bundle root** (many runs/models):
  - `results/ingested/<bundle_root>/<model>/<run_id>/predictions.jsonl`

You can point `bash scripts/validate_run_dir.sh` at either a leaf run dir *or* a bundle root; bundle roots will validate each discovered leaf run dir and fail fast on the first invalid one.

## Minimum artifact expectations

### For archiving (inspectable later)

- **`predictions.jsonl`** (required): raw launcher responses (one JSON object per line). Must include at least one `sample_id` and one `predictions` field somewhere in the file (i.e. `ModelPrediction`-shaped rows).
- **`model_identity.json`** (optional): model identity snapshot

### For offline rescoring (required by `score-from-file`)

- **`predictions.jsonl`** (required)
- **`processed_predictions.jsonl`** (required): must contain `sample_id` + `targets` for each sample (and ideally `task_id` too)

`beans-next score-from-file` **does not** re-load datasets, so it cannot recover targets from HuggingFace. It reads targets from the sibling `processed_predictions.jsonl` and joins them by `sample_id`.
This sibling file must be located **next to** the input `predictions.jsonl` (same directory), even if you write outputs somewhere else via `-o/--output-dir`.

Notes:

- `processed_predictions.jsonl` is treated as a **targets sidecar**, not as an authoritative previous scoring output. `score-from-file` will recompute post-processing and metrics and write new artifacts.
- The sidecar must contain, for at least some rows, **both** `sample_id` and non-null `targets` (and ideally `task_id`). This file is produced by `beans-next run`. If some lines are malformed, offline rescoring will skip those lines; if that results in no usable targets, `score-from-file` fails.
- If your copied run directory only contains `predictions.jsonl`, you can still archive it, but you **cannot** rescore it offline without also copying `processed_predictions.jsonl` from the original run.
- If you rescore into the same directory, `score-from-file` will overwrite `processed_predictions.jsonl`, `scored_predictions.jsonl`, and `summary.json` in that directory. If you want to preserve the original copied artifacts unchanged, rescore into a new directory via `-o/--output-dir`.
- If you rescore into a **different** output directory via `-o/--output-dir`, that output directory typically **does not** include `predictions.jsonl` (it’s an output/inspection directory, not a complete re-runnable input bundle). The validator supports this and will switch to “output directory” checks automatically.

### Recommended: copy the entire run directory

Copying the whole directory avoids “looks present but missing sidecar files” issues.

Example with `rsync` (replace paths):

```bash
# On your local machine (this repo)
mkdir -p results/ingested/

# Copy one run directory
rsync -avP --delete \
  $USER@slurm:/home/$USER/code/beans-next/results/<run_id>/ \
  results/ingested/<run_id>/
```

If your cluster writes runs under `$SCRATCH` and you want to copy many runs:

```bash
rsync -avP \
  user@cluster:$SCRATCH/beans-next/results/ \
  results/ingested/
```

## Validate the copy is complete (fast checklist)

Run this in the **copied** run directory:

```bash
cd results/ingested/<run_id>

# minimum for offline rescoring
test -s predictions.jsonl
test -s processed_predictions.jsonl
```

If `processed_predictions.jsonl` is missing or lacks `targets`, you can still archive the run, but `score-from-file` will fail.

Recommended: run the repo’s validator for a slightly more actionable check. It **fast-fails** on the first missing required artifact/field. Optional files are reported as **INFO**, and optional “nice to have” fields are reported as **WARN**:

```bash
bash scripts/validate_run_dir.sh results/ingested/<run_id>
```

Note: the validator uses **cheap string checks** (it does not fully parse JSONL). If it passes, the run is very likely usable for offline rescoring, but schema-corrupted JSONL can still fail later when `beans-next score-from-file` parses the files.

## Offline rescoring + regenerating summaries (no launcher)

`beans-next score-from-file` loads `predictions.jsonl`, recomputes post-processing + per-sample metrics on CPU, and writes:

- `processed_predictions.jsonl`
- `scored_predictions.jsonl`
- `summary.json`
- `model_identity.json`

### CLI shape (source of truth: `--help`)

```
uv run beans-next score-from-file [-h] [-o OUTPUT_DIR] PREDICTIONS_JSONL
```

Notes:

- `PREDICTIONS_JSONL` is the path to the `predictions.jsonl` you copied (typically produced by `beans-next run` on the cluster).
- `-o/--output-dir` defaults to the directory containing `PREDICTIONS_JSONL`.
- `processed_predictions.jsonl` is **read from** the input predictions file directory (sibling) to obtain `targets` (and `task_id` when present).
- Fresh `processed_predictions.jsonl` and `scored_predictions.jsonl` are **written to** `-o/--output-dir`.
- `model_identity.json` is derived from the first prediction row that includes `server_info`; if no prediction row contains `server_info`, it will be `{}`.

### What `score-from-file` does (and does not) do

- **Does**: post-process raw `predictions` text and compute metrics using targets found in the sibling `processed_predictions.jsonl`.
- **Does**: uses a small **default post-process pipeline** intended to be “good enough for inspection” (it derives a label vocabulary from available `targets` to improve fuzzy matching when appropriate).
- **Does not**: contact any launcher, re-load datasets from HuggingFace, or attempt to reconstruct the exact original run configuration (prompt choice, post-process steps, scorer configuration). Treat offline rescoring as a **convenience for inspection and iteration**, not as a bit-exact reproduction of the original run unless your run used the same default post-processing and metrics.

## Optional: archive layout suggestion

If you will ingest many runs, keep a stable naming scheme so it’s easy to trace provenance:

- `results/ingested/<cluster>_<serve_job_id>_<run_id>/` (raw copied)
- `results/ingested/<cluster>_<serve_job_id>_<run_id>__rescored/` (offline rescoring outputs)

### Rescore in-place (writes into the same directory)

```bash
uv run beans-next score-from-file results/ingested/<run_id>/predictions.jsonl
```

### Rescore into a fresh output directory (recommended)

This preserves the original copied artifacts unchanged.

```bash
uv run beans-next score-from-file \
  results/ingested/<run_id>/predictions.jsonl \
  -o results/ingested/<run_id>__rescored
```

### Common failure modes

- **Missing targets**:
  - Error excerpt: "Cannot score metrics without targets. Provide a sibling `processed_predictions.jsonl` containing `targets` for each sample (typically produced by `beans-next run`)."
  - Root cause: the sibling `processed_predictions.jsonl` is missing, empty, or does not contain any valid `ScoredPrediction` rows with `targets`.
  - Fix: copy `processed_predictions.jsonl` from the original run directory (same parent as `predictions.jsonl`). If it exists but is corrupted, recopy the run directory.
- **Empty predictions**:
  - Error: `No JSONL rows found in .../predictions.jsonl`
  - Fix: treat as a failed/partial run; re-run inference on the cluster or locate the correct run directory.
- **Wrong path**:
  - Error: `Not found: .../predictions.jsonl`
  - Fix: pass the path to the file, not just the directory, and confirm the copy succeeded.
- **Corrupted JSONL / wrong schema**:
  - Symptom: `score-from-file` exits early with a schema validation error, or succeeds but yields an unexpected `n_errors` count.
  - Root cause: `predictions.jsonl` rows are not `ModelPrediction`-shaped JSON objects, or `processed_predictions.jsonl` is not `ScoredPrediction`-shaped.
  - Fix: recopy the run directory; if you’re producing artifacts from a custom harness, ensure you are writing BEANS-Next’s standard artifact formats.

## Quick rubric

- **Archivable**: has at least `predictions.jsonl` and `model_identity.json` (or server info embedded in prediction rows).
- **Offline-rescorable**: has `predictions.jsonl` plus a sibling `processed_predictions.jsonl` with `targets`.
- **Fully comparable**: also has `scored_predictions.jsonl` and `summary.json` (or can be regenerated via `score-from-file`).

