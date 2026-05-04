## Paper workflow (iteration 1): NatureLM v1.0 vs v1.1 side-by-side

This page documents the **iteration-1** “paper workflow” scaffold: running BEANS-Next against NatureLM-audio **v1.0** and **v1.1** endpoints (locally), producing side-by-side artifacts under `results/`.

Scope matches **`AGENT_SPEC.md` §3** (CPU end-to-end validation, launcher integration, correct artifacts on small samples) and **`IMPLEMENTATION_PLAN.md` §1.9 / Phase 2** (side-by-side config + documented workflow without requiring GPU for core validation). Architecture boundaries are the same as everywhere else in the repo: **`DESIGN.md` §1.6** (HTTP-only core, `predictions_v1`, mandatory `/predict` + `/info` + `/health`, **`HttpClient` as the shipped inference adapter**). Full GPU-backed NatureLM runs and published BEANS-Zero reproduction are **aspirational / post–iteration-1** per **`DESIGN.md` §1.6.1**—this doc does not assert those are done.

Iteration 1 is about **pipeline + HTTP contract correctness on small samples**. It is **not** a claim that full BEANS-Zero paper numbers are reproduced.

### What the side-by-side config is (and what it is not)

The file [`configs/paper/beans_zero_naturelm_side_by_side.yaml`](../configs/paper/beans_zero_naturelm_side_by_side.yaml) is:

- **A paper-workflow scaffold** describing:
  - the intended suite (`beans_zero_core`)
  - two local launcher endpoints (v1.0 on port 8000, v1.1 on port 8001)
  - which launcher workdirs to start for each model
- **A stable reference** for scripts and docs (ports, names, required env vars like `HF_TOKEN`)

It is **not** (yet):

- **A runnable `beans-next run --config ...` input**.

In iteration 1, `beans-next run` explicitly rejects `--config`. The runnable entrypoints are:

- `scripts/reproduce.sh` (one launcher + one task)
- `scripts/reproduce_all.sh` (start multiple launchers from the YAML, then run a suite)

### Preconditions

- You are at repo root.
- You have `uv` installed.
- You can run small-sample evaluation (default `LIMIT=5`).

Model-specific notes:

- **NatureLM v1.0**: iteration 1 launcher defaults to **stub mode** (`NATURELM_V1_0_STUB=1`) for HTTP contract testing.
- **NatureLM v1.1**: weights are gated; real mode requires `HF_TOKEN` with access to `EarthSpeciesProject/naturelm-audio-1.1.00-private`. Stub mode is available via `NATURELM_STUB_MODE=1`.

### Degraded mode (what it means)

“Degraded mode” means **the workflow continues with whichever model is healthy**:

- `scripts/reproduce_all.sh` will **skip a model** if:
  - its `/health` never returns 200, or
  - it is `naturelm-v1.1` and `HF_TOKEN` is missing (and `NATURELM_STUB_MODE!=1`)
- the script still runs the suite for any **healthy** launcher(s)

This is intentional for iteration 1: v1.1 is often unavailable (gated weights), and the side-by-side scaffold should still be usable as a single-model run.

### Stub mode (contract-only testing without GPU/weights)

Stub modes exist to validate the **HTTP contract** (`/health`, `/info`, `/predict` with `predictions_v1`, batching semantics) without requiring GPUs or downloading weights.

- **NatureLM v1.0**: stub mode is the default.
  - Override explicitly (optional): `export NATURELM_V1_0_STUB=1`
- **NatureLM v1.1**: enable stub mode explicitly:
  - `export NATURELM_STUB_MODE=1`

In stub mode, the launchers return deterministic placeholder predictions; the benchmark run is meant to validate end-to-end plumbing, not model quality.

### Run one model on a small sample (`scripts/reproduce.sh`)

`scripts/reproduce.sh` is the “one launcher + one eval task” workflow:

- prepares the launcher environment
- starts the launcher
- waits for `GET /health`
- runs launcher conformance (`scripts/check_launcher.sh`)
- runs `beans-next run` for the selected task id (or a shorthand)
- writes artifacts under `results/`

Run v1.0 (stub mode by default):

```bash
LIMIT=5 ./scripts/reproduce.sh naturelm-v1.0 esc50
```

Run v1.1 in stub mode (no token required):

```bash
export NATURELM_STUB_MODE=1
LIMIT=5 ./scripts/reproduce.sh naturelm-v1.1 unseen-species-tax
```

Run v1.1 in real mode (requires gated access token):

```bash
export HF_TOKEN=hf_...
LIMIT=5 ./scripts/reproduce.sh naturelm-v1.1 unseen-species-tax
```

Notes:

- **Task argument** accepts:
  - full eval task ids like `beans_zero_esc50`
  - shorthands like `esc50`, `unseen-species-tax` (mapped to bundled eval-task YAMLs)
- Output directory defaults to `results/reproduce-<launcher>-<task>/` unless you override:
  - `OUT_DIR=...` (artifact directory)
  - `RUN_ID=...` (run id recorded in `summary.json`)

### Run both models on a small sample (`scripts/reproduce_all.sh`)

Use `scripts/reproduce_all.sh` to start launchers from the side-by-side YAML and run a **suite**.

Small-sample run (best-effort; degraded mode enabled):

```bash
LIMIT=5 ./scripts/reproduce_all.sh configs/paper/beans_zero_naturelm_side_by_side.yaml
```

If you have v1.1 access and want to include it in non-stub mode:

```bash
export HF_TOKEN=hf_...
LIMIT=5 ./scripts/reproduce_all.sh configs/paper/beans_zero_naturelm_side_by_side.yaml
```

If you do not have v1.1 access but want to exercise the “two model” path anyway:

```bash
export NATURELM_STUB_MODE=1
LIMIT=5 ./scripts/reproduce_all.sh configs/paper/beans_zero_naturelm_side_by_side.yaml
```

### What artifacts to expect under `results/`

Both scripts ultimately run `beans-next run` which writes deterministic JSON/JSONL artifacts.

#### Single-task runs (`scripts/reproduce.sh`)

Default output directory:

- `results/reproduce-<launcher>-<task>/`

Files created directly under that directory:

- `predictions.jsonl`
- `processed_predictions.jsonl`
- `scored_predictions.jsonl`
- `summary.json`
- `checkpoint.json`

#### Suite runs (`scripts/reproduce_all.sh`)

Default output root:

- `results/reproduce-all-beans_zero_naturelm_side_by_side/`

Per model, the script uses a subdirectory:

- `results/<run_id>/<model>/`

Within each model directory, `beans-next run --suite beans_zero_core ...` writes per-task subdirectories:

- `results/<run_id>/<model>/suite/beans_zero_core/<eval_task_id>/`

Each per-task directory contains:

- `predictions.jsonl`
- `processed_predictions.jsonl`
- `scored_predictions.jsonl`
- `summary.json`
- `checkpoint.json`

### Troubleshooting quick notes

- **`--config` doesn’t work**: this is expected in iteration 1. Use `scripts/reproduce*.sh`.
- **v1.1 keeps getting skipped**: either set `HF_TOKEN` (gated access) or run in stub mode with `NATURELM_STUB_MODE=1`.
- **Conformance fails**: the scripts run `scripts/check_launcher.sh` against the base URL; fix the launcher contract first before trusting any benchmark outputs.

