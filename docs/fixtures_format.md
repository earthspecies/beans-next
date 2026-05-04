# BEANS-Next regression fixtures — format spec (v1)

This document specifies a **versioned, CI-friendly fixture bundle format** for BEANS-Next regression fixtures derived from BEANS-Zero / NatureLM behavior.

The format is designed for a **two-phase workflow**:

- **Phase A (CPU-only)**: generate and validate fixture *inputs* + structural correctness; optionally run a dummy launcher sanity pass.
- **Phase B (GPU / real model)**: capture *golden* predictions + metrics from a real NatureLM-audio v1.0 launcher (and later, other launchers), and commit those outputs as regression targets.

## Non-goals

- This spec does **not** require NatureLM inference on CPU.
- This spec does **not** prescribe the exact fixture generator implementation (see Increment 7 tasks I7-A2/I7-A3).

## Hard requirements (must)

- **Bundle location**: every fixture bundle lives under `tests/fixtures/`.
- **Versioning**: every bundle declares a `fixture_format_version` and is readable by future tooling.
- **Repro metadata**: every bundle records:
  - **pinned model identity fields** from `/info`: `name`, `model`, `model_revision`
  - best-effort **runner/prompt/postprocess/scorer** versions or hashes (when available)
  - an **exact regeneration command** (copy/paste, including required env vars)
- **CI-friendly size**:
  - avoid large binaries
  - if audio is included, use **tiny WAVs** (a few seconds) and/or **base64 WAV snippets**
- **Two-phase workflow support**: bundle structure must allow Phase A to exist without Phase B goldens present yet (placeholders allowed).

## Bundle identity and naming

A fixture bundle is a directory:

```
tests/fixtures/<bundle_id>/
```

Where:

- `<bundle_id>` is **stable** and filesystem-safe (recommended: kebab-case).
- A bundle may contain multiple “cases” (samples) and multiple “expected outputs” variants (e.g., different launcher versions), but must keep each variant explicitly named.

Recommended naming pattern:

- `<bundle_id> = naturelm-v1_0__beans_zero_core__tiny_v1`

## Directory layout (normative)

Each bundle MUST contain the following files/directories:

```
tests/fixtures/<bundle_id>/
  manifest.yaml
  inputs/
    slice.json
    requests.jsonl
    audio/
      README.md
      <sample_id>.wav            # optional; tiny
      <sample_id>.wav.base64.txt # optional alternative to wav files
  expected/
    README.md
    predictions.jsonl            # may be empty in Phase A
    processed_predictions.jsonl  # may be empty in Phase A
    scored_predictions.jsonl     # may be empty in Phase A
    summary.json                 # may be empty in Phase A
    model_identity.json          # may be empty in Phase A
  metadata/
    provenance.md
    versions.json
    regeneration.md
```

Notes:

- The `expected/` directory **may be placeholder-only** until Phase B.
- The `audio/` directory is optional if the inputs don’t require inline audio (e.g., “file_path” payloads); however, **CI fixtures should prefer** `base64_wav` snippets or tiny WAVs so they are self-contained.

## `manifest.yaml` (normative)

`manifest.yaml` is the entrypoint describing bundle contents and invariants.

### Required keys

- `fixture_format_version` (string): **must be** `1` for this spec.
- `bundle_id` (string): must match the directory name.
- `created_at_utc` (string, RFC3339): bundle creation time.
- `description` (string): human summary.
- `phase` (string): one of:
  - `phase_a_inputs_only`
  - `phase_b_golden_captured`
- `model_identity` (object):
  - `source`: `info_endpoint`
  - `info` (object):
    - `name` (string)
    - `model` (string)
    - `model_revision` (string)
  - `info_captured_at_utc` (string, RFC3339) — when `/info` was captured
- `inputs` (object):
  - `slice_path` (string): must be `inputs/slice.json`
  - `requests_path` (string): must be `inputs/requests.jsonl`
  - `audio` (object):
    - `payload_type` (string): `base64_wav` or `file_path` (CI should prefer `base64_wav`)
    - `storage` (string): one of `inline_base64_txt`, `wav_files`, `none`
- `expected` (object):
  - `variant_id` (string): identifies the golden capture target (e.g., `naturelm-v1.0__gpu_run_2026-04-xx`)
  - `predictions_path` (string): `expected/predictions.jsonl`
  - `processed_predictions_path` (string): `expected/processed_predictions.jsonl`
  - `scored_predictions_path` (string): `expected/scored_predictions.jsonl`
  - `summary_path` (string): `expected/summary.json`
  - `model_identity_path` (string): `expected/model_identity.json`
- `regenerate` (object):
  - `command` (string): exact command(s) to regenerate (see “Regeneration”)
  - `notes` (string): optional caveats (tokens, GPU requirements, etc.)

### Optional keys (recommended)

- `source_snapshot` (object):
  - `beans_zero_reference` (object): a *descriptive* reference to the BEANS-Zero snapshot consulted, without importing it at runtime:
    - `repo_path_hint` (string): e.g. `~/code/esp-research__beans-zero-project`
    - `branch` (string)
    - `commit` (string) — if known
- `compat` (object):
  - `predictions_schema_version` (string): must be `predictions_v1`
  - `audio_placeholder` (string): recommended: `<Audio><AudioHere></Audio>`
- `ci` (object):
  - `max_total_bytes` (int): recommended CI budget for this bundle
  - `max_audio_seconds` (float): recommended per-sample cap

### Example `manifest.yaml`

```yaml
fixture_format_version: "1"
bundle_id: "naturelm-v1_0__beans_zero_core__tiny_v1"
created_at_utc: "2026-04-20T19:05:00Z"
description: "Tiny BEANS-Zero slice for NatureLM-audio v1.0 regression fixtures."
phase: "phase_a_inputs_only"

model_identity:
  source: "info_endpoint"
  info:
    name: "naturelm-v1.0-launcher"
    model: "EarthSpeciesProject/NatureLM-audio"
    model_revision: "UNKNOWN_YET"
  info_captured_at_utc: "2026-04-20T19:05:00Z"

inputs:
  slice_path: "inputs/slice.json"
  requests_path: "inputs/requests.jsonl"
  audio:
    payload_type: "base64_wav"
    storage: "inline_base64_txt"

expected:
  variant_id: "naturelm-v1.0__golden_pending"
  predictions_path: "expected/predictions.jsonl"
  processed_predictions_path: "expected/processed_predictions.jsonl"
  scored_predictions_path: "expected/scored_predictions.jsonl"
  summary_path: "expected/summary.json"
  model_identity_path: "expected/model_identity.json"

regenerate:
  command: |
    # Phase A (CPU-only)
    uv run python -m beans_next_fixtures.generate \
      --bundle tests/fixtures/naturelm-v1_0__beans_zero_core__tiny_v1 \
      --seed 123 \
      --max-audio-seconds 2.0

    # Phase B (GPU; run on a GPU machine)
    export PREDICT_URL="http://localhost:8000/predict"
    uv run python -m beans_next_fixtures.capture_golden \
      --bundle tests/fixtures/naturelm-v1_0__beans_zero_core__tiny_v1 \
      --predict-url "$PREDICT_URL"
  notes: "Phase B requires NatureLM v1.0 launcher running with GPU and weights available."

compat:
  predictions_schema_version: "predictions_v1"
  audio_placeholder: "<Audio><AudioHere></Audio>"

ci:
  max_total_bytes: 2000000
  max_audio_seconds: 2.0
```

## `inputs/slice.json` (normative)

`inputs/slice.json` describes **which dataset rows** this fixture targets, independent of any particular launcher.

### Required fields

- `dataset_id` (string): BEANS-Next dataset registry id (e.g. `beans_zero`)
- `split` (string): HF split name or equivalent (e.g. `test`)
- `selection_method` (string): one of:
  - `explicit_sample_ids` (preferred)
  - `deterministic_query` (allowed if stable across time and datasets)
- `samples` (array): each item must include:
  - `sample_id` (string): the BEANS-Next `sample_id` used end-to-end
  - `source` (object): minimal reference to the underlying dataset row, e.g.:
    - `hf_dataset` (string): e.g. `EarthSpeciesProject/BEANS-Zero`
    - `subset` (string): subset identifier (e.g. `esc50`, `watkins`, etc.)
    - `row_index` (int) or `row_key` (string) — whichever BEANS-Next uses deterministically
  - `task` (object): minimal eval-task identity:
    - `eval_task_id` (string) if a registry task exists
    - OR `task_type` (string) plus additional task fields if needed

### Audio strategy (required to state)

`slice.json` must state whether audio is vendored:

- `audio_strategy.type` must be one of:
  - `vendored_wav`
  - `vendored_base64_wav`
  - `external_hf_audio` (discouraged for CI; allowed for Phase B capture workflows)

If vendored, the slice must specify per-sample audio paths:

- `audio.path`: `inputs/audio/<sample_id>.wav` OR `inputs/audio/<sample_id>.wav.base64.txt`
- `audio.sample_rate_hz`: required if the base64 file does not encode sample rate elsewhere

## `inputs/requests.jsonl` (normative)

`inputs/requests.jsonl` is a line-delimited JSON file where each line is a **single request item** matching the BEANS-Next `predictions_v1` **request element** shape (not the outer batch wrapper).

Each JSON object MUST include:

- `sample_id` (string)
- `messages` (array of `{role, content}`)
- `audio_inputs` (array), where each element includes at least:
  - `payload_type` (string): `base64_wav` or `file_path`
  - `data` (string): base64 bytes if `base64_wav`, or a relative path if `file_path`
  - `sample_rate` (int): recommended for `base64_wav`
- `generation_config` (object): include at minimum:
  - `temperature` (number)
  - `max_tokens` (int)

### Normalization rules

To keep fixtures stable across machines:

- `file_path` payloads MUST use **relative paths** rooted at the bundle directory.
- `base64_wav` payloads SHOULD be generated from a **canonical WAV encoding** (PCM 16-bit recommended) to avoid tool-dependent differences.
- `generation_config` MUST be fully specified (no “use server defaults”).

## `expected/*` (normative)

The `expected/` directory holds the **golden regression targets** captured in Phase B.

### Placeholders (Phase A)

Until Phase B capture happens, the files may exist but be empty/minimal:

- `expected/README.md` must explain what is missing and how to capture it.
- `expected/predictions.jsonl` may be empty (0 lines) or contain sentinel records (discouraged).
- `expected/summary.json` may be `{}`.

### Golden semantics (Phase B)

Once Phase B is complete:

- `expected/predictions.jsonl` MUST match BEANS-Next’s `predictions.jsonl` artifact row schema (one `ModelPrediction` per `sample_id`).
- `expected/processed_predictions.jsonl` and `expected/scored_predictions.jsonl` MUST match BEANS-Next artifact schemas (one row per `sample_id`).
- `expected/model_identity.json` MUST include the `/info` snapshot (at least `name`, `model`, `model_revision`) and any other identity fields BEANS-Next writes.
- `expected/summary.json` MUST match BEANS-Next’s `RunSummary` schema (or an explicitly defined stable subset if the schema is still evolving).

If BEANS-Next schemas evolve, the fixture tooling must either:

- provide a migration path keyed by `fixture_format_version`, or
- bump `fixture_format_version` and define a new spec (preferred when changes are breaking).

## `metadata/versions.json` (normative)

`metadata/versions.json` records best-effort **version identifiers** for the BEANS-Next pipeline pieces that affect determinism.

It MUST include:

- `beans_next`:
  - `package_version` (string) if available
  - `git_sha` (string) if available
- `schemas`:
  - `predictions_wire_schema` (string): must be `predictions_v1`
- `prompt`:
  - `prompt_id` (string) and `prompt_version` (string) if known
- `postprocess`:
  - `pipeline_id` (string) and `pipeline_version` (string) if known
- `scorers`:
  - mapping of scorer ids to version/hash strings if known

It SHOULD include:

- `generator`:
  - `tool_name` and `tool_version` (if the generator is a separate module)
- `python`:
  - `python_version`

## `metadata/provenance.md` (normative)

`metadata/provenance.md` is human-readable and must contain:

- why this slice was chosen (coverage goals, size budget)
- any known quirks (non-determinism risks, special postprocess rules)
- licensing note (audio is derived from BEANS-Zero dataset; include only what is permitted)

## `metadata/regeneration.md` (normative)

`metadata/regeneration.md` is human-readable “how to regenerate” guidance and MUST include:

- **Phase A (CPU-only)** regeneration steps
- **Phase B (GPU)** golden capture steps
- required environment variables (e.g., `HF_TOKEN`, `PREDICT_URL`)
- expected outputs (which files get overwritten)

The `manifest.yaml`’s `regenerate.command` must be the **copy/paste** form; `regeneration.md` can add explanation.

## Two-phase workflow details (normative)

### Phase A (CPU-only): input generation + structural validation

Phase A must be able to run in CPU-only CI without any NatureLM inference and should ensure:

- the bundle directory layout is correct
- `manifest.yaml` parses and required fields exist
- `inputs/requests.jsonl` is well-formed JSONL and matches the expected request-item schema
- any vendored audio files are present and within the size/time budgets
- optional: a dummy-launcher sanity run that can execute the requests end-to-end and produce *non-golden* outputs (not committed to `expected/`)

Phase A explicitly does **not** require `expected/*` to be populated.

### Phase B (GPU): golden capture from real NatureLM v1.0 launcher

Phase B runs on a GPU machine with a real NatureLM v1.0 launcher. It must:

- capture `/info` and write pinned identity fields into:
  - `manifest.yaml` → `model_identity.info.*`
  - `expected/model_identity.json`
- run BEANS-Next against **exactly** the inputs in `inputs/requests.jsonl` (or against `slice.json` if the runner is used directly) and write golden artifacts into `expected/`
- keep `inputs/` unchanged (goldens must correspond to the committed inputs)

If the NatureLM launcher or weights change:

- capture a new golden variant by updating `expected.variant_id` and the identity fields, or
- create a new bundle id if the change is large enough to justify separate fixtures.

## Size budgets (recommended defaults)

Unless a task requires otherwise, a bundle should aim for:

- total bundle size ≤ **2 MB**
- number of samples ≤ **20**
- audio duration per sample ≤ **2 seconds**
- single WAV file size ≤ **200 KB**

## Determinism notes and tolerances

Some outputs can be sensitive to:

- sampling settings (`temperature`, `top_p`, etc.)
- upstream tokenization/model revisions
- floating point nondeterminism in scorers

Fixture tooling and tests should:

- force deterministic generation settings for goldens where possible (e.g., `temperature: 0`)
- treat `/info.model_revision` changes as **expected invalidation** of goldens
- define explicit tolerances only when unavoidable, and record them in `manifest.yaml` (future extension: `expected.tolerances`).

## Forward-compatibility

This spec is **fixture_format_version = 1**.

Breaking changes (directory layout changes, schema changes, new required metadata) must:

- increment `fixture_format_version`
- keep old bundles readable by old tooling (best-effort), or provide a migration tool.

