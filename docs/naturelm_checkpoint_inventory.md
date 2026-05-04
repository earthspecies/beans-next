## NatureLM-audio-1.5 checkpoint inventory (GCS)

- **Increment**: I17
- **Task**: I17-A
- **Date**: 2026-04-27
- **Base path**: `gs://foundation-models/naturelm-audio-1.5/all_backup/`

### GCS access status (local host)

- **Local auth**: WORKS
  - `gcloud auth list`: active account `marius@earthspecies.org`
  - `gsutil ls gs://foundation-models/naturelm-audio-1.5/all_backup/` succeeded

### Loader expectations (what `serve.py` will look for)

From `examples/servers/naturelm-v1.1/serve.py`:

- **Config file**: `config.json` preferred; if missing, falls back to `model.json`.
- **Checkpoint directory**: prefers `checkpoint/` subdir if it exists; otherwise uses the snapshot root directory.
- **Implication for GCS URIs**: `NATURELM_GCS_CHECKPOINT_URI` should point at a prefix that, once downloaded, contains either:
  - `config.json` or `model.json` at the root, and
  - either a `checkpoint/` directory or the weight/layout files directly under the root.

### Top-level checkpoint prefixes present under `all_backup/`

Verified via `gsutil ls gs://foundation-models/naturelm-audio-1.5/all_backup/` (non-recursive):

- `drasdic_multi_audio_v0.00/`
- `drasdic_multi_audio_v0.01/`
- `drasdic_multi_audio_v0.02/`
- `merged_variations_f0_v1/`
- `merged_variations_f0_v2/`
- `merged_variations_f0_v3/`
- `merged_variations_f0_v4/`
- `merged_variations_f0_v5/`
- `stage_1_16khz_v0.01/`
- `stage_1_16khz_v0.04/`
- `stage_1_32khz/`
- `stage_1_32khz_v0.0/`
- `stage_1_a100/`
- `stage_1_h100_3/`
- `stage_1_qwen35_2b_32khz_f0_multimodal_v0.01/`
- `stage_1_qwen35_2b_32khz_frozen_multitask_highres_v0.00/`
- `stage_1_qwen35_2b_32khz_lora_from_frozen_100k_multitask_highres_v0.00/`
- `stage_1_qwen35_2b_32khz_v0.03/`
- `stage_1_qwen35_32khz_frozen_multitask_highres_v0.01/`
- `stage_1_qwen35_32khz_highres_v0.01/`
- `stage_1_qwen35_32khz_highres_v0.02/`
- `stage_1_qwen35_32khz_lora_multitask_highres_v0.00/`
- `stage_1_qwen35_32khz_lora_multitask_highres_v0.01/`
- `stage_1_qwen35_32khz_lora_multitask_highres_v0.03/`
- `stage_1_qwen35_32khz_qlora_multitask_highres_v0_qlorano/`
- `stage_1_qwen35_32khz_qlora_multitask_highres_v0_raw2/`
- `stage_2_32khz_v0.03/`
- `stage_2_ft_variations/`
- `stage_2_qwen35_2b_32khz_lora_multimodal_sed/`
- `stage_2_qwen35_32khz_highres_from_multitask_v0.00/`
- `stage_2_qwen35_32khz_highres_from_multitask_v0.03/`
- `stage_2_qwen35_32khz_highres_from_multitask_v0.04/`
- `stage_2_qwen35_32khz_highres_from_multitask_v0.05/`
- `stage_2_qwen35_32khz_highres_from_multitask_v0.06/`
- `stage_2_qwen35_32khz_highres_from_multitask_v0.07/`
- `stage_2_qwen35_32khz_highres_v0.00_bf16/`
- `stage_2_qwen35_32khz_highres_v0.00_bf16bf3/`
- `stage_2_qwen35_32khz_highres_v0.00_bf16bf4nofa/`

### Per-prefix structure table (what’s directly under each top-level prefix)

This table is about the *top-level prefixes* under `all_backup/` (not deeper step subdirs).

Legend:
- **Leaf checkpoint prefix**: looks directly runnable (contains `model.json` or `config.json` at the top level).
- **Container prefix**: contains subdirectories (often numeric step dirs) that appear to be the actual checkpoints.

| Prefix | GCS URI | Direct children (sample) | Has `config.json`? | Has `model.json`? | Has `checkpoint/`? | Notes / compatibility expectation |
|---|---|---|---:|---:|---:|---|
| `merged_variations_f0_v1` | `gs://.../merged_variations_f0_v1/` | `manifest.json`, `model.json`, `model/` | no | yes | no | **Leaf prefix**. Likely usable with config=`model.json`, ckpt_dir=root. |
| `merged_variations_f0_v2` | `gs://.../merged_variations_f0_v2/` | (same pattern as v5) | no | yes | no | **Leaf prefix**. Same as above. |
| `merged_variations_f0_v3` | `gs://.../merged_variations_f0_v3/` | (same pattern as v5) | no | yes | no | **Leaf prefix**. Same as above. |
| `merged_variations_f0_v4` | `gs://.../merged_variations_f0_v4/` | (same pattern as v5) | no | yes | no | **Leaf prefix**. Same as above. |
| `merged_variations_f0_v5` | `gs://.../merged_variations_f0_v5/` | `manifest.json`, `model.json`, `model/` | no | yes | no | **Leaf prefix**. `model/` contains `audio_encoder/`, `audio_qformer/`, `llm/`. |
| `stage_1_16khz_v0.04` | `gs://.../stage_1_16khz_v0.04/` | `5000/` | no | no | no | **Container prefix**. Step dir `5000/` contains `model.json`, `training_state.pt`, `model/` (and patch/manifest). Use the step dir as the checkpoint URI. |
| `stage_2_qwen35_32khz_highres_from_multitask_v0.07` | `gs://.../stage_2_qwen35_32khz_highres_from_multitask_v0.07/` | many step dirs (e.g. `100000/`, `105000/`, ...), plus `backends/` | no | no | no | **Container prefix**. Step dir `100000/` contains `model.json`, `training_state.pt`, `model/` (and patch/manifest). Use the step dir as the checkpoint URI. |
| `drasdic_multi_audio_v0.02` | `gs://.../drasdic_multi_audio_v0.02/` | `backends/` | no | no | no | **Container prefix** (observed only `backends/train/` + `backends/val/` at top level). Likely not directly usable as a checkpoint root for `from_checkpoint_dir` without selecting a deeper checkpoint dir. |
| *(all other non-merged prefixes)* | `gs://.../<prefix>/` | typically subdirectories | no | no | no | Based on probing for `config.json`/`model.json`/`checkpoint/` at the top level, these appear to be **container prefixes**. Expect to select a *deeper* step directory that contains `model.json`/`config.json`. |

### Recommended sweep order (top 5 to run first)

These are the most “leaf-like” and align best with the current loader (top-level has `model.json`).

1. `merged_variations_f0_v5`
2. `merged_variations_f0_v4`
3. `merged_variations_f0_v3`
4. `merged_variations_f0_v2`
5. `merged_variations_f0_v1`

### Checkpoints to skip (for now) and why

- **Skip top-level `stage_*` prefixes**: they appear to be containers of multiple step directories; the top-level itself does not have `model.json`/`config.json`, so it likely won’t load unless the URI points at a specific step (e.g. `.../stage_1_16khz_v0.04/5000/`).
- **Skip top-level `drasdic_multi_audio_v*` prefixes**: observed only `backends/` at the top level (no `model.json`/`config.json`), so not an obvious `from_checkpoint_dir` root without deeper selection.

