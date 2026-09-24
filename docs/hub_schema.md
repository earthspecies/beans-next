# Compact Hub metadata

The compact release stores evaluation examples in `test/metadata.parquet`.
All four tiers use the same conversation format. Audio paths and both identifiers
retain their previous values.

| Columns | Meaning |
| --- | --- |
| `id`, `sample_id` | Stable example identifiers. Both are retained for compatibility. |
| `tier`, `task` | Integer tier from 1 to 4, and task identifier. |
| `messages` | One user message with the prompt, followed by one assistant message with the expected answer. |
| `file_name` | Single-audio path, relative to `test/`. Null for tier 4. |
| `context_audio_paths`, `query_audio_path` | Tier 4 reference clips in prompt order, and the final query clip. Paths are relative to `test/`. |
| `source_dataset`, `source_id`, `license` | Source references and license metadata, where available. These are not audio download paths. |
| `metadata` | JSON text with task-specific annotations. Keys vary by task. |

For tier 4, load `context_audio_paths + [query_audio_path]`. Keep every clip in
order, including repeated paths. Supply the user prompt and audio to the model.
Use the assistant message only for scoring.

Crop and event annotations describe source material. The supplied audio already
contains the benchmark clips. Do not crop it again from those annotations.
The annotation `duration_sec`, where present, retains its existing value.
It is not a replacement for reading the supplied file's actual duration or sample rate.

## Provenance

`provenance/metadata.parquet` contains `id`, `sample_id`, and `original_fields`.
Join it to the main table by `id`. Evaluation does not need this file.

`original_fields` is JSON text. It preserves removed columns and the previous
values of cleaned fields, including source paths, construction settings, quality
checks, legacy prompt fields, and the original metadata object.

To reconstruct a previous row exactly:

```python
import json

original_row = {**compact_row, **json.loads(provenance_row["original_fields"])}
```

Paths to source data, templates, and construction configurations are provenance
references. They do not resolve to files in this dataset repository.

## Loader compatibility

The Hub loader accepts compact `messages` rows and older `instruction`/`output`
rows. Select the repository through the existing dataset configuration or the
`repo_id` argument to `iter_hf_beans_next_examples`.

The compact table removes `instruction`, `instruction_text`, `output`, `label`,
and alternate audio identifiers from the evaluation interface. External tools
that use those columns must read `messages` and the audio path columns instead.

## Build a compact bundle

```bash
uv run python scripts/compact_beans_next_hf.py legacy-metadata.parquet compact-bundle
```

This command writes metadata and provenance locally. It does not upload files
or copy audio. It checks exact reconstruction after writing both Parquet files.
Keep the source metadata until the release passes evaluation parity checks.

The bundle verifier accepts both schemas and checks tier 4 audio as well as
single-audio examples:

```bash
uv run python scripts/verify_beans_next_hf_bundle.py --bundle compact-bundle --sample 300
```

This check needs the referenced audio files under the bundle's `test/audio/`
directory. Use `--sample 0` to check every example.
