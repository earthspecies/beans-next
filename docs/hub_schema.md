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
| `source_dataset`, `license` | Normalized source name and license metadata, where available. |
| `metadata` | JSON text with task-specific annotations. Keys vary by task. |
| `source_datasets`, `source_audio_ids`, `source_file_paths` | Ordered source names, recording IDs or filenames, and source paths for each clip. |
| `source_urls` | Recording-specific URLs where established, otherwise null. |
| `audio_start_seconds`, `audio_end_seconds` | Source crop boundaries in seconds, where established. |
| `source_id_types`, `provenance_status` | Identifier meanings and evidence supporting each source mapping. |

For tier 4, load `context_audio_paths + [query_audio_path]`. Keep every clip in
order, including repeated paths. Supply the user prompt and audio to the model.
Use the assistant message only for scoring.

Crop and event annotations describe source material. The supplied audio already
contains the benchmark clips. Do not crop it again from those annotations.
The annotation `duration_sec`, where present, retains its existing value.
It is not a replacement for reading the supplied file's actual duration or sample rate.

## Provenance

All source lists follow evaluation audio order, including repeated clips and the
final tier 4 query. `source_file_paths` describes origins and is never used as an
evaluation download path. Unknown values remain null.

`source_audio_ids` distinguishes recording IDs from source or derived filenames
through `source_id_types`. Ambiguous iNaturalist sound IDs remain null. DCASE and
gibbon clips retain their exact filenames without invented raw crop timestamps.

`provenance/metadata.parquet` contains `id`, `sample_id`, `source_id`,
`original_fields`, and `added_columns`. `source_id` is construction bookkeeping,
not an original recording ID.
Join it to the main table by `id`. Evaluation does not need this file.

`original_fields` is JSON text. It preserves removed columns and the previous
values of cleaned fields, including source paths, construction settings, quality
checks, legacy prompt fields, and the original metadata object.

To reconstruct a previous row exactly:

```python
import json

added = set(json.loads(provenance_row.get("added_columns") or "[]"))
original_row = {k: v for k, v in compact_row.items() if k not in added}
original_row.update(json.loads(provenance_row["original_fields"]))
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

This base conversion produces the earlier 12-column compact schema. Source
enrichment uses `ordered_source_provenance` in
`beans_next.datasets.hub_provenance` with the original rows and source manifests.
It adds the eight ordered source fields and moves `source_id` into provenance,
giving the enriched table 19 columns. Rebuild `original_fields` against the
enriched row and list added fields in `added_columns` to retain exact reversal.
`restore_provenance_row` handles both enriched and earlier compact tables.

The bundle verifier accepts both schemas and checks tier 4 audio as well as
single-audio examples:

```bash
uv run python scripts/verify_beans_next_hf_bundle.py --bundle compact-bundle --sample 300
```

This check needs the referenced audio files under the bundle's `test/audio/`
directory. Use `--sample 0` to check every example.
