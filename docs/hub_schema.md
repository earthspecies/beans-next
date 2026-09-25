# Hub metadata

`test/metadata.parquet` contains the evaluation examples. All tiers use `messages` for the user prompt and expected assistant answer.

| Columns | Meaning |
| --- | --- |
| `id`, `sample_id` | Example ID; both contain the same value. |
| `tier`, `task` | Integer tier from 1 to 4, and task identifier. |
| `messages` | One user message with the prompt, followed by one assistant message with the expected answer. |
| `file_name` | Single-audio path, relative to `test/`. Null for tier 4. |
| `context_audio_paths`, `query_audio_path` | Tier 4 reference clips in prompt order, and the final query clip. Paths are relative to `test/`. |
| `source_dataset`, `license` | Normalized source name and license metadata, where available. |
| `metadata` | JSON text with task-specific annotations. Keys vary by task. |
| `source_datasets`, `source_audio_ids`, `source_file_paths` | Ordered source names, recording IDs or filenames, and source paths for each clip. |
| `source_urls` | Recording-specific URLs where established, otherwise null. |
| `audio_start_seconds`, `audio_end_seconds` | Source crop boundaries in seconds, where established. |
| `source_id_types` | Meaning of each source identifier. |

For tiers 1–3, load `file_name`. For tier 4, load `context_audio_paths` followed by `query_audio_path`. Paths are relative to `test/`. Keep the clip order and repeated paths.

Give the model the user prompt and audio. Use the assistant message for scoring. The supplied clips already include any crops.

Source lists follow the same audio order. `source_file_paths` describes origins, not repository download paths. Null means that no value is provided.
