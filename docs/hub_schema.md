# Hub metadata

`test/metadata.parquet` contains the evaluation examples. All tiers use `messages` for the user prompt and expected assistant answer.

| Columns | Meaning |
| --- | --- |
| `id` | Unique example ID. |
| `tier`, `task` | Integer tier from 1 to 4, and task identifier. |
| `messages` | One user message with the prompt, followed by one assistant message with the expected answer. |
| `audio_paths` | Ordered audio paths relative to `test/`. One clip for tiers 1–3; reference clips followed by the query for tier 4. |
| `license` | Source license metadata, where available. |
| `metadata` | JSON text with task-specific annotations. Keys vary by task. |
| `source_datasets`, `source_audio_ids`, `source_file_paths` | Ordered source names, recording IDs or filenames, and source paths for each clip. |
| `source_urls` | Recording-specific URLs where established, otherwise null. |
| `audio_start_seconds`, `audio_end_seconds` | Source crop boundaries in seconds, where established. |
| `source_id_types` | Meaning of each source identifier. |

Load every entry in `audio_paths`. Keep the clip order and repeated paths. For tier 4, the final clip is the query.

Give the model the user prompt and audio. Use the assistant message for scoring. The supplied clips already include any crops.

Source lists follow the same audio order. `source_file_paths` describes origins, not repository download paths. Null means that no value is provided.
