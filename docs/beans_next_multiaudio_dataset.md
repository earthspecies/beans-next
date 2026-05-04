## BeansProMultiAudio dataset (esp-data `beans-next` branch) — split inventory + stats

This doc describes the **BeansProMultiAudio** dataset splits as stored under:

- **JSONL root**: `gs://esp-data-ingestion/beans-next/v0.1.0/raw/`
- **Audio paths**: rows contain relative paths like `audio/...` which resolve under the same root.

### Important note on source-code location

I attempted to inspect the required source file `esp_data/datasets/beans_next_multiaudio.py` on the `esp-data` `beans-next` branch, but it was **not present** in the checked-out branch at the time of analysis. All stats below were computed by inspecting the **actual JSONL** in GCS (metadata-only; no audio downloads).

---

## Per-split stats (from `test.jsonl` in each split directory)

All splits below are backed by a single JSONL at:
`gs://esp-data-ingestion/beans-next/v0.1.0/raw/<split_dir>/test.jsonl`, where `<split_dir>` is the split name with `-` replaced by `_`.

Key conventions observed across splits:

- **Multi-audio**: `audio_paths` is a list of strings.
- **Prompt**: stored in `messages[0]` (role `user`) as a single string containing one `<Audio><AudioHere></Audio>` per audio file.
- **Gold output**: stored in `messages[1]` (role `assistant`) as a short string (e.g. `"A"`, `"None"`).
- **Placeholder count**: `messages[0].content.count("<AudioHere>")` **always matched** `len(audio_paths)` for the inspected splits.

### Table

| Split | N | `audio_paths` per row (min/median/max) | `<AudioHere>` per row (min/median/max) | `<AudioHere>` == audio count? | Output format | Output examples (most common) | `task` value | JSONL URI | Audio root |
|---|---:|---:|---:|---|---|---|---|---|---|
| `gibbon-fewshot-detection-balanced` | 868 | 4 / 5 / 5 | 4 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`None` | `None` (434), `A` (370), `B` (42), `C` (22) | `fewshot_detection` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/gibbon_fewshot_detection_balanced/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `crow-4way` | 200 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | `A` (50), `B` (50), `C` (50), `D` (50) | `call_type_multiple_choice` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/crow_4way/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `giant-otter-4way` | 500 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | all ~125 each | `call_type_multiple_choice` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/giant_otter_4way/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `dcase-fewshot-detection-balanced` | 3,158 | 5 / 5.5 / 6 | 5 / 5.5 / 6 | yes (0 mismatches) | `A`/`B`/`C`/`D`/`None` | `None` (1579) then `D`/`B`/`A`/`C` | `fewshot_detection` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/dcase_fewshot_detection_balanced/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `unseen-species-4way` | 1,227 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | ~uniform (307/307/307/306) | `species_mcq` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/unseen_species_4way/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `unseen-species-4way-hard` | 1,227 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | ~same as non-hard | `species_mcq` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/unseen_species_4way_hard/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `zebra-4way` | 40 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | 10 each | `call_type_multiple_choice` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/zebra_4way/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `unseen-genus-4way` | 927 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | ~uniform (232/232/232/231) | `species_mcq` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/unseen_genus_4way/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `unseen-genus-4way-hard` | 927 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | ~uniform | `species_mcq` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/unseen_genus_4way_hard/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `unseen-family-4way` | 440 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | 110 each | `species_mcq` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/unseen_family_4way/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |
| `unseen-family-4way-hard` | 440 | 5 / 5 / 5 | 5 / 5 / 5 | yes (0 mismatches) | `A`/`B`/`C`/`D` | 110 each | `species_mcq` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/unseen_family_4way_hard/test.jsonl` | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/` |

---

## Field inventory (observed keys in JSONL)

All inspected splits had the following top-level keys:

- `id` (str)
- `task` (str)
- `messages` (list[dict]): at least two entries: `user` then `assistant`
- `audio_paths` (list[str]): relative paths, typically under `audio/...`
- `audio_path_original_sample_rate` (str): a single path (appears to correspond to the **query** audio)
- `audio_ids` (list[str] or list[int]-like serialized): present
- `dataset_name`, `source_dataset`, `license`, `template_path`, `skills`, `metadata` (varies by split but present across all inspected splits)

Split-specific extras:

- `original_beans_zero_id`: present in gibbon/dcase/unseen-* splits.
- `original_beans_next_id`: present in `crow-4way` and `zebra-4way`.

### `messages` schema

Example structure (representative):

- `messages[0]`: `{ "role": "user", "content": "<prompt string with N <AudioHere> tags>" }`
- `messages[1]`: `{ "role": "assistant", "content": "<gold label>" }`

---

## Prompt template comparison vs BEANS-Zero

- **Tag format**: all inspected prompts use **exactly** `<Audio><AudioHere></Audio>` for each audio slot (same tag shape as BEANS-Zero single-audio prompts).
- **Placeholder count per example**: equals `len(audio_paths)` for every inspected row across splits.
- **Self-contained?**: yes — the `messages[0].content` already contains the full prompt text and the `<AudioHere>` slots; runner likely only needs to splice in audio payload(s) to match the number/order of `audio_paths`.

### Prompt examples (user message, truncated to ~400 chars)

- `crow-4way`:
  - `Here are four call types... A: <Audio><AudioHere></Audio> ... Which call type best matches ... <Audio><AudioHere></Audio>`
- `dcase-fewshot-detection-balanced`:
  - `Here are examples of 4 sounds... Which of the above sounds are present ... <Audio><AudioHere></Audio>`
- `unseen-species-4way`:
  - `Here are four species... Which species best matches ... <Audio><AudioHere></Audio>`

---

## Output format notes

- **MCQ-like splits** (`crow-4way`, `giant-otter-4way`, `zebra-4way`, unseen-* 4way): outputs are single letters `A`/`B`/`C`/`D`.
- **Few-shot detection splits** (`gibbon-fewshot-detection-balanced`, `dcase-fewshot-detection-balanced`): outputs include `"None"` plus letter(s); in inspected rows the output was a **single** letter or `"None"` (no comma-separated multi-label observed in the quick histogram, but downstream should not assume that cannot occur).

---

## GCS access status

- **PASS (metadata reachable)**: all split JSONLs listed above were readable via `gsutil cat`.
- **Audio objects**: not accessed (by design).

---

## Implications for runner/launcher (high-signal)

- **Multi-audio payload requirement is real**: prompts contain **N `<AudioHere>` tags** where \(N = len(audio_paths)\), and \(N\) can vary by split and by example (notably `gibbon-fewshot-detection-balanced` min 4; `dcase-fewshot-detection-balanced` up to 6).
- **Canonical prompt/output live in `messages`, not `instruction`/`output`**: any loader that expects `instruction`/`output` fields will see zeros/`None`. The runner should treat `messages[0].content` as the instruction and `messages[1].content` as the gold output.
- **Ordering matters**: since choices `A`–`D` are encoded in the prompt, launchers must preserve payload ordering so the \(i\)-th audio payload corresponds to the \(i\)-th `<AudioHere>` in `messages[0].content`.

