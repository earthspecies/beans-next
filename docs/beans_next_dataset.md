# BeansPro (single-audio) dataset — splits, schema, and prompt notes

This document summarizes **BeansPro v0.1.0** as registered in `esp_data.datasets.beans_next.BeansPro` (esp-data `origin/beans-next` branch), and verifies split-level statistics and example rows by streaming the public **GCS JSONL** files (metadata only; no audio downloads).

## How to access (esp-data)

- **Dataset class**: `esp_data.datasets.beans_next.BeansPro` (only present on esp-data `beans-next` branch at time of writing).
- **Source file**: `earthspecies/esp-data` → `esp_data/datasets/beans_next.py` (branch `beans-next`).
- **Note on local import**: In this `beans-next` repo environment, `import esp_data` works, but the installed `esp_data` does **not** include `datasets.beans_next`. Importing esp-data from a worktree via `sys.path.insert(0, "/home/marius_miron_earthspecies_org/code/esp-data")` failed due to a missing dependency (`cloudpathlib`). For this reason, the stats below are computed directly from the GCS JSONL split files (which are what `BeansPro._load()` reads).

## Splits (per-split stats + paths)

All splits live under `gs://esp-data-ingestion/beans-next/v0.1.0/raw/**/test.jsonl` and share the same top-level row schema (see next section).

| Split | N (counted from JSONL) | Task type (intended) | JSONL path |
|---|---:|---|---|
| `crow-description` | 200 | 4-way acoustic description MCQ (A/B/C/D) | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/carrion_crow_descriptions/test.jsonl` |
| `zebra-description` | 40 | 4-way acoustic description MCQ (A/B/C/D) | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/zebra_descriptions/test.jsonl` |
| `f0-mean-seen-taxa` | 2086 | mean F0 regression (seen taxa) | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/f0_mean_seen_taxa/test.jsonl` |
| `f0-mean-heldout-taxa` | 571 | mean F0 regression (held-out taxon) | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/f0_mean_heldout_taxa/test.jsonl` |
| `bird-presence` | 3478 | binary Yes/No presence | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/bird_presence/test.jsonl` |
| `mammal-presence` | 468 | binary Yes/No presence | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/mammal_presence/test.jsonl` |
| `insect-presence` | 1176 | binary Yes/No presence | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/insect_presence/test.jsonl` |
| `amphibian-presence` | 1818 | binary Yes/No presence | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/amphibian_presence/test.jsonl` |
| `alarm-call-presence` | 36 | binary Yes/No presence | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/alarm_call_presence/test.jsonl` |
| `flight-call-presence` | 192 | binary Yes/No presence | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/flight_call_presence/test.jsonl` |
| `call-type-fixed-vocab` | 999 | fixed-vocab multilabel (5 labels) | `gs://esp-data-ingestion/beans-next/v0.1.0/raw/call_type_fixed_vocab/test.jsonl` |

### Split → audio root resolution (esp-data behavior)

In `esp_data/datasets/beans_next.py`, `BeansPro` stores `audio_path_original_sample_rate` as a **relative path** and resolves it as:

`anypath(data_root) / row["audio_path_original_sample_rate"]`

If `data_root` is not provided, it defaults per split (`_default_data_roots`), notably:

- `crow-description`: `gs://esp-data-ingestion/beans-next/v0.1.0/raw/carrion_crow_descriptions/`
- `zebra-description`: `gs://esp-data-ingestion/beans-next/v0.1.0/raw/zebra_descriptions/`
- `f0-mean-*`: `gs://esp-data-ingestion/f0-prediction/audio/`
- presence + call-type splits: `gs://esp-ml-datasets/`

## Row schema (observed from JSONL)

For the first row of every split, the top-level keys were identical:

- `audio_path_original_sample_rate` (str): relative path to audio file (examples below).
- `dataset_name` (str)
- `file_name` (str)
- `id` (str): per-row identifier.
- `instruction` (str): full prompt, includes `<Audio><AudioHere></Audio>`.
- `instruction_text` (str): appears to mirror or simplify `instruction` (verify if needed).
- `license` (str)
- `metadata` (str): JSON-encoded object (string-valued field containing JSON).
- `output` (str): answer / label / target (format varies by split).
- `source_dataset` (str)
- `task` (str)

**Difference vs the I15 background assumption**: BeansPro rows include additional bookkeeping fields beyond the minimal BEANS-Zero-like quartet (`instruction`, `output`, `audio_path_original_sample_rate`, `metadata`). The core four are present, but there are extra keys that loaders should either pass through or ignore explicitly.

## Instruction / prompt format notes

Across all splits sampled, `instruction` begins with a **single** audio placeholder:

- `<Audio><AudioHere></Audio>`
- Observed `<AudioHere>` count in `instruction`: **1** for every split.

### MCQ (description) splits

- **Instruction**: question text + four labelled choices `A: ... B: ... C: ... D: ...` and an instruction like “Answer with the letter …”.
- **Output**: one of `A`, `B`, `C`, `D`.
- **Example (truncated)** from `crow-description`:
  - `instruction`: `<Audio><AudioHere></Audio> Which acoustic description best matches this sound? ... A: ... B: ...`
  - `output`: `A`

### Presence (Yes/No) splits

- **Instruction**: direct question, explicitly says “Answer Yes or No.”
- **Output**: `Yes` / `No` (string).
- Example from `bird-presence`:
  - `instruction`: `<Audio><AudioHere></Audio> Is there a bird vocalizing in this recording? Answer Yes or No.`
  - `output`: `No`

### F0 mean splits (regression)

- **Instruction**: `<Audio><AudioHere></Audio> What is the mean fundamental frequency of this vocalization?`
- **Output**: string with units, e.g. `3100 Hz`, `500 Hz`.
  - This is best treated as **regression**, but requires parsing the numeric value (strip `Hz`).

### Call type fixed vocab (multilabel)

- **Instruction**: “Choose all that apply: alarm call, flight call, begging call, song, call.”
- **Output**: observed as a comma-separated string when multiple labels apply.
  - Example: `flight call, call`
  - Also observed: single-label strings like `song`.

## Audio path field notes (observed examples)

The field `audio_path_original_sample_rate` is present in all splits and looks like:

- `crow-description`: `audio/calltype_3__00230123__2019_AL_Rosa_normed.wav`
- `zebra-description`: `audio/04030.snort.trim.wav` (note the split’s `metadata.sample_rate` may be 44100 even if models resample later)
- `f0-mean-seen-taxa`: `La_Palma_chaffinches/cut/FCP104_93.67.wav`
- `alarm-call-presence`: `beans-zero/v0.1.0/raw/audio/call-type/original_sample_rate/XC633022-MIXPRE-027.flac`

## Metric / evaluation implications (likely)

- **MCQ 4-way** (`crow-description`, `zebra-description`)
  - **Metric**: top-1 accuracy (exact match on `A|B|C|D`).
  - **Parsing**: strip whitespace; optionally accept `Answer: A`-style generations with a regex.

- **Binary presence** (bird/mammal/insect/amphibian/alarm/flight)
  - **Metric**: accuracy (and optionally balanced accuracy if class balance is imperfect in practice).
  - **Parsing**: canonicalize case; map `Yes/No`.

- **F0 regression** (`f0-mean-*`)
  - **Metric**: MAE / RMSE on Hz (after numeric extraction).
  - **Parsing**: extract first float/int from output string (e.g. `3100 Hz` → `3100`).

- **Fixed-vocab multilabel** (`call-type-fixed-vocab`)
  - **Metric**: micro-F1 / macro-F1, or example-wise Jaccard / exact-set-match.
  - **Parsing**: split by comma, trim, map into the fixed label set `{alarm call, flight call, begging call, song, call}`.

## Commands used to verify split stats (metadata-only)

Counts and samples were computed by streaming each JSONL file from GCS and parsing the first record + line count, without reading any audio objects.

