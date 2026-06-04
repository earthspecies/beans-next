# BEANS-Next dataset structures (authoritative)

This document defines the **authoritative, implementation-aligned** row contracts for the
two BEANS-Next datasets implemented in this repo:

- `beans_next/datasets/beans_next.py` → `BEANSNext` (**single audio**)
- `beans_next/datasets/beans_next_multiaudio.py` → `BEANSNextMultiAudio` (**multi audio**)

The intent is that benchmark loaders, prompt renderers, and scorers can rely on these
contracts without depending on undocumented behavior of upstream sources.

## `BEANSNext` (single-audio)

### Split inventory

`BEANSNext.info.split_paths` defines the split ids. At time of writing:

- `crow-description`
- `zebra-description`
- `f0-mean-seen-taxa`
- `f0-mean-heldout-taxa`
- `bird-presence`
- `mammal-presence`
- `insect-presence`
- `amphibian-presence`
- `alarm-call-presence`
- `flight-call-presence`
- `call-type-fixed-vocab`

Each split is backed by a JSONL file (one JSON object per line) and a corresponding audio
root. `BEANSNext` resolves each row’s audio path as:

`anypath(data_root) / row["audio_path_original_sample_rate"]`

When `data_root=None`, the class chooses a per-split default from `_default_data_roots`.

### Input row schema (JSONL)

The loader expects the following **minimum** fields in each JSON object:

- `instruction` (str): prompt text. Must include exactly one `<Audio><AudioHere></Audio>`
  placeholder.
- `output` (str): gold target label for the task.
- `audio_path_original_sample_rate` (str): relative path under `data_root` to the audio file.
- `metadata` (str): JSON-encoded string (i.e., a string whose content is JSON).

In practice, upstream JSONL rows also include extra bookkeeping keys (examples include
`id`, `task`, `dataset_name`, `source_dataset`, `license`, `file_name`, `instruction_text`).
`BEANSNext` passes these through unchanged unless `output_take_and_give` is used.

### Output row schema (Python dict)

Indexing (`ds[i]`) returns a Python `dict[str, object]` containing all input keys plus:

- `audio` (np.ndarray): mono `float32` waveform at `sample_rate` if resampling was requested,
  otherwise the original sample rate.

If `output_take_and_give` is provided, the returned dict contains only the mapped keys.

### Task-specific `output` conventions

`BEANSNext` does not itself parse targets; it returns raw strings.

Observed conventions (from existing BEANS-Next JSONL docs in this repo):

- **4-way MCQ** (`*-description`): `output ∈ {"A","B","C","D"}` — `task_type: classification`
- **T3 full-label MCQ** (`t3-*-mcq`): `output` is a full option string like `"(A) Thrush nightingale"` —
  `task_type: classification`. The scorer accepts bare letters, option text alone, letter + text,
  and option text embedded in a sentence; matching is case-insensitive.
- **Binary presence** (`*-presence`, `t3-vocalization-*-binary`): `output ∈ {"Yes","No"}` — `task_type: presence_binary` (case-insensitive)
- **Mean F0** (`f0-mean-*`): `output` like `"3100 Hz"` — `task_type: regression`, MAE on Hz
- **SNR** (`t1-snr-regression`): `output` like `"39 dB"` — `task_type: regression`, MAE on dB
- **Count OE** (`t3-species-count-oe`, `t3-vocalization-count-total-oe`): `output` integer — `task_type: regression`, MAE
- **Captioning** (`t1-caption`, `t2-captioning`, `t3-structural-captioning`): free-form text — `task_type: captioning`, corpus CIDEr
- **Fixed vocab multilabel** (`call-type-fixed-vocab`): comma-separated labels, e.g. `"flight call, call"`
- **T3 OE single species** (`t3-species-by-*-oe`, `t3-species-by-vocalization-order-oe`):
  single species name — `task_type: species_name`. Labels are scientific names; prompts are
  format-agnostic. The scorer accepts both scientific and common names via the bundled lookup
  `registry/t3_species_names.json` (46 species, English common names from GBIF June 2026).
  Multiple common name variants per species are supported (e.g. `Thekla Lark` /
  `Thekla's Lark` for `Galerida theklae`). Species outside the lookup fall back to exact match.
- **T3 per-species count** (`t3-vocalization-count-per-species-oe`):
  `"Species A: N, Species B: M"` (scientific names) — `task_type: species_count_dict`,
  metrics: `species_f1` + `count_mae` over matched species.
- **T3 per-species frequency range** (`t3-frequency-range-description`):
  `"Species A: 2440-5130 Hz, Species B: 1510-10370 Hz"` — `task_type: species_freq_range`,
  metrics: `species_f1`, `freq_mae_low`, `freq_mae_high`, `freq_mean_iou` over matched species.
  `Unknown` entries excluded from both sides.
- **T3 ordered species summary** (`t3-ordered-species-summary`):
  `"Species A: 2 calls, 2330-5150 Hz; Species B: 1 call, 4650-7320 Hz"` — `task_type: species_summary`,
  metrics: `species_f1`, `count_mae`, `freq_mae_low`, `freq_mae_high`, `freq_mean_iou` over matched species.
  `None` label when no identifiable species present; `Unknown` excluded.

## `BEANSNextMultiAudio` (multi-audio)

### Split inventory

`BEANSNextMultiAudio` defines split ids in the module-level `_SPLITS` mapping. At time of
writing it includes (non-exhaustive):

- `gibbon-fewshot-detection`
- `gibbon-fewshot-detection-balanced`
- `giant-otter-4way`
- `dcase-fewshot-detection-balanced`
- `crow-4way`
- `zebra-4way`
- `unseen-species-4way`, `unseen-species-4way-hard`
- `unseen-genus-4way`, `unseen-genus-4way-hard`
- `unseen-family-4way`, `unseen-family-4way-hard`

### Input row schema (JSONL)

Each JSON object must include:

- `audio_paths` (list[str]): non-empty list of relative audio paths, ordered to match prompt
  placeholder order.

Recommended / commonly present fields:

- `messages` (list[dict]): at least two entries:
  - `messages[0]`: `{ "role": "user", "content": "<prompt with N <AudioHere> tags>" }`
  - `messages[1]`: `{ "role": "assistant", "content": "<gold label>" }`
- `id` (str): stable row id.
- `task` (str): a task identifier; if absent, the dataset sets `task = split`.

Audio is resolved as `anypath(data_root) / rel_path` for each element of `audio_paths`.
If `data_root=None`, the dataset chooses a split-specific default.

### Output row schema (Python dict)

Indexing (`ds[i]`) returns a Python `dict[str, object]` containing all input keys plus:

- `audios` (list[np.ndarray]): list of mono `float32` waveforms, one per `audio_paths` entry.
- `task` (str): ensured present (`row.get("task", split)`).

If `output_take_and_give` is provided, the returned dict contains only the mapped keys.

### Multi-audio prompt alignment requirement

Consumers must preserve ordering:

- The i-th element of `audios` corresponds to the i-th element of `audio_paths`.
- For JSONLs that include `messages[0].content`, the number/order of `<AudioHere>` placeholders
  is expected to align with this audio ordering.

## Serving note (authoritative): audio clipping is a launcher concern

BEANS-Next datasets (`BEANSNext`, `BEANSNextMultiAudio`) return full audio waveforms by design. Any
audio **clipping/capping** (e.g., “first 30 seconds only”) is intentionally handled at the **launcher**
layer (model server / adapter), not in the dataset contract.

As of 2026-04-28, the `vllm` adapter sidecar supports optional canonicalization + clipping of
`base64_wav` payloads via environment variables:

- `VLLM_ADAPTER_MAX_AUDIO_SECONDS` (e.g. `30`)
- `VLLM_ADAPTER_CANONICALIZE_WAV=1`

This mirrors the OpenAI-compatible proxy launcher’s `OPENAI_PROXY_MAX_AUDIO_SECONDS` behavior and is
recommended for Qwen omni models to reduce multimodal token pressure without changing dataset semantics.

