## BEANS-Next MultiAudio — schema extension decision + launcher capability matrix

Increment: **I16**  
Task: **I16-B**  
Date: **2026-04-27**

### Context

BEANS-Next currently uses a single-audio request shape (`payload_type`/`payload`) and prompt templates that reference audio via the BEANS-Zero placeholder tag **`<Audio><AudioHere></Audio>`**. Increment 16 introduces multi-audio dataset rows (multiple audio payloads per *single* item) and needs a wire-compatible schema extension plus clear per-launcher implementation guidance.

---

## Schema extension decision (Options A/B/C)

### Decision: **Option A**

Add an *optional* field `payloads: list[{payload_type: str, payload: str}]` alongside the existing single-audio `payload_type`/`payload` fields.

### Justification (5–10 lines)

- **Non-breaking**: existing beans-zero and beans-next single-audio requests keep using the current top-level `payload_type`/`payload` fields unchanged.
- **Incremental adoption**: only multi-audio dataset rows need to send `payloads`; launchers can be updated one-by-one without a flag day.
- **Clear invariants**: when `payloads` is present, its length should align with the number of `<AudioHere>` positions in the chosen prompt template (or follow a documented fallback mapping when no placeholders are present).
- **Avoids forced refactors**: Option B would require updating every launcher and any external client immediately.
- **Naming stays generic**: `payloads` can support future non-audio modalities (if ever needed) without renaming.

---

## Beans-zero regression plan (must run after future code changes)

Exact command from `INCREMENTS.md` I16-B:

```bash
beans-next run --suite beans_zero_esc50_official --predict-url http://localhost:8000/predict --limit 5
```

---

## Launcher-by-launcher multi-audio capability matrix (current + minimal change)

### Summary matrix (decision = implement/skip)

| Launcher | Upstream supports multi-audio? | Current multi-audio per item? | Maps N payloads → N `<AudioHere>`? | Minimal change for Option A (`payloads`) | Effort | Decision |
|---|---|---:|---:|---|---:|---|
| `openai_compatible_proxy` | **Yes-ish** (OpenAI-compat multimodal content parts) | **Yes** (multiple `audio_inputs`) | **Yes** (placeholder replacement; fallback append) | Accept `payloads` and convert to the same internal `audio_inputs` / audio content parts | Low | **implement** |
| `vllm` adapter (OpenAI-compat upstream) | **Maybe** (depends on upstream model + OpenAI-compat variant) | **Yes** (multiple `audio_inputs`) | **No (today)** (does not replace placeholders; always attaches to last user) | Accept `payloads`; optionally add placeholder-aware injection to align with `<AudioHere>` | Medium | **implement** |
| `af3` (Audio Flamingo Next) | **Yes** (conversation supports multiple audio items) | **Yes** (multiple `audio_inputs`) | **Yes** (splits on placeholder, inserts audio paths, appends extras) | Accept `payloads` and resolve to audio paths list (same flow as current `audio_inputs`) | Low | **implement** |
| `naturelm-v1.0` | **No** (effective single-audio input per request) | **Stub: yes** / **Real: no** (uses `audio_inputs[0]`) | **No** (real inference ignores >1) | Either hard-reject multi-audio (len!=1) or implement a lossy “mix/concat to one waveform” policy | High | **skip/document** |
| `naturelm-v1.1` | **No** (explicit single-audio enforcement) | **Stub: yes** / **Real: no** (rejects len!=1) | **No** (real inference returns error on >1) | Same as v1.0; would require upstream/model redesign or lossy merge | High | **skip/document** |
| `dummy` | N/A | **Yes** (accepts list; hashes metadata) | N/A | Accept `payloads` and fold into existing deterministic stub metadata | Low | **implement** |

---

## Per-launcher notes (required details)

### `openai_compatible_proxy` (`examples/servers/openai_compatible_proxy/serve.py`)

- **Current support for multiple audio payloads per item**
  - Accepts `audio_inputs: list[HttpAudioInput]` and builds one OpenAI-compat “audio content part” per entry.
  - In proxy (real) mode it currently supports **`base64_wav` only**; other payload types error.
- **How it maps `<AudioHere>` today**
  - Counts occurrences of **`<Audio><AudioHere></Audio>`** across message contents.
  - If placeholders exist, it **replaces each placeholder** sequentially with an audio content-part.
  - If no placeholders exist, it **attaches all audio parts** to the last user message content (Gemini upstream prepends; non-Gemini appends).
- **Minimal change to support Option A `payloads`**
  - Extend request schema for this launcher to accept `payloads` (list of `{payload_type, payload}`) in addition to existing `audio_inputs` (or replace `audio_inputs` with a compatibility layer).
  - Convert `payloads` into the same internal list used today (conceptually identical to `audio_inputs`) and reuse existing placeholder-count validation + `_inject_audio_items`.
- **Hard limitations (upstream interface)**
  - OpenAI-compatible upstream support varies:
    - Some upstreams accept multiple `input_audio` parts; others accept only one or apply strict size/time limits.
  - This launcher also enforces `base64_wav` in proxy mode, so multi-audio via file paths/URLs would require additional server-side fetch/decode logic (or runner-side base64 encode).

### `vllm` adapter (`examples/servers/vllm/adapter.py`)

- **Current support for multiple audio payloads per item**
  - Accepts `audio_inputs: list[HttpAudioInput]` and converts each to an audio content part (`input_audio` or `audio_url` via env `VLLM_AUDIO_CONTENT_FORMAT`).
- **How it maps `<AudioHere>` today**
  - It **does not** parse or replace `<Audio><AudioHere></Audio>` within message text.
  - Instead, it attaches audio parts to the last user message content (and **prepends audio before text**), regardless of placeholder positions.
- **Minimal change to support Option A `payloads`**
  - Accept `payloads` and convert each entry into an audio content part as it does for `audio_inputs`.
  - If strict `<AudioHere>` alignment is required for MultiAudio prompts, add placeholder-aware injection similar to the OpenAI proxy (split message string by placeholder and splice audio parts at those positions).
- **Hard limitations (upstream interface)**
  - Multi-audio support depends on:
    - vLLM OpenAI-compat support for multimodal content for the chosen model,
    - the model’s own instruction-tuning expectations (some may implicitly assume one audio).
  - Adapter already allows modality config (`VLLM_OUTPUT_MODALITIES`) and may be constrained by upstream’s validation of content parts.

### `af3` (`examples/servers/af3/serve.py`)

- **Current support for multiple audio payloads per item**
  - Accepts `audio_inputs: list[HttpAudioInput]`; in real mode it resolves each entry to a local WAV file path (decoding base64/download URL into a temp dir).
- **How it maps `<AudioHere>` today**
  - Splits each message content on **`<Audio><AudioHere></Audio>`** and inserts `{"type": "audio", "path": ...}` content items sequentially.
  - If there are **extra audio inputs** beyond placeholders, it appends them to a target user message content.
- **Minimal change to support Option A `payloads`**
  - Add `payloads` support and feed it through the same resolution path used by `_resolve_audio_path`, producing `audio_paths`.
  - Reuse `_build_conversation` for placeholder placement; keep current fallback of appending extras.
- **Hard limitations (upstream interface)**
  - AF-Next uses audio file paths in its processor chat template; this is compatible with multiple audio segments as long as the processor/model supports multiple audio items in the conversation.
  - Practical limitation is likely request size / GPU memory rather than strict API constraints.

### `naturelm-v1.0` (`examples/servers/naturelm-v1.0/serve.py`)

- **Current support for multiple audio payloads per item**
  - Schema accepts `audio_inputs: list[...]` and validates placeholder count vs list length.
  - **Real inference uses only the first audio** (`audio_slot = item.audio_inputs[0]`) and is documented as “one audio per sample”.
- **How it maps `<AudioHere>` today**
  - Only validates counts (if placeholders exist) but does not map multiple audio inputs into multiple placeholder positions during real inference.
- **Minimal change to support Option A `payloads`**
  - If upstream remains single-audio, the minimal safe behavior is to **reject** requests with `len(payloads) != 1` (or `len(audio_inputs) != 1`) in real mode.
  - A more invasive alternative is to define a runner/launcher policy to **merge** multiple audios into one waveform (concat with silence, or mix), but that is lossy and must be explicitly decided + tested.
- **Hard limitations (upstream interface)**
  - NatureLM v1.0 pipeline call is effectively \(1 audio, 1 query\) per request item.
  - Multi-audio “native” support would require upstream/model changes, not just schema wiring.

### `naturelm-v1.1` (`examples/servers/naturelm-v1.1/serve.py`)

- **Current support for multiple audio payloads per item**
  - Request schema accepts `audio_inputs: list[...]`.
  - In real inference, it explicitly enforces **exactly 1** audio input (`if len(decoded_audios) != 1: return error`).
- **How it maps `<AudioHere>` today**
  - Treats multiple audios as “multiple chunks”, but immediately rejects >1; no mapping is performed in real mode.
- **Minimal change to support Option A `payloads`**
  - Same as v1.0: either keep hard single-audio limitation (reject multi-audio) or introduce a defined merge policy (concat/mix) with clear evaluation implications.
- **Hard limitations (upstream interface)**
  - The esp-research NatureLM generate call takes a single audio tensor; supporting multiple audios per item is upstream-breaking.

### `dummy` (`examples/servers/dummy/serve.py`)

- **Current support for multiple audio payloads per item**
  - Accepts `audio_inputs: list[...]` and includes the entire list metadata in its deterministic hash; no placeholder logic.
- **How it maps `<AudioHere>` today**
  - It does not interpret `<AudioHere>`; it is a pure contract conformance stub.
- **Minimal change to support Option A `payloads`**
  - Accept `payloads` and incorporate it into the same deterministic key used for hashing (or map `payloads` → `audio_inputs` internally).
- **Hard limitations (upstream interface)**
  - None (stub only).

---

## Recommendation for I16-C through I16-F implementation

- **Adopt Option A** (`payloads` optional) as the schema extension.
- **Implement multi-audio mapping** for:
  - `openai_compatible_proxy` (placeholder-aware mapping already exists; reuse it)
  - `af3` (placeholder-aware mapping already exists; reuse it)
  - `vllm` adapter (recommended to add placeholder-aware mapping for correctness parity)
  - `dummy` (keep deterministic conformance coverage)
- **Document/skip** NatureLM v1.0 and v1.1 for multi-audio until upstream supports it or a formally approved merge policy exists.

