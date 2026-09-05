## Tier-2 reference launcher: `gemma4`

Serves `google/gemma-4-12B-it` (the 11.95B-parameter dense, encoder-free
Gemma 4 Unified checkpoint) via the BEANS-Next HTTP contract
(`predictions_v1`).

**License**: `google/gemma-4-12B-it` is released under the Apache 2.0
license. Not gated, no non-commercial restriction.

Upstream references agents must check before changing this launcher:

- Hugging Face model card: `https://huggingface.co/google/gemma-4-12B-it`
- Audio capability docs: `https://ai.google.dev/gemma/docs/capabilities/audio`
- vLLM recipe: `https://recipes.vllm.ai/Google/gemma-4-12B-it`

### Serving notes

- **Not gated**, ~24 GB bf16 safetensors. Encoder-free: there is no audio
  tower; raw 16 kHz audio is sliced into 40 ms frames and linearly projected
  into the LM embedding space (25 audio tokens/second).
- **16 kHz mono float32, normalised to [-1, 1]** is the native input format.
- **Hard 30 s audio limit** (Gemma's own capability docs). Enforced
  launcher-side as `GEMMA4_MAX_AUDIO_SECONDS` (default 30), applied as
  `min(item.generation_config.max_length_seconds or infinity,
  GEMMA4_MAX_AUDIO_SECONDS)` — see `_effective_cap_seconds` in `serve.py`.
  Unlike the Audex launcher, this cap always applies, even when the caller
  sends no `max_length_seconds` at all (e.g. BEANS-Next tiers 1-3, which
  have no per-subset protocol cap of their own).
- **Audio goes after the text prompt** by default
  (`GEMMA4_AUDIO_POSITION=after_text`), the opposite of the Audex/generic
  vLLM-adapter convention (`examples/servers/vllm/adapter.py` puts audio
  first, per Qwen3-Omni's guidance). Switch to
  `GEMMA4_AUDIO_POSITION=before_text` for the pilot's audio-position A/B
  test; both are driven purely by this env var, no code change needed.
- **No codec-token suppression.** The Audex launcher suppresses/strips
  Audex-specific audio-codec/placeholder token ids (29/30/31) because the
  vendor Audex model has an audio-generation path that can emit them. Gemma
  4 12B has no such path, so this launcher only strips a leaked
  `<think>...</think>` span and nothing else.
- Gemma 4 has built-in thinking, off unless `<|think|>` is in the system
  prompt or `enable_thinking` is set. Disable it explicitly via
  `GEMMA4_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'`
  (same mechanism the Audex launcher and the repo's judge-model setup use).
  This launcher still strips any leaked `<think>...</think>` span as a
  belt-and-braces fallback.
- Vendor sampling for open-ended tasks: `temperature=1.0, top_p=0.95,
  top_k=64`. Greedy decoding (`temperature=0.0`) is the repo default and the
  reported pilot arm. Both are driven purely through `GEMMA4_EXTRA_BODY_JSON`
  — no code change needed to switch between the two pilot arms.

### Backends (`GEMMA4_BACKEND`)

- **`vllm` (default).** Proxies to a local OpenAI-compatible
  `vllm serve google/gemma-4-12B-it` upstream, exactly like the Audex
  launcher's proxy path. Needs **vLLM >= 0.23.0** with the `audio` extra
  (`vllm[audio]`) for encoder-free Gemma 4 Unified support:
  ```bash
  vllm serve google/gemma-4-12B-it \
    --served-model-name gemma4_12b --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
    --reasoning-parser gemma4 --allowed-local-media-path <root>
  ```
  (`--limit-mm-per-prompt '{"audio": 1}'` means this launcher only ever
  supports single-audio tasks — tier 4 / multi-audio is out of scope, same
  boundary as Audex.)
- **`transformers` (costed fallback, Decision 1).** Loads
  `Gemma4UnifiedForConditionalGeneration` + `AutoProcessor` in-process
  (bf16, one GPU), batches `/predict` items (serialized on the GPU — no
  thread-pool concurrency for this backend), and feeds audio as 16 kHz mono
  float32 in [-1, 1], resampling defensively (via `librosa`, even though
  `esp_data` supplies `audio_path_16KHz`) and asserting the final rate
  rather than trusting the declared one.

  **This path has not been exercised against real GPU hardware or model
  weights in this change and needs on-cluster validation** before it is
  trusted for a full run: confirm `AutoProcessor.from_pretrained(...)`
  actually exposes a `feature_extractor` for this snapshot (its absence is
  the documented symptom of audio being silently ignored), confirm
  `apply_chat_template(..., audio=..., sampling_rate=16000, ...)` is the
  correct call shape for whatever `transformers` release ships Gemma 4
  Unified support, and confirm generation throughput is acceptable given
  the lack of continuous batching (expect roughly a 3-5x throughput loss
  vs. the vLLM path per the plan).

### `/info`

Advertises `predictions_v1`, `file_path` among `audio_payload_types`,
`max_audio_seconds: 30.0`, and the active `backend` (`vllm` or
`transformers`), so the run manifest can capture which path produced a given
results column.

### Usage

```bash
# 1. Start the model (separate terminal, GPU required, vllm[audio]>=0.23.0):
vllm serve google/gemma-4-12B-it \
  --served-model-name gemma4_12b --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --trust-remote-code \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
  --reasoning-parser gemma4 --allowed-local-media-path <root> \
  --host 127.0.0.1 --port 8001

# 2. Start this adapter sidecar:
GEMMA4_ADAPTER_STUB=0 GEMMA4_BACKEND=vllm \
  GEMMA4_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  GEMMA4_MODEL_ID=gemma4_12b \
  GEMMA4_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false},"temperature":0.0}' \
  PORT=8000 ./serve.sh
```

See `./serve.sh --help` for the full environment variable reference and
`beans_next/registry/model/gemma4_12b_local_8000.yaml` for the BEANS-Next
model preset that points at this launcher.
