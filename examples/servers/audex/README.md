## Tier-2 reference launcher: `audex`

Serves `nvidia/Nemotron-Labs-Audex-30B-A3B` (the 30B MoE Audex checkpoint,
not the 2B variant) via the BEANS-Next HTTP contract (`predictions_v1`).

**License**: `nvidia/Nemotron-Labs-Audex-30B-A3B` is released under the NVIDIA
OneWay Noncommercial License. Only non-commercial research use is permitted.

Upstream references agents must check before changing this launcher:

- Hugging Face model card: `https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B`
- Vendor inference scripts ship inside the model repo at
  `inference_scripts_vllm/audioqa_scripts/` (package `audex_30b_a3b_vllm`).

### Serving notes

- **Not gated**, 71.7 GB repo; `checkpoint_folder_full` alone is 65.3 GB.
- **16 kHz** target sample rate — same as AF3/NatureLM, no prompt YAML changes needed.
- Audio is chunked into 30 s non-overlapping windows, last one zero-padded.
  Every BEANS-Next clip used here is ≤ 30.0 s, so exactly one window per sample.
- **Plain `vllm serve` will not find the architecture.** Serving requires the
  vendor vLLM plugin editable-installed into the serving venv:
  ```bash
  pip install -e <model_snapshot>/inference_scripts_vllm/audioqa_scripts --no-deps
  ```
  The plugin targets vLLM 0.20.0; this launcher's own dependency pins vLLM
  0.20.2 (see `requirements.txt`) as the closest validated match. Give Audex
  its own venv — do not install the plugin into the shared text-eval vLLM venv.
- Vendor serve flags:
  ```bash
  vllm serve nvidia/Nemotron-Labs-Audex-30B-A3B \
    --served-model-name audex_audio --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 --max-model-len 32768 --trust-remote-code \
    --limit-mm-per-prompt '{"audio": 1}' --allowed-local-media-path <root>
  ```
  (`--limit-mm-per-prompt '{"audio": 1}'` means this launcher only ever
  supports single-audio tasks — tier 4 / multi-audio is out of scope.)
- Audio mode wants the non-thinking chat template
  (`chat_template_kwargs={"enable_thinking": false}`), and the vendor client
  suppresses audio-codec token ids (`29`, `30`, `31`) from generated output.
  This launcher suppresses the same ids via `logit_bias` (set through
  `AUDEX_EXTRA_BODY_JSON`, cheaper than the vendor's full-vocab
  `allowed_token_ids`) and additionally strips any `<think>...</think>` span
  or stray codec/placeholder token that leaks through, so the shared
  BEANS-Next postprocess pipeline sees clean text regardless.
- Vendor-recommended sampling for audio understanding: `temperature=0.7,
  top_p=0.9`. Greedy decoding (`temperature=0.0`) is the repo default. Both
  are driven purely through `AUDEX_EXTRA_BODY_JSON` — no code change needed
  to switch between the two pilot arms.

### Usage

```bash
# 1. Start the model (separate terminal, GPU required, plugin venv active):
vllm serve nvidia/Nemotron-Labs-Audex-30B-A3B \
  --served-model-name audex_audio --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --trust-remote-code \
  --limit-mm-per-prompt '{"audio": 1}' --allowed-local-media-path <root> \
  --host 127.0.0.1 --port 8001

# 2. Start this adapter sidecar:
AUDEX_ADAPTER_STUB=0 \
  AUDEX_UPSTREAM_BASE_URL=http://127.0.0.1:8001 \
  AUDEX_MODEL_ID=audex_audio \
  AUDEX_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false},"temperature":0.0}' \
  PORT=8000 ./serve.sh
```

See `./serve.sh --help` for the full environment variable reference and
`beans_next/registry/model/audex_vllm_local_8000.yaml` for the BEANS-Next
model preset that points at this launcher.
