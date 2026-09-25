# Audio Flamingo Next adapter

The adapter uses the HF model `nvidia/audio-flamingo-next-hf`.
It supports single-audio and multi-audio requests through the benchmark HTTP contract.

Install the adapter dependencies:

```bash
uv sync --group gpu
```

Start real inference on a GPU:

```bash
AF3_MODEL=nvidia/audio-flamingo-next-hf PORT=8000 ./serve.sh
```

Set `AF3_MODEL_REVISION` to an exact model revision for reproducibility.
For a CPU contract check, use `AF3_STUB=1`.

The model runtime supports interleaved audio and text.
The adapter preserves clip order and gives each clip its own conversation turn.
It also validates audio counts and removes empty trailing generation windows.

Model use is subject to the upstream model license.
