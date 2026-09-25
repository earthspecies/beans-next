# NatureLM v1 adapter

This adapter serves NatureLM-audio v1 through the benchmark HTTP contract.
It accepts exactly one audio clip per request.

Install the compatible NatureLM runtime in the adapter environment before real inference.
Set `NATURELM_CFG_PATH` to the runtime's `configs/inference.yml` file.
Use `NATURELM_V1_0_MODEL` and `NATURELM_V1_0_MODEL_REVISION` to select the checkpoint.
Set `NATURELM_V1_0_LLAMA_PATH` when using a local Llama checkpoint.

```bash
uv sync --group real
NATURELM_CFG_PATH=/path/to/NatureLM/configs/inference.yml \
  NATURELM_V1_0_STUB=0 PORT=8000 ./serve.sh
```

For a CPU contract check, use `NATURELM_V1_0_STUB=1`.
Stub outputs are not model predictions.
