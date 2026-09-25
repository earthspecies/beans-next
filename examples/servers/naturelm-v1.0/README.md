# NatureLM v1 adapter

This adapter serves NatureLM-audio v1 through the benchmark HTTP contract.
It supports single-audio requests. Multiple clips cause an explicit error.

Install the compatible NatureLM runtime in the adapter environment before real inference.
Use the checkpoint identifier in `NATURELM_V1_0_MODEL`.

```bash
uv sync --group real
NATURELM_V1_0_STUB=0 PORT=8000 ./serve.sh
```

For a CPU contract check, use `NATURELM_V1_0_STUB=1`.
Stub outputs are not model predictions.
