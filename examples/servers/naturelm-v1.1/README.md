# NatureLM v1.1 adapter

See [model serving](../../../docs/model_servers.md#naturelm-v11) for runtime, checkpoint, and environment requirements.

```bash
uv sync
NATURELM_STUB_MODE=1 ./serve.sh
```

The command uses stub mode for contract checks. Real inference requires a compatible runtime, checkpoint, and GPU dependencies.
