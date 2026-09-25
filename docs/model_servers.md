# Model servers

Adapters implement `GET /info`, `GET /health`, and `POST /predict`.
See the [HTTP contract](http_contract.md) for request and response formats.
Install model dependencies in each adapter's own environment.

| Adapter | Use |
|---|---|
| `examples/servers/af3` | Audio Flamingo Next |
| `examples/servers/naturelm-v1.0` | NatureLM-audio v1 |
| `examples/servers/naturelm-v1.1` | Compatible NatureLM runtime and base or fine-tuned checkpoints |
| `examples/servers/vllm` | Qwen3-Omni and supported text-only models |
| `examples/servers/dummy` | Contract checks without weights |

Tier 4 requires a multi-audio server. NatureLM v1 rejects multiple clips instead of silently dropping them.

Check a running server:

```bash
uv run python scripts/check_launcher.py http://localhost:8000
```

Stub modes validate HTTP behavior only. They do not produce benchmark model results.

## NatureLM v1.1

Install the compatible `naturelm` runtime, or set `NATURELM_RUNTIME_PATH` to its directory.
It must provide `NatureLM.from_pretrained` and `GenerationConfig` for the checkpoint's architecture.
The model runtime source is not bundled in this evaluation repository.

Set `NATURELM_HF_REPO_ID` and `NATURELM_HF_REVISION` for an HF checkpoint.
Alternatively, set `NATURELM_LOCAL_CHECKPOINT_DIR` to a local checkpoint copy.
Set `NATURELM_ENABLE_INFERENCE=1` for real inference, then start `serve.sh` from the adapter directory.

`NATURELM_STUB_MODE=1` provides a contract-only server without a runtime or weights.
Real inference requires a compatible runtime and checkpoints.
