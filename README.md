# BEANS-Next evaluation

Evaluate audio-language models on the benchmark tasks and comparison datasets in the paper.
All evaluation data comes from Hugging Face. Local HF snapshots also work.
Model servers run in separate environments and communicate with the evaluator through HTTP.

## Install

Use Python 3.11 or 3.12 and `uv`.

```bash
uv sync --locked --group dev
uv run beans-next --help
```

## Evaluate

Start a [model server](docs/model_servers.md), then run a suite:

```bash
uv run beans-next run \
  --suite beans_next_tier1 \
  --predict-url http://localhost:8000/predict \
  --output-dir results/tier1 \
  --limit 10
```

Remove `--limit` for full task splits. The limit applies to each task.
Use `--hf-revision <commit>` to select an exact dataset revision.

| Suite | Scope |
|---|---|
| `beans_next_tier1` | Acoustic perception and description |
| `beans_next_tier2` | Semantic understanding |
| `beans_next_tier3` | Structural understanding |
| `beans_next_tier4` | In-context evaluation with ordered reference and query audio |
| `beans_next_all_tiers` | All four tiers |
| `beans_zero_core` | BEANS-Zero comparison tasks |
| `birdset_core` | BirdSet comparison tasks |
| `beans_zero_smoke` | Small suite for installation checks |

Set `--modality-mode` to `text-only`, `text-only-informed`, or `gaussian-noise` for input ablations.
The default is `audio`.

See the [evaluation guide](docs/evaluation.md) for metrics, ablations, local snapshots, and result files.
The [reproduction guide](docs/reproduction.md) records dataset requirements and table-generation commands.

## Tests

```bash
UV_NO_SYNC=1 uv run pytest -q
```

Tests cover metrics, HF snapshots, audio ordering, ablations, resume behavior, and model-server contracts.
Server contract tests use local HTTP ports. They do not require GPUs or model weights.

## Layout

- `beans_next/`: data loading, prompts, HTTP client, evaluation, scoring, and registries.
- `configs/`: a portable evaluation configuration.
- `examples/servers/`: model adapters and a dummy server.
- `scripts/`: server checks, result tables, and Gaussian-noise analysis.
- `tests/`: local fixtures and regression tests.

HF repository identifiers remain unchanged pending the dataset release update.
