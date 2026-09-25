# Evaluation

## Data

The evaluator uses Hugging Face datasets only. BEANS-Next uses metadata rows with ordered audio paths.
BEANS-Zero and BirdSet have separate HF loaders because their schemas differ.

For BEANS-Next, set `BEANS_NEXT_HF_BEANS_NEXT_ROOT` to a local HF snapshot directory.
The directory can contain `test/metadata.parquet` and its audio files, or a supported older HF layout.
Missing files cause an error. The local loader does not fall back to remote downloads.

For BEANS-Next, revision precedence is: `--hf-revision`, `BEANS_NEXT_HF_REVISION`, task revision, then `main`.
Use a commit SHA for reproducible runs.

## Tasks and metrics

Use `uv run beans-next list --kind suite` to list suites.
Use `uv run beans-next describe eval_task <task-id>` to inspect a task.

Task definitions set the prompt, task type, audio cap, and metric.
Tiers 1 and 2 use a 10-second cap. Tier 3 uses a 30-second cap.
Tier 4 uses a 10-second cap for each clip, with references before the query.

| Task type | Reported metric |
|---|---|
| Classification and multiple choice | Accuracy, with macro-F1 where specified |
| Numeric prediction | Mean absolute error in the target units |
| Captioning | Corpus CIDEr |
| Species listing or summary | Species F1 |
| Species frequency ranges | Species-band IoU, including missed reference species |

Captioning needs the full reference corpus for its IDF calculation.
A one-example smoke run does not produce a meaningful CIDEr score.
The paper displays CIDEr multiplied by 100. Evaluation stores its normalized value.
There is no LLM judge or judge-assisted extraction step.

## Input modes

- `audio`: use the original audio.
- `text-only`: remove audio slots and audio placeholders.
- `text-only-informed`: also tell the model that audio is unavailable.
- `gaussian-noise`: replace every audio slot with deterministic Gaussian noise.

Text-only modes do not download audio.
Gaussian noise preserves audio duration and slot order, with default RMS -20 dBFS and global seed zero.
Use `--gaussian-noise-cache-dir` for a shared cache.
The default is `~/.cache/beans-next/gaussian-noise`.
Each run writes a noise manifest for protocol checks and matched comparisons.

## Results and resume

Each task writes predictions, processed predictions, scored predictions, a summary, and model identity.
Use `--resume` with the same output directory to continue a run.
Use `--cache-dir` for persistent inference and scoring caches.

Stored dataset IDs remain unchanged. Rows without an ID use a deterministic `beans_next:hf:` fallback ID.
Caches and exclusions that used the old backend's fallback IDs need regeneration.
Published rows with explicit IDs are unaffected.

Rescore an existing prediction file on CPU:

```bash
uv run beans-next score-from-file results/task/predictions.jsonl \
  --task-type classification --output-dir results/rescored
```

Keep the original sibling processed-prediction file with its targets. The rescorer reads those targets before scoring.
