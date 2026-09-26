# Reproduction

Install the evaluator and start a [model server](model_servers.md).
Select a dataset commit with `--hf-revision` and pin the model checkpoint in the server configuration.
Keep the code revision, runtime version, input mode, and seed with each run.
For the paper's BEANS-Next evaluation, use dataset revision
`f93a692878a898bb22b876064f0ddfa79f8759ee`. Its five Tier 4 paper tasks
contain 200 examples each, with the evaluated Gibbon support order and answers.

## Run a benchmark suite

```bash
uv run beans-next run \
  --suite beans_next_tier1 \
  --hf-revision f93a692878a898bb22b876064f0ddfa79f8759ee \
  --predict-url http://localhost:8000/predict \
  --output-dir results/model/audio/tier1
```

Repeat with `beans_next_tier2`, `beans_next_tier3`, and `beans_next_tier4`.
Tier 4 requires a server that accepts multiple audio clips per example.
Use `beans_zero_core` and `birdset_core` for the comparison datasets.
Each HF repository has its own revision.

Task definitions select prompts, audio duration limits, and scoring metrics.
Run full splits for result tables. Use `--limit` for a short installation check.

## Input ablations

Repeat a suite with `--modality-mode text-only`, `text-only-informed`, or `gaussian-noise`.
Use a separate output directory for each model, suite, and input mode.
Keep dataset revisions and sample IDs consistent across compared runs.

For Gaussian noise, `--gaussian-noise-cache-dir` lets runs share generated waveforms.
Each run records noise parameters and audio identities in its manifest.
See the [evaluation guide](evaluation.md) for defaults, caching, and resume options.

## Result tables

The scripts accept local result directories. Inspect their arguments with `--help`:

```bash
uv run python scripts/build_core_suite_results_tables.py --help
uv run python scripts/build_t3_results_table.py --help
uv run python scripts/build_text_only_paper_rows.py --help
uv run python scripts/build_gaussian_appendix_tables.py --help
uv run python scripts/analyze_gaussian_matched.py --help
uv run python scripts/validate_gaussian_noise_manifest.py --help
```

Check sample counts and errors in each task's `summary.json` before comparing results.
Captioning uses corpus CIDEr; score the full task together.
