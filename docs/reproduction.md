# Reproduction

Use the canonical suites in the README.
Some task IDs retain date prefixes so existing results and Gaussian-noise comparisons remain compatible.
Their `subset` fields contain HF task names. No dated backend configuration is required.

Record the checkpoint, model runtime revision, dataset commit, input mode, seed, and code revision for each run.
Keep errors in the result artifacts. A short run cannot substitute for a complete paper result.

## Dataset release requirements

The HF metadata at commit `2fc58150c9541698ffc82aaf1f5d5a44993c54bf` was inspected during cleanup.

| Tier-4 task | Examples |
|---|---:|
| `gibbon-fewshot-detection-balanced` | 868 |
| `giant-otter-4way` | 500 |
| `dcase-fewshot-detection-balanced` | 3158 |
| `crow-4way` | 200 |
| `unseen-species-4way` | 218 |

The paper suite uses these five tasks. It excludes zebra and the unpublished hard unseen variant.
The HF inventory supports this selection. It does not establish which exact data revision produced every paper result.

That HF commit lacks `insect-presence` and `begging-call-presence`, which the paper requires.
Both tasks remain in tier 2. Use a completed HF release before a full reproduction run.
Do not silently drop missing tasks or substitute an older task version.

HF repository identifiers and the final dataset structure await a separate update.
NatureLM v1.1 also needs a compatible model runtime and the corresponding checkpoints for real inference.

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

Keep task IDs and sample IDs aligned across audio, text-only, and Gaussian-noise runs.
The Gaussian-noise scripts remain available for ongoing experiments.

## Anonymous distribution

Distribute a clean archive or fresh repository without the original Git history.
Exclude caches, model weights, generated results, local environments, and local configuration.
Keep required third-party license notices.
After the HF update, review repository IDs, model-runtime sources, and result metadata before distribution.
