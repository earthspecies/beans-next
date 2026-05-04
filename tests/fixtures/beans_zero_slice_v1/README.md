# beans_zero_slice_v1 (synthetic)

This directory is a tiny, deterministic fixture bundle intended for later
golden-run regression tests. It is CPU-only and contains no HuggingFace
downloads and no model outputs.

Regenerate with:

```bash
uv run python scripts/fixtures/generate_beans_zero_slice.py --out tests/fixtures/beans_zero_slice_v1 --force
```

