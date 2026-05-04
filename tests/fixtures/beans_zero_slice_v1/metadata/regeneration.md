# Regeneration

Phase A (CPU-only):

```bash
uv run python scripts/fixtures/generate_beans_zero_slice.py --out tests/fixtures/beans_zero_slice_v1 --force
```

Phase B (GPU): populate `expected/*` by running a real NatureLM launcher and capturing outputs.

