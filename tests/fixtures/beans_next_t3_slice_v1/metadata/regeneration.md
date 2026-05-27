# Regeneration

Phase A (CPU-only):

```bash
uv run python scripts/fixtures/generate_beans_next_t3_slice.py --out tests/fixtures/beans_next_t3_slice_v1 --force
```

Phase B (GPU): populate `expected/*` by running a real launcher and
capturing outputs.

