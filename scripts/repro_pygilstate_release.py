"""Reproduce interpreter-finalization crash (PyGILState_Release).

This script is intentionally minimal and should be run WITHOUT any "hard-exit"
workarounds (no os._exit). It is used to validate whether the current native
dependency set (datasets/pyarrow/pandas/numpy) is safe on interpreter shutdown.
"""

from __future__ import annotations

import time


def main() -> int:
    # Import order here matters for some native teardown bugs.
    import datasets  # noqa: F401

    # Streaming load + early termination tends to reproduce threadpool shutdown bugs.
    ds = datasets.load_dataset(
        "EarthSpeciesProject/BEANS-Zero",
        "BEANS-Zero",
        split="test",
        streaming=True,
    )

    # Touch a couple rows, then stop early.
    it = iter(ds)
    for _ in range(3):
        row = next(it)
        # Force materialization of some fields.
        _ = row.get("id")
        _ = row.get("dataset_name")
        _ = len(row.get("audio") or [])

    # A small sleep sometimes changes timing; keep but short.
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
