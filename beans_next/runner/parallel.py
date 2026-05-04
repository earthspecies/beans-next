"""Parallel helpers for runner CPU-side work."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

__all__ = ["map_ordered"]

T = TypeVar("T")
R = TypeVar("R")


def map_ordered(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int,
    executor: ThreadPoolExecutor | None = None,
) -> list[R]:
    """Apply a function concurrently, returning results in the input order.

    Parameters
    ----------
    fn
        Per-item function to execute.
    items
        Items to process.
    workers
        Maximum number of worker threads. Must be at least 1.
    executor
        Optional pre-created :class:`~concurrent.futures.ThreadPoolExecutor`.
        When provided, it is used directly (no new pool is created), which
        avoids per-batch pool construction overhead. When ``None`` and
        ``workers > 1``, a temporary pool is created for this call.

    Returns
    -------
    list
        Results aligned with `items` order.

    Raises
    ------
    ValueError
        If `workers` is less than 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers == 1 or len(items) <= 1:
        return [fn(item) for item in items]

    if executor is not None:
        futures = [executor.submit(fn, item) for item in items]
        return [f.result() for f in futures]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        return [f.result() for f in futures]
