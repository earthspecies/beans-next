"""Batch size helpers for ``predictions_v1`` requests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

__all__ = ["DEFAULT_MAX_BATCH_FALLBACK", "effective_max_batch_size", "iter_batches"]

DEFAULT_MAX_BATCH_FALLBACK: int = 8
"""Conservative default when ``/info`` omits ``max_batch_size`` or it is invalid."""

T = TypeVar("T")


def effective_max_batch_size(
    server_info: dict[str, Any] | None,
    *,
    fallback: int = DEFAULT_MAX_BATCH_FALLBACK,
    hard_cap: int = 512,
) -> int:
    """Return the maximum in-flight batch size for ``POST /predict``.

    Honors ``max_batch_size`` from launcher ``/info`` when it is a positive
    integer. If ``supports_batching`` is false, returns ``1``. Caps extremely
    large advertised values at ``hard_cap`` for client-side safety.

    Parameters
    ----------
    server_info
        Parsed ``GET /info`` object (for example ``HttpClient.server_info``), or
        ``None`` when unknown.
    fallback
        Value used when ``max_batch_size`` is missing or not an ``int`` ``>= 1``.
    hard_cap
        Upper bound on returned batch size.

    Returns
    -------
    int
        Batch size in ``[1, hard_cap]``.

    Raises
    ------
    ValueError
        If ``fallback`` or ``hard_cap`` is not a positive integer.
    """
    if fallback < 1:
        msg = "`fallback` must be >= 1"
        raise ValueError(msg)
    if hard_cap < 1:
        msg = "`hard_cap` must be >= 1"
        raise ValueError(msg)
    if server_info is None:
        return min(fallback, hard_cap)
    supports = server_info.get("supports_batching", True)
    if supports is False:
        return 1
    raw = server_info.get("max_batch_size")
    if isinstance(raw, int) and raw >= 1:
        return min(raw, hard_cap)
    return min(fallback, hard_cap)


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield contiguous slices of ``items`` with length at most ``batch_size``.

    Parameters
    ----------
    items
        Ordered sequence to chunk (caller should pre-sort for determinism).
    batch_size
        Maximum length of each yielded slice; must be ``>= 1``.

    Yields
    ------
    Sequence[T]
        Non-empty slices covering ``items`` in order.

    Raises
    ------
    ValueError
        If ``batch_size`` is less than ``1``.
    """
    if batch_size < 1:
        msg = "`batch_size` must be >= 1"
        raise ValueError(msg)
    n = len(items)
    for i in range(0, n, batch_size):
        yield items[i : i + batch_size]
