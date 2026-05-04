"""Shared helpers and a minimal scorer registry for deterministic metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

__all__ = [
    "MetricsError",
    "get_scorer",
    "list_scorers",
    "register_scorer",
    "validate_equal_length",
]

T = TypeVar("T")

_SCORERS: dict[str, Callable[..., float]] = {}


class MetricsError(ValueError):
    """Raised when metric inputs are invalid or inconsistent."""


def validate_equal_length(
    predictions: Sequence[T],
    targets: Sequence[Any],
    *,
    name_predictions: str = "predictions",
    name_targets: str = "targets",
) -> None:
    """Ensure two sequences have the same length.

    Parameters
    ----------
    predictions : sequence
        Model outputs (one per example).
    targets : sequence
        Ground truth (one per example).
    name_predictions : str, optional
        Label used in error messages for ``predictions``.
    name_targets : str, optional
        Label used in error messages for ``targets``.

    Raises
    ------
    MetricsError
        If either sequence is empty or lengths differ.
    """
    if predictions is None or len(predictions) == 0:
        raise MetricsError(f"{name_predictions} must be a non-empty sequence.")
    if targets is None or len(targets) == 0:
        raise MetricsError(f"{name_targets} must be a non-empty sequence.")
    if len(predictions) != len(targets):
        raise MetricsError(
            f"{name_predictions} and {name_targets} must have the same length "
            f"({len(predictions)} vs {len(targets)}).",
        )


def register_scorer(func: Callable[..., float]) -> Callable[..., float]:
    """Register ``func`` under ``func.__name__`` for ``get_scorer``.

    Parameters
    ----------
    func : callable
        Metric function returning a scalar ``float``.

    Returns
    -------
    callable
        The same function (decorator pattern).

    Raises
    ------
    MetricsError
        If a scorer with the same name is already registered.
    """
    key = func.__name__
    if key in _SCORERS:
        raise MetricsError(f"Scorer {key!r} is already registered.")
    _SCORERS[key] = func
    return func


def get_scorer(score_name: str) -> Callable[..., float]:
    """Return a registered metric callable by name.

    Parameters
    ----------
    score_name : str
        Registered function name (for example ``\"accuracy\"``).

    Returns
    -------
    callable
        The registered metric.

    Raises
    ------
    LookupError
        If ``score_name`` is unknown.
    """
    scorer = _SCORERS.get(score_name)
    if scorer is None:
        raise LookupError(f"Scorer {score_name!r} not found.")
    return scorer


def list_scorers() -> list[str]:
    """List registered scorer names.

    Returns
    -------
    list of str
        Sorted scorer names.
    """
    return sorted(_SCORERS.keys())
