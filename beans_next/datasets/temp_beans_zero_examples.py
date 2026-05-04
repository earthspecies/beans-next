"""Temporary compatibility shim for removed dataset adapters.

This module previously exposed helpers for a private dataset backend. That
backend integration is not part of BEANS-Next and has been removed.
"""

from __future__ import annotations

from collections.abc import Iterator

from beans_next.api.types import DatasetExample

__all__ = ["iter_temp_beans_zero_esc50_examples"]


def iter_temp_beans_zero_esc50_examples(
    *,
    limit: int | None = None,
    sample_rate: int = 16000,
    task_id: str | None = None,
) -> Iterator[DatasetExample]:
    """Yield `DatasetExample` rows from a removed temporary backend.

    Raises
    ------
    ImportError
        Always raised. Use HuggingFace-backed dataset loaders instead.
    """
    raise ImportError(
        "This temporary dataset adapter was removed. Use the HuggingFace dataset "
        "loader paths configured via registry YAML instead."
    )
