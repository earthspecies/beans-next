"""Streaming HuggingFace ``datasets`` iterators.

Streaming splits return an ``IterableDataset``. Identifiers match
``beans_next.datasets.base``: explicit string ``id`` rows reuse that value;
synthetic ids use the digest scheme with a monotonic **yield** ordinal (rows
skipped by ``row_filter`` do not advance the ordinal).

Raises
------
ImportError
    If ``datasets`` is not installed.
ValueError
    If ``streaming`` is not enabled in ``load_dataset_kwargs``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from beans_next.api.types import DatasetExample
from beans_next.datasets.base import (
    hf_row_to_dataset_example,
    require_datasets,
    resolve_hf_sample_id,
)


def iter_hf_streaming_examples(
    path_or_id: str,
    *,
    split: str,
    config_name: str | None = None,
    revision: str | None = None,
    task_id: str | None = None,
    label_field: str = "output",
    id_field: str = "id",
    row_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    load_dataset_kwargs: Mapping[str, Any] | None = None,
) -> Iterator[DatasetExample]:
    """Yield normalized examples from a streaming Hub dataset split.

    Parameters
    ----------
    path_or_id
        Path or repository id for ``datasets.load_dataset``.
    split
        Split name (for example ``\"test\"`` for BEANS-Zero).
    config_name
        Optional configuration / subset name (second positional argument to
        ``load_dataset``).
    revision
        Optional Hub revision string.
    task_id
        Optional eval-task id stored on each example.
    label_field
        Column used for ``DatasetExample.labels``.
    id_field
        Column inspected for a stable string id.
    row_filter
        Optional predicate to skip rows client-side.
    load_dataset_kwargs
        Extra keyword arguments forwarded to ``datasets.load_dataset``. This
        iterator always passes ``streaming=True``; callers may supply additional
        keys (for example ``trust_remote_code``) but must not set
        ``streaming=False``.

    Yields
    ------
    DatasetExample
        One normalized row per retained streaming example.

    Raises
    ------
    ValueError
        If ``load_dataset_kwargs`` sets ``streaming=False``.

    Notes
    -----
    :func:`~beans_next.datasets.base.require_datasets` may raise
    ``ImportError`` when the optional ``datasets`` package is missing.
    """
    kwargs = dict(load_dataset_kwargs or ())
    if kwargs.get("streaming") is False:
        msg = "non-streaming loads belong in beans_next.datasets.hf"
        raise ValueError(msg)
    kwargs["streaming"] = True
    datasets = require_datasets()
    if config_name is None:
        loaded = datasets.load_dataset(
            path_or_id,
            split=split,
            revision=revision,
            **kwargs,
        )
    else:
        loaded = datasets.load_dataset(
            path_or_id,
            config_name,
            split=split,
            revision=revision,
            **kwargs,
        )
    yield_ordinal = 0
    for row in loaded:
        if row_filter is not None and not row_filter(row):
            continue
        mapping = dict(row) if not isinstance(row, Mapping) else row
        sample_id = resolve_hf_sample_id(
            mapping,
            path_or_id=path_or_id,
            split=split,
            revision=revision,
            ordinal=yield_ordinal,
            id_field=id_field,
        )
        yield hf_row_to_dataset_example(
            mapping,
            sample_id=sample_id,
            task_id=task_id,
            split=split,
            label_field=label_field,
        )
        yield_ordinal += 1
