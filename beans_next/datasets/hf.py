"""Map-style (non-streaming) HuggingFace ``datasets`` loaders.

This module wraps ``datasets.load_dataset`` for finite splits and yields
:class:`~beans_next.api.types.DatasetExample` rows with deterministic ids (see
``beans_next.datasets.base``).

Raises
------
ImportError
    If ``datasets`` is not installed.
TypeError
    If ``load_dataset`` does not return a map-style ``datasets.Dataset``.
ValueError
    If ``streaming=True`` is passed in ``load_dataset_kwargs`` (use
    ``beans_next.datasets.hf_streaming`` instead).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from beans_next.api.types import DatasetExample
from beans_next.datasets.base import (
    hf_row_to_dataset_example,
    iter_filtered_indices,
    require_datasets,
    resolve_hf_sample_id,
)

_LOG = logging.getLogger(__name__)
# Env var controlling parallel WAV materialization workers (soundfile releases GIL).
_WORKERS_ENV = "BEANS_PRO_HF_WORKERS"


def _heavy_columns(features: Mapping[str, Any]) -> list[str]:
    """Return column names whose HF feature type is expensive to decode per-row.

    Sequences of scalars (e.g. raw audio float arrays) and ``Audio`` features
    are the main culprits in BEANS-Zero. Skipping them during the filter scan
    avoids decoding large arrays for every row just to check a string field.

    Parameters
    ----------
    features
        ``dataset.features`` mapping (HuggingFace ``Features`` object).

    Returns
    -------
    list[str]
        Column names that should be dropped for the lightweight filter pass.
    """
    heavy: list[str] = []
    for col, feat in features.items():
        type_name = type(feat).__name__
        if type_name in ("Sequence", "Audio", "Image"):
            heavy.append(col)
        elif type_name == "Value" and getattr(feat, "dtype", "") in (
            "binary",
            "large_binary",
        ):
            heavy.append(col)
    return heavy


def iter_hf_dataset_examples(
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
    """Yield normalized examples from a map-style Hub dataset split.

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
        Column inspected for a stable string id (see ``resolve_hf_sample_id``).
    row_filter
        Optional predicate to skip rows client-side.
    load_dataset_kwargs
        Extra keyword arguments forwarded to ``datasets.load_dataset``. Must
        not set ``streaming=True``.

    Yields
    ------
    DatasetExample
        One normalized row per retained dataset index.

    Raises
    ------
    TypeError
        If the loaded object is not a finite ``datasets.Dataset``.
    ValueError
        If ``streaming=True`` is requested via ``load_dataset_kwargs``.

    Notes
    -----
    :func:`~beans_next.datasets.base.require_datasets` may raise
    ``ImportError`` when the optional ``datasets`` package is missing.
    """
    kwargs = dict(load_dataset_kwargs or ())
    if kwargs.get("streaming"):
        msg = "streaming loads belong in beans_next.datasets.hf_streaming"
        raise ValueError(msg)
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
    if not hasattr(loaded, "__len__") or not hasattr(loaded, "__getitem__"):
        msg = (
            "load_dataset returned a non-map dataset; use "
            "beans_next.datasets.hf_streaming.iter_hf_streaming_examples for "
            "streaming IterableDataset instances."
        )
        raise TypeError(msg)

    # Two-phase fast-filter for map-style datasets with a row_filter predicate.
    #
    # Naive approach: loaded[idx] for every idx decodes all columns including
    # large audio arrays, making the scan O(N * audio_size).
    #
    # Fast approach (two sub-steps):
    #   1. Strip heavy columns → bulk to_dict() of lightweight metadata columns
    #      (all columns as Python lists in one shot, ~1-2s for 91965 rows).
    #   2. loaded.select(matching_indices) → iterate only the matched rows.
    #
    # For BEANS-Zero (91965 rows, audio Sequence(float64)), this reduces the
    # filter scan from ~57 min (cold, downloading) or ~10 s (warm, enumerate)
    # to ~1-2 s.
    if row_filter is not None and hasattr(loaded, "select") and hasattr(
        loaded, "features"
    ):
        try:
            heavy = _heavy_columns(loaded.features)
            if heavy:
                light = loaded.remove_columns(heavy)
                # Bulk convert to dict of lists (one Arrow read per column).
                light_dict: dict[str, list[Any]] = {
                    col: light[col] for col in light.column_names  # type: ignore[union-attr]
                }
                n_total = len(light)  # type: ignore[arg-type]
                matching: list[int] = [
                    i
                    for i in range(n_total)
                    if row_filter({col: light_dict[col][i] for col in light_dict})
                ]
                _LOG.debug(
                    "hf fast-filter: scanned %d rows → %d matches (dropped %s)",
                    n_total,
                    len(matching),
                    heavy,
                )
                loaded = loaded.select(matching)
                row_filter = None  # already filtered; don't re-check below
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("hf fast-filter skipped (%s); falling back to sequential", exc)

    # Use numpy format for audio to avoid GIL-heavy Arrow→Python list conversion.
    # set_format("numpy") makes dataset[i]["audio"] return a numpy float32 array
    # instead of a Python list, giving ~27× speedup for 220k-sample rows.
    try:
        if hasattr(loaded, "set_format"):
            loaded.set_format("numpy", output_all_columns=True)
            _LOG.debug("hf: set_format numpy enabled")
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("hf: set_format numpy failed (%s); using default format", exc)

    # Resolve parallel workers: BEANS_PRO_HF_WORKERS env var (default 1).
    # soundfile.write releases the GIL, so ThreadPoolExecutor gives real speedup
    # for the WAV-materialization step inside hf_row_to_dataset_example.
    try:
        _workers = max(1, int(os.environ.get(_WORKERS_ENV, "1")))
    except ValueError:
        _workers = 1

    length = len(loaded)  # type: ignore[arg-type]

    def _getter(idx: int) -> Mapping[str, Any]:
        return loaded[idx]  # type: ignore[index]

    def _make_example(idx_row: tuple[int, Mapping[str, Any]]) -> DatasetExample:
        idx, row = idx_row
        sample_id = resolve_hf_sample_id(
            row,
            path_or_id=path_or_id,
            split=split,
            revision=revision,
            ordinal=idx,
            id_field=id_field,
        )
        return hf_row_to_dataset_example(
            row,
            sample_id=sample_id,
            task_id=task_id,
            split=split,
            label_field=label_field,
        )

    items = iter_filtered_indices(length, _getter, row_filter)

    if _workers > 1:
        with ThreadPoolExecutor(max_workers=_workers) as pool:
            yield from pool.map(_make_example, items)
    else:
        for item in items:
            yield _make_example(item)
