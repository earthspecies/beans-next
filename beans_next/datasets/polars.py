"""Parquet-backed dataset iteration using optional ``polars``.

Rows are normalized with the same helpers as HuggingFace loaders
(``hf_row_to_dataset_example``, ``hf_row_metadata``) so Parquet exports that
mirror BEANS-Zero column names work without extra glue code.

Sample identifiers
------------------
Explicit non-empty string values under the configured ``id_field`` (default
``\"id\"``) are reused as ``DatasetExample.sample_id``. Otherwise a synthetic id
is built as ``beanspro:pl:`` followed by a 64-character lower-case hex
``hashlib.sha256`` digest. The digest input is UTF-8 segments joined by ``\\0``
in order: normalized ``parquet_path``, subset filter string (empty when no
``split`` filter is applied), a reserved empty segment (parity with the HF
loader's revision slot), and ``ordinal``. ``ordinal`` is the zero-based row
index assigned **before** optional ``split`` / subset-column filtering (via
``with_row_index``), matching map-style HF loaders where the ordinal is the
underlying dataset index.

Raises
------
ImportError
    If optional ``polars`` is required but not installed.
ValueError
    If subset filtering is misconfigured or the Parquet schema is incompatible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from functools import lru_cache
from types import ModuleType
from typing import Any

from beans_next.api.types import DatasetExample
from beans_next.datasets.base import hf_row_to_dataset_example

_SAMPLE_ID_PREFIX = "beanspro:pl:"
_ROW_INDEX = "__beanspro_row"


@lru_cache(maxsize=1)
def require_polars() -> ModuleType:
    """Import and return the ``polars`` package (cached).

    Returns
    -------
    module
        The installed ``polars`` module.

    Raises
    ------
    ImportError
        If ``polars`` is not installed.
    """
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        msg = (
            "Parquet dataset loaders require the optional `polars` dependency. "
            "Install it with the project extra, for example "
            "`uv sync --extra polars` or `pip install beans-next[polars]`."
        )
        raise ImportError(msg) from exc
    return pl


def synthesize_polars_sample_id(
    *,
    parquet_path: str,
    subset: str | None,
    ordinal: int,
) -> str:
    """Build the fallback ``beanspro:pl:…`` sample id for a Parquet row.

    Parameters
    ----------
    parquet_path
        Path string passed to ``scan_parquet`` / ``read_parquet`` (should match
        the value used when hashing rows in a given run).
    subset
        Active subset / split filter string, or ``None`` when no filter applies.
    ordinal
        Stable zero-based row index **before** subset filtering.

    Returns
    -------
    str
        Deterministic synthetic id.

    Raises
    ------
    ValueError
        If ``ordinal`` is negative.
    """
    if ordinal < 0:
        msg = "ordinal must be non-negative"
        raise ValueError(msg)
    parts = (
        parquet_path,
        subset if subset is not None else "",
        "",
        str(int(ordinal)),
    )
    joined = "\0".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"{_SAMPLE_ID_PREFIX}{digest}"


def resolve_polars_sample_id(
    row: Mapping[str, Any],
    *,
    parquet_path: str,
    subset: str | None,
    ordinal: int,
    id_field: str = "id",
) -> str:
    """Pick ``sample_id`` for a Parquet row using the BEANS-Next Polars scheme.

    Parameters
    ----------
    row
        One row mapping (without the internal row-index column).
    parquet_path, subset, ordinal
        Forwarded to :func:`synthesize_polars_sample_id` when no explicit id
        exists.
    id_field
        Column inspected for a stable string identifier.

    Returns
    -------
    str
        Either the trimmed row id or a synthetic digest id.
    """
    raw = row.get(id_field)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            return stripped
    return synthesize_polars_sample_id(
        parquet_path=parquet_path,
        subset=subset,
        ordinal=ordinal,
    )


def iter_polars_parquet_examples(
    parquet_path: str,
    *,
    split: str | None = None,
    subset_column: str = "split",
    task_id: str | None = None,
    label_field: str = "output",
    id_field: str = "id",
    row_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> Iterator[DatasetExample]:
    """Yield :class:`~beans_next.api.types.DatasetExample` rows from Parquet.

    ``split`` selects a subset when provided: rows where
    ``pl.col(subset_column) == split`` are retained. This mirrors the HF
    loader argument name; for Polars it is a **subset filter**, not a Hub split
    name.

    Parameters
    ----------
    parquet_path
        Filesystem path or glob accepted by ``polars.scan_parquet``.
    split
        When not ``None``, only rows with this value in ``subset_column`` are
        loaded. Each yielded example's ``DatasetExample.split`` is set to this
        string.
    subset_column
        Column used for subset filtering when ``split`` is not ``None``.
    task_id
        Optional eval-task id stored on each example.
    label_field
        Column used for ``DatasetExample.labels`` (BEANS-Zero uses ``output``).
    id_field
        Column inspected for a stable string id (see ``resolve_polars_sample_id``).
    row_filter
        Optional predicate applied to each row mapping after materialization.

    Yields
    ------
    DatasetExample
        One normalized row per retained Parquet row.

    Raises
    ------
    ValueError
        If ``split`` is set with an empty ``subset_column`` name, or the column
        is missing from the file.

    Notes
    -----
    Missing ``polars`` surfaces as ``ImportError`` from :func:`require_polars`.
    """
    pl = require_polars()
    if split is not None and not subset_column.strip():
        msg = "subset_column must be non-empty when split is not None"
        raise ValueError(msg)

    lf0 = pl.scan_parquet(parquet_path)
    if split is not None:
        base_cols = lf0.collect_schema().names()
        if subset_column not in base_cols:
            msg = f"subset_column {subset_column!r} not found in Parquet schema"
            raise ValueError(msg)

    lf = lf0.with_row_index(_ROW_INDEX)
    if split is not None:
        lf = lf.filter(pl.col(subset_column) == split)
    df = lf.collect()

    subset_token = split

    for rec in df.iter_rows(named=True):
        row_dict = dict(rec)
        ordinal_any = row_dict.pop(_ROW_INDEX)
        if not isinstance(ordinal_any, int):
            ordinal = int(ordinal_any)
        else:
            ordinal = ordinal_any

        if row_filter is not None and not row_filter(row_dict):
            continue

        sample_id = resolve_polars_sample_id(
            row_dict,
            parquet_path=parquet_path,
            subset=subset_token,
            ordinal=ordinal,
            id_field=id_field,
        )
        yield hf_row_to_dataset_example(
            row_dict,
            sample_id=sample_id,
            task_id=task_id,
            split=split,
            label_field=label_field,
        )
