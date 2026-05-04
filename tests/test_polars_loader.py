"""Tests for optional Polars-backed Parquet dataset iteration."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("polars")

import polars as pl

from beans_next.datasets import (
    iter_polars_parquet_examples,
    require_polars,
    resolve_polars_sample_id,
    synthesize_polars_sample_id,
)


def test_require_polars_returns_module() -> None:
    """``require_polars`` returns the installed third-party module."""
    assert require_polars().__name__ == "polars"


def test_synthesize_polars_sample_id_rejects_negative_ordinal() -> None:
    """Ordinal used in digests must be non-negative."""
    with pytest.raises(ValueError, match="ordinal"):
        synthesize_polars_sample_id(parquet_path="p", subset=None, ordinal=-1)


def test_iter_polars_parquet_all_rows(tmp_path: Path) -> None:
    """Rows without an ``id`` get deterministic synthetic sample ids."""
    path = tmp_path / "ex.parquet"
    pl.DataFrame(
        {
            "split": ["train", "test"],
            "output": ["a", "b"],
        },
    ).write_parquet(path)

    examples = list(iter_polars_parquet_examples(str(path)))
    assert len(examples) == 2
    assert examples[0].labels == "a"
    assert examples[0].sample_id.startswith("beanspro:pl:")
    assert examples[1].labels == "b"


def test_iter_polars_parquet_subset_filter(tmp_path: Path) -> None:
    """``split`` filters via ``subset_column`` and sets ``DatasetExample.split``."""
    path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "id": ["x1", "x2"],
            "split": ["train", "test"],
            "output": ["0", "1"],
        },
    ).write_parquet(path)

    test_rows = list(
        iter_polars_parquet_examples(str(path), split="test", subset_column="split"),
    )
    assert len(test_rows) == 1
    assert test_rows[0].sample_id == "x2"
    assert test_rows[0].split == "test"
    assert test_rows[0].labels == "1"


def test_iter_polars_row_filter(tmp_path: Path) -> None:
    """Optional ``row_filter`` drops rows before id resolution."""
    path = tmp_path / "f.parquet"
    pl.DataFrame(
        {
            "id": ["keep", "drop"],
            "split": ["test", "test"],
            "output": ["y", "n"],
        },
    ).write_parquet(path)

    def _keep_first(row: dict) -> bool:
        return row["id"] == "keep"

    rows = list(
        iter_polars_parquet_examples(
            str(path),
            split="test",
            row_filter=_keep_first,
        ),
    )
    assert len(rows) == 1
    assert rows[0].sample_id == "keep"


def test_missing_subset_column_raises(tmp_path: Path) -> None:
    """A non-``None`` ``split`` requires the configured column in the file."""
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"id": ["1"], "output": ["z"]}).write_parquet(path)
    with pytest.raises(ValueError, match="subset_column"):
        list(iter_polars_parquet_examples(str(path), split="x"))


def test_empty_subset_column_with_split_raises() -> None:
    """Reject an empty subset column name when filtering."""
    with pytest.raises(ValueError, match="subset_column"):
        list(
            iter_polars_parquet_examples(
                "missing.parquet",
                split="a",
                subset_column="  ",
            ),
        )


def test_resolve_polars_explicit_id() -> None:
    """Non-empty string ``id_field`` wins over synthetic ids."""
    sid = resolve_polars_sample_id(
        {"id": "  u1  ", "output": "q"},
        parquet_path="/tmp/x.parquet",
        subset=None,
        ordinal=0,
    )
    assert sid == "u1"
