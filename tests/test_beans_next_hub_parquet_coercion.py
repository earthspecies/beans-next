"""Unit tests for Hub Parquet row coercion utilities."""

from __future__ import annotations

import pytest

from beans_next.datasets.beans_next_hub import _coerce_str_sequence


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["a", "b"], ["a", "b"]),
        ([" a ", "b "], ["a", "b"]),
        (("a", "b"), ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        (' ["a", "b"] ', ["a", "b"]),
    ],
)
def test_coerce_str_sequence_valid(value: object, expected: list[str]) -> None:
    assert _coerce_str_sequence(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-json",
        '{"a": 1}',
        ["a", ""],
        ["a", None],
        [1, 2],
    ],
)
def test_coerce_str_sequence_invalid(value: object) -> None:
    assert _coerce_str_sequence(value) is None

