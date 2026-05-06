"""Unit tests for Hub Parquet row coercion utilities."""

from __future__ import annotations

import pytest

from beans_next.datasets.beans_next_hub import (
    _coerce_str_sequence,
    _multiaudio_repo_rel_paths,
)
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER


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


def test_multiaudio_repo_rel_paths_uses_context_when_query_matches_tail() -> None:
    user = (
        "A: " + AUDIO_PLACEHOLDER + "\n"
        "B: " + AUDIO_PLACEHOLDER + "\n"
        "Q: " + AUDIO_PLACEHOLDER + "\n"
    )
    row = {
        "messages": [{"role": "user", "content": user}],
        "context_source_paths": ["audio/a.wav", "audio/b.wav", "audio/q.wav"],
        "query_source_path": "audio/q.wav",
        "source_audio_paths": ["audio/legacy.wav", "audio/x.wav", "audio/y.wav"],
    }
    assert _multiaudio_repo_rel_paths(row) == [
        "audio/a.wav",
        "audio/b.wav",
        "audio/q.wav",
    ]


def test_multiaudio_repo_rel_paths_replaces_tail_for_fewshot_query() -> None:
    user = (
        "A: " + AUDIO_PLACEHOLDER + "\n"
        "B: " + AUDIO_PLACEHOLDER + "\n"
        "C: " + AUDIO_PLACEHOLDER + "\n"
        "Q: " + AUDIO_PLACEHOLDER + "\n"
    )
    row = {
        "messages": [{"role": "user", "content": user}],
        "context_source_paths": [
            "audio/a.wav",
            "audio/b.wav",
            "audio/c.wav",
            "audio/wrong.wav",
        ],
        "query_source_path": "audio/q.wav",
    }
    assert _multiaudio_repo_rel_paths(row) == [
        "audio/a.wav",
        "audio/b.wav",
        "audio/c.wav",
        "audio/q.wav",
    ]


def test_multiaudio_repo_rel_paths_back_compat_old_keys() -> None:
    user = "Q: " + AUDIO_PLACEHOLDER + "\n"
    row = {
        "messages": [{"role": "user", "content": user}],
        "context_audio_paths": ["audio/q.wav"],
        "query_audio_path": "audio/q.wav",
    }
    assert _multiaudio_repo_rel_paths(row) == ["audio/q.wav"]
