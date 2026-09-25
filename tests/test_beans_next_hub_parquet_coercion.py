"""Unit tests for Hub Parquet row coercion utilities."""

from __future__ import annotations

import pytest

from beans_next.datasets.beans_next_hub import (
    _multiaudio_repo_rel_paths,
)
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER


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


def test_multiaudio_repo_rel_paths_rejects_conflicting_query() -> None:
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
    with pytest.raises(ValueError, match="Conflicting query"):
        _multiaudio_repo_rel_paths(row)


@pytest.mark.parametrize("legacy", [False, True])
def test_multiaudio_preserves_references_and_true_query(legacy: bool) -> None:
    row = {
        "messages": [{"role": "user", "content": AUDIO_PLACEHOLDER * 3}],
        "context_audio_paths": ["a.wav", "b.wav", "q.wav"]
        if legacy
        else ["a.wav", "b.wav"],
        "query_audio_path": "a.wav" if legacy else "q.wav",
        "audio_paths": ["unavailable/original.wav"] * 3,
    }
    assert _multiaudio_repo_rel_paths(row) == ["a.wav", "b.wav", "q.wav"]


def test_multiaudio_repo_rel_paths_back_compat_old_keys() -> None:
    user = "Q: " + AUDIO_PLACEHOLDER + "\n"
    row = {
        "messages": [{"role": "user", "content": user}],
        "context_audio_paths": ["audio/q.wav"],
        "query_audio_path": "audio/q.wav",
    }
    assert _multiaudio_repo_rel_paths(row) == ["audio/q.wav"]
