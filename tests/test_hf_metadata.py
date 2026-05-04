"""Regression tests for HuggingFace row metadata normalization."""

from __future__ import annotations

from beans_next.datasets import hf_row_metadata


def test_hf_row_metadata_audio_path_alias_from_decoded_audio() -> None:
    """``metadata_key: audio_path`` resolves; Hub rows store path under ``audio``."""
    row = {
        "audio": {"path": "/tmp/example.wav", "sampling_rate": 16000},
        "output": "dog",
    }
    meta = hf_row_metadata(row, sample_id="test:0")
    assert meta["audio_path"] == "/tmp/example.wav"
    assert meta["audio"]["path"] == "/tmp/example.wav"


def test_hf_row_metadata_no_audio_path_without_string_path() -> None:
    """In-memory audio arrays are materialized to a temp WAV ``audio_path``."""
    row = {"audio": {"array": [0.0, 0.0], "sampling_rate": 16000}, "output": "x"}
    meta = hf_row_metadata(row, sample_id="test:1")
    assert isinstance(meta.get("audio_path"), str)
    assert str(meta["audio_path"]).endswith(".wav")
