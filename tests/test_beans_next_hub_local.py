"""Tests for loading BEANS-Next from a local Hugging Face snapshot."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from beans_next.datasets import beans_next_hub


def test_local_snapshot_resolves_split_metadata_and_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = tmp_path / "test" / "metadata.parquet"
    audio = tmp_path / "test" / "audio" / "ab" / "sample.wav"
    metadata.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    metadata.touch()
    audio.touch()
    monkeypatch.setenv("BEANS_NEXT_HF_BEANS_NEXT_ROOT", str(tmp_path))

    assert (
        beans_next_hub._hub_metadata_filename("unused", "unused")
        == "test/metadata.parquet"
    )
    assert beans_next_hub._local_hub_file(
        "unused", "test/metadata.parquet", revision="unused"
    ) == str(metadata)
    assert beans_next_hub._hf_download_audio_path(
        "unused",
        "audio/ab/sample.wav",
        revision="unused",
        base_dir="test",
    ) == str(audio)


def test_local_snapshot_never_falls_back_to_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BEANS_NEXT_HF_BEANS_NEXT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        beans_next_hub,
        "hf_hub_download",
        lambda *args, **kwargs: pytest.fail("must not contact Hugging Face"),
    )

    with pytest.raises(FileNotFoundError, match="File not found"):
        beans_next_hub._hf_download_audio_path(
            "unused", "audio/missing.wav", revision="unused", base_dir="test"
        )


def test_hf_workers_env_applies_to_beans_next_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, int] = {}

    def fake_iter(
        *args: object, workers: int, **kwargs: object
    ) -> Iterator[beans_next_hub.DatasetExample]:
        seen["workers"] = workers
        return iter(())

    monkeypatch.setenv("BEANS_NEXT_HF_WORKERS", "8")
    monkeypatch.setattr(beans_next_hub, "_iter_single_examples", fake_iter)

    assert (
        list(
            beans_next_hub.iter_hf_beans_next_examples(
                subset="t1-caption", revision="unused"
            )
        )
        == []
    )
    assert seen == {"workers": 8}
