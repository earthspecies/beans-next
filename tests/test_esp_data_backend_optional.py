"""Tests for the optional `esp_data` dataset backend.

These tests are designed to pass on public installs where `esp_data` is not
available: the backend must be optional and fail with a clear ImportError when
requested explicitly.
"""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import pytest

_ESP_DATA_INSTALLED = importlib.util.find_spec("esp_data") is not None


@pytest.mark.skipif(
    _ESP_DATA_INSTALLED,
    reason="esp_data is installed in this environment; test requires it to be absent",
)
def test_iter_esp_data_beans_zero_examples_missing_dep_is_clear() -> None:
    from beans_next.datasets.esp_data import iter_esp_data_beans_zero_examples

    with pytest.raises(ImportError, match=r"`esp_data` is not installed"):
        next(
            iter_esp_data_beans_zero_examples(
                subset="esc50",
                split="test",
                limit=1,
            )
        )


def test_runner_prefers_explicit_data_source_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from beans_next.runner import runner as runner_mod

    # Env asks for esp_data...
    monkeypatch.setenv("BEANS_PRO_DATA_SOURCE", "esp_data")

    # ...but args explicitly request HF; we should then fail on missing HF-only
    # required key instead of trying to import esp_data.
    args = Namespace(
        data_source="hf",
        hf_path=None,
        split="test",
        hf_config=None,
        limit=1,
    )
    eval_task: Mapping[str, Any] = {
        "eval_task_id": "t1",
        "subset": "esc50",
        "split": "test",
        "hf_path": None,
    }

    with pytest.raises(SystemExit, match=r"Eval task must define `hf_path`"):
        runner_mod._load_examples_for_eval_task(eval_task, args=args)


def test_esp_data_audio_resolution_tries_multiple_gcs_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from beans_next.datasets import esp_data as mod

    attempted: list[str] = []

    def _fake_download(
        gcs_path: str,
        *,
        sample_id: str,
        timeout_s: float | None,
        diagnostics: bool = False,
    ) -> str | None:
        del sample_id, timeout_s, diagnostics
        attempted.append(gcs_path)
        # Fail the first candidate, succeed on the second.
        if gcs_path.endswith("/bad.wav"):
            return None
        if gcs_path.endswith("/good.wav"):
            return "/tmp/good.wav"
        return None

    monkeypatch.setattr(mod, "_download_gcs_to_wav", _fake_download)

    row = {
        mod._DATA_ROOT_KEY: "gs://bucket/root/",
        # Prefer 16k first, but it fails.
        "audio_path_16KHz": "bad.wav",
        # Then fallback to 32k.
        "audio_path_32KHz": "good.wav",
    }

    resolved = mod._resolve_audio_for_row(
        row,
        sample_id="s1",
        subset="HSN-test_5s",
        split="test",
        diagnostics=True,
    )
    assert resolved == "/tmp/good.wav"
    assert attempted == [
        "gs://bucket/root/bad.wav",
        "gs://bucket/root/good.wav",
    ]


def test_esp_data_audio_resolution_supports_birdset_gcs_path_without_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from beans_next.datasets import esp_data as mod

    attempted: list[str] = []

    def _fake_download(
        gcs_path: str,
        *,
        sample_id: str,
        timeout_s: float | None,
        diagnostics: bool = False,
    ) -> str | None:
        del sample_id, timeout_s, diagnostics
        attempted.append(gcs_path)
        if gcs_path == "gs://birdset/audio.wav":
            return "/tmp/birdset.wav"
        return None

    monkeypatch.setattr(mod, "_download_gcs_to_wav", _fake_download)

    row = {
        # No _DATA_ROOT_KEY on purpose.
        "gcs_path": "gs://birdset/audio.wav",
        # Also include a bogus local absolute candidate to prove gcs_path wins
        # (file does not exist).
        "audio_path": "/does/not/exist.wav",
    }

    resolved = mod._resolve_audio_for_row(
        row,
        sample_id="s2",
        subset="HSN-test_5s",
        split="test",
        diagnostics=True,
    )
    assert resolved == "/tmp/birdset.wav"
    assert attempted == ["gs://birdset/audio.wav"]
