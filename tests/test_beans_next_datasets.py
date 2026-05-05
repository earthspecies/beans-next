"""Unit tests for custom BEANS-Next dataset classes.

These tests validate the offline, local-file behavior of:
- `beans_next.datasets.beans_next.BeansPro` (single-audio rows)
- `beans_next.datasets.beans_next_multiaudio.BeansProMultiAudio` (multi-audio rows)

The goal is to lock down the per-row contract (required keys, audio decoding,
and `output_take_and_give` mapping) without depending on GCS availability.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("esp_data")


def _write_wav(path: Path, *, sample_rate: int = 16_000, n_frames: int = 160) -> None:
    """Write a minimal mono PCM16 WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = (np.zeros((n_frames,), dtype=np.int16)).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def test_beans_next_single_audio_row_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from beans_next.datasets.beans_next import BeansPro

    audio_rel = Path("audio") / "example.wav"
    _write_wav(tmp_path / audio_rel, sample_rate=16_000, n_frames=400)

    jsonl_path = tmp_path / "split.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "row0",
                "instruction": "<Audio><AudioHere></Audio>\nPick A/B/C/D.\nA: x\nB: y\nC: z\nD: w",
                "output": "B",
                "audio_path_original_sample_rate": str(audio_rel),
                "metadata": json.dumps({"species": "testus", "duration_s": 0.025}),
                "dataset_name": "beans_next",
                "source_dataset": "unit_test",
                "license": "CC0",
                "task": "crow-description",
                "file_name": "example.wav",
                "instruction_text": "Pick A/B/C/D.",
            }
        ],
    )

    monkeypatch.setattr(
        BeansPro,
        "info",
        BeansPro.info.model_copy(update={"split_paths": {"unit": str(jsonl_path)}}),
    )
    monkeypatch.setattr(BeansPro, "_default_data_roots", {"unit": str(tmp_path)})

    ds = BeansPro(split="unit", sample_rate=16_000, data_root=str(tmp_path), backend="polars")
    item = ds[0]

    assert isinstance(item, dict)
    assert "audio" in item
    assert isinstance(item["audio"], np.ndarray)
    assert item["audio"].dtype == np.float32
    assert item["output"] == "B"
    assert item["audio_path_original_sample_rate"] == str(audio_rel)


def test_beans_next_single_audio_take_and_give(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from beans_next.datasets.beans_next import BeansPro

    audio_rel = Path("audio") / "example.wav"
    _write_wav(tmp_path / audio_rel, sample_rate=16_000, n_frames=80)
    jsonl_path = tmp_path / "split.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "row0",
                "instruction": "<Audio><AudioHere></Audio>\nAnswer Yes or No.",
                "output": "Yes",
                "audio_path_original_sample_rate": str(audio_rel),
                "metadata": "{}",
            }
        ],
    )

    monkeypatch.setattr(
        BeansPro,
        "info",
        BeansPro.info.model_copy(update={"split_paths": {"unit": str(jsonl_path)}}),
    )
    monkeypatch.setattr(BeansPro, "_default_data_roots", {"unit": str(tmp_path)})

    ds = BeansPro(
        split="unit",
        data_root=str(tmp_path),
        output_take_and_give={
            "instruction": "prompt",
            "output": "label",
            "audio_path_original_sample_rate": "audio_relpath",
        },
        backend="polars",
    )
    item = ds[0]

    assert set(item) == {"prompt", "label", "audio_relpath"}
    assert item["label"] == "Yes"
    assert item["audio_relpath"] == str(audio_rel)


def test_beans_next_multiaudio_row_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from beans_next.datasets import beans_next_multiaudio as mod

    # Build 4 audio files and a JSONL row referencing them.
    audio_paths = []
    for i in range(4):
        rel = Path("audio") / f"clip_{i}.wav"
        _write_wav(tmp_path / rel, sample_rate=32_000, n_frames=320)
        audio_paths.append(str(rel))

    jsonl_path = tmp_path / "multi.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "row0",
                "task": "unit_task",
                "audio_paths": audio_paths,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<Audio><AudioHere></Audio> <Audio><AudioHere></Audio> "
                            "<Audio><AudioHere></Audio> <Audio><AudioHere></Audio>\n"
                            "Which option matches?"
                        ),
                    },
                    {"role": "assistant", "content": "A"},
                ],
            }
        ],
    )

    # Patch split map to point at local JSONL.
    monkeypatch.setattr(mod, "_SPLITS", {"unit": str(jsonl_path)})

    ds = mod.BeansProMultiAudio(split="unit", data_root=str(tmp_path), sample_rate=32_000)
    item = ds[0]

    assert "audios" in item
    assert isinstance(item["audios"], list)
    assert len(item["audios"]) == 4
    for audio in item["audios"]:
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32


def test_beans_next_multiaudio_take_and_give(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from beans_next.datasets import beans_next_multiaudio as mod

    rel = Path("audio") / "clip.wav"
    _write_wav(tmp_path / rel, sample_rate=32_000, n_frames=128)
    jsonl_path = tmp_path / "multi.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "row0",
                "audio_paths": [str(rel)],
                "messages": [{"role": "user", "content": "<Audio><AudioHere></Audio> hi"}],
            }
        ],
    )

    monkeypatch.setattr(mod, "_SPLITS", {"unit": str(jsonl_path)})
    ds = mod.BeansProMultiAudio(
        split="unit",
        data_root=str(tmp_path),
        output_take_and_give={"id": "sample_id", "audios": "audios"},
        sample_rate=32_000,
    )
    item = ds[0]
    assert set(item) == {"sample_id", "audios"}
    assert item["sample_id"] == "row0"
    assert isinstance(item["audios"], list) and len(item["audios"]) == 1
