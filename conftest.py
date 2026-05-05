"""Root pytest configuration: CLI options and shared test fixtures."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from beans_next.api.types import DatasetExample


# ---------------------------------------------------------------------------
# CLI options (used by tests/consistency/)
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI options passed to the test suite."""
    parser.addoption("--base_folder", action="store", default=".")
    parser.addoption("--skip_files_list", action="store", default=[".pyc"])
    parser.addoption("--device", action="store", default="cpu")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize fixtures that mirror CLI option values."""
    option_value = metafunc.config.option.skip_files_list
    if "skip_files_list" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("skip_files_list", [option_value])

    option_value = metafunc.config.option.base_folder
    if "base_folder" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("base_folder", [option_value])

    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value])


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


def make_wav(path: Path, *, sample_rate: int = 16_000, n_frames: int = 160) -> Path:
    """Write a minimal silent mono PCM-16 WAV and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros(n_frames, dtype=np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    return path


# ---------------------------------------------------------------------------
# Fixtures: single-audio BEANS-Next row
# ---------------------------------------------------------------------------


@pytest.fixture
def beans_next_wav(tmp_path: Path) -> Path:
    """Silent 10 ms WAV file at 16 kHz."""
    return make_wav(tmp_path / "audio" / "example.wav", sample_rate=16_000, n_frames=160)


@pytest.fixture
def beans_next_row(beans_next_wav: Path) -> dict[str, object]:
    """Minimal single-audio BEANS-Next metadata row (crow-description schema)."""
    return {
        "id": "test_row_0",
        "dataset_name": "crow-description",
        "source_dataset": "unit_test",
        "task": "multiple_choice",
        "license": "CC0",
        "file_name": beans_next_wav.name,
        "audio_path_original_sample_rate": str(beans_next_wav),
        "instruction": (
            "<Audio><AudioHere></Audio> Which acoustic description best matches?"
            "\nA: short buzz\nB: long whine\nC: click\nD: trill"
        ),
        "instruction_text": "Which acoustic description best matches?",
        "output": "A",
        "metadata": json.dumps({"call_type": "E1", "species": "Corvus corone"}),
    }


@pytest.fixture
def beans_next_example(beans_next_row: dict[str, object], beans_next_wav: Path) -> DatasetExample:
    """``DatasetExample`` from a minimal single-audio row (no esp_data needed)."""
    from beans_next.datasets.esp_data import _build_dataset_example

    return _build_dataset_example(
        beans_next_row,
        sample_id="beanspro:test:0",
        audio_path=str(beans_next_wav),
        split="test",
        task_id="beans_next_crow_description",
    )


# ---------------------------------------------------------------------------
# Fixtures: multi-audio BEANS-Next row
# ---------------------------------------------------------------------------


@pytest.fixture
def beans_next_multiaudio_wavs(tmp_path: Path) -> list[Path]:
    """Four silent WAV files at 32 kHz (context clips for a tier-4 row)."""
    return [
        make_wav(tmp_path / "audio" / f"clip_{i}.wav", sample_rate=32_000, n_frames=320)
        for i in range(4)
    ]


@pytest.fixture
def beans_next_multiaudio_row(beans_next_multiaudio_wavs: list[Path]) -> dict[str, object]:
    """Minimal multi-audio BEANS-Next metadata row (crow-4way schema)."""
    return {
        "id": "test_multi_0",
        "task": "unit_task",
        "audio_ids": [str(p) for p in beans_next_multiaudio_wavs],
        "query_audio_id": str(beans_next_multiaudio_wavs[0]),
        "messages": [
            {
                "role": "user",
                "content": (
                    "<Audio><AudioHere></Audio> " * len(beans_next_multiaudio_wavs)
                    + "Which matches the query?"
                ),
            },
            {"role": "assistant", "content": "A"},
        ],
        "output": "A",
        "metadata": json.dumps({"call_type": "E1"}),
    }


@pytest.fixture
def beans_next_multiaudio_example(
    beans_next_multiaudio_row: dict[str, object],
    beans_next_multiaudio_wavs: list[Path],
) -> DatasetExample:
    """``DatasetExample`` from a minimal multi-audio row (no esp_data needed)."""
    from beans_next.datasets.esp_data import _build_multiaudio_dataset_example

    paths = [str(p) for p in beans_next_multiaudio_wavs]
    return _build_multiaudio_dataset_example(
        beans_next_multiaudio_row,
        sample_id="beanspro:test:multi:0",
        audio_paths=paths,
        query_audio_path=paths[0],
        split="test",
        task_id="beans_next_crow_4way",
    )
