"""Tests for text-only prompt rendering and metadata-only dataset loading."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from beans_next.api.types import DatasetExample
from beans_next.datasets import beans_next_hub, hf
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER
from beans_next.prompts.renderer import (
    TEXT_ONLY_INSTRUCTION,
    AudioSlotSpec,
    PromptRenderer,
    PromptSpec,
)
from beans_next.runner.runner import _sample_examples


def _prompt_spec(content: str, *, list_slot: bool = False) -> PromptSpec:
    slot = (
        AudioSlotSpec(list_metadata_key="audio_paths")
        if list_slot
        else AudioSlotSpec(metadata_key="audio_path")
    )
    return PromptSpec(
        prompt_id="test.prompt",
        message_templates=(("user", content),),
        audio_slots=(slot,),
    )


def test_text_only_removes_single_audio_without_resolving_slot() -> None:
    spec = _prompt_spec(f"{AUDIO_PLACEHOLDER}\nChoose one. A: fox B: bird")
    example = DatasetExample(sample_id="sample", metadata={})

    request = PromptRenderer(spec, modality_mode="text-only").render(example)

    assert request.audio_inputs == []
    assert AUDIO_PLACEHOLDER not in request.messages[0].content
    assert "A: fox B: bird" in request.messages[0].content


def test_text_only_removes_multiple_and_query_only_audio_tokens() -> None:
    content = (
        f"{AUDIO_PLACEHOLDER} [audio] {AUDIO_PLACEHOLDER}\n"
        "Which option matches? A: first B: second"
    )
    spec = _prompt_spec(content, list_slot=True)
    example = DatasetExample(sample_id="sample", metadata={})

    request = PromptRenderer(spec, modality_mode="text-only").render(example)

    rendered = request.messages[0].content
    assert request.audio_inputs == []
    assert AUDIO_PLACEHOLDER not in rendered
    assert "[audio]" not in rendered.lower()
    assert "A: first B: second" in rendered


def test_text_only_informed_prepends_fixed_user_instruction() -> None:
    spec = _prompt_spec(f"{AUDIO_PLACEHOLDER}\nAnswer Yes or No.")
    example = DatasetExample(sample_id="sample", metadata={})

    request = PromptRenderer(spec, modality_mode="text-only-informed").render(example)

    assert request.audio_inputs == []
    assert request.messages[0].content.startswith(TEXT_ONLY_INSTRUCTION)
    assert request.messages[0].content.endswith("Answer Yes or No.")


def test_audio_mode_keeps_existing_alignment_behavior() -> None:
    spec = _prompt_spec(f"{AUDIO_PLACEHOLDER}\nAnswer Yes or No.")
    example = DatasetExample(
        sample_id="sample",
        metadata={"audio_path": "/shared/example.wav"},
    )

    request = PromptRenderer(spec, modality_mode="audio").render(example)

    assert len(request.audio_inputs) == 1
    assert request.audio_inputs[0].data == "/shared/example.wav"
    assert request.messages[0].content.startswith(AUDIO_PLACEHOLDER)


def test_beans_next_metadata_only_loader_does_not_resolve_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "id": "row-1",
        "task": "t1-caption",
        "file_name": "audio/missing.wav",
        "instruction": f"{AUDIO_PLACEHOLDER}\nDescribe the sound.",
        "output": "bird call",
    }
    monkeypatch.setattr(beans_next_hub, "_hub_metadata_filename", lambda *_: "meta")
    monkeypatch.setattr(
        beans_next_hub,
        "_local_hub_file",
        lambda *_args, **_kwargs: "meta",
    )
    monkeypatch.setattr(
        beans_next_hub,
        "iter_parquet_row_dicts",
        lambda _path: iter((row,)),
    )
    monkeypatch.setattr(
        beans_next_hub,
        "_prefetch_hub_files",
        lambda *_args, **_kwargs: pytest.fail("audio files must not be resolved"),
    )
    monkeypatch.setattr(
        beans_next_hub,
        "_hub_has_legacy_audio_parquet",
        lambda *_args, **_kwargs: pytest.fail("legacy audio must not be inspected"),
    )

    examples = list(
        beans_next_hub.iter_hf_beans_next_examples(
            subset="t1-caption",
            revision="test",
            load_audio=False,
        )
    )

    assert len(examples) == 1
    assert examples[0].labels == "bird call"
    assert "audio_path" not in examples[0].metadata


def test_beans_next_multiaudio_metadata_only_loader_skips_all_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = (
        f"{AUDIO_PLACEHOLDER} {AUDIO_PLACEHOLDER}\n"
        "Which option matches? A: first B: second"
    )
    row = {
        "id": "row-2",
        "task": "crow-4way",
        "source_audio_paths": ["audio/first.wav", "audio/second.wav"],
        "messages": [
            {"role": "user", "content": conversation},
            {"role": "assistant", "content": "B"},
        ],
    }
    monkeypatch.setattr(beans_next_hub, "_hub_metadata_filename", lambda *_: "meta")
    monkeypatch.setattr(
        beans_next_hub,
        "_local_hub_file",
        lambda *_args, **_kwargs: "meta",
    )
    monkeypatch.setattr(
        beans_next_hub,
        "iter_parquet_row_dicts",
        lambda _path: iter((row,)),
    )
    monkeypatch.setattr(
        beans_next_hub,
        "_prefetch_hub_files",
        lambda *_args, **_kwargs: pytest.fail("audio files must not be resolved"),
    )
    monkeypatch.setattr(
        beans_next_hub,
        "_hub_has_legacy_audio_parquet",
        lambda *_args, **_kwargs: pytest.fail("legacy audio must not be inspected"),
    )

    examples = list(
        beans_next_hub.iter_hf_beans_next_examples(
            subset="crow-4way",
            revision="test",
            load_audio=False,
        )
    )

    assert len(examples) == 1
    assert examples[0].labels == "B"
    assert examples[0].metadata["conversation"] == conversation
    assert "audio_paths" not in examples[0].metadata


class Audio:
    pass


class _ValueFeature:
    dtype = "string"


class _FakeDataset:
    def __init__(self, *, include_audio: bool = True) -> None:
        self.include_audio = include_audio
        self.features = {"dataset_name": _ValueFeature()}
        if include_audio:
            self.features["audio"] = Audio()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        assert index == 0
        if self.include_audio:
            pytest.fail("audio column was accessed")
        return {"id": "row-1", "dataset_name": "esc50", "output": "dog"}

    def remove_columns(self, columns: list[str]) -> _FakeDataset:
        assert columns == ["audio"]
        return _FakeDataset(include_audio=False)

    def set_format(self, *_args: object, **_kwargs: object) -> None:
        return None


def test_generic_hf_metadata_only_loader_removes_audio_before_iteration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del tmp_path
    datasets = SimpleNamespace(load_dataset=lambda *_args, **_kwargs: _FakeDataset())
    monkeypatch.setattr(hf, "require_datasets", lambda: datasets)

    examples = list(
        hf.iter_hf_dataset_examples(
            "unused",
            split="test",
            load_audio=False,
        )
    )

    assert len(examples) == 1
    assert examples[0].labels == "dog"
    assert "audio_path" not in examples[0].metadata


def test_deterministic_fraction_samples_each_task_exactly() -> None:
    examples = [DatasetExample(sample_id=f"sample-{index}") for index in range(20)]

    first = _sample_examples(examples, fraction=0.1, seed=17)
    second = _sample_examples(examples, fraction=0.1, seed=17)
    different_seed = _sample_examples(examples, fraction=0.1, seed=18)

    assert len(first) == 2
    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert [row.sample_id for row in first] != [
        row.sample_id for row in different_seed
    ]


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_deterministic_fraction_rejects_invalid_values(fraction: float) -> None:
    with pytest.raises(SystemExit, match="sample-fraction"):
        _sample_examples(
            [DatasetExample(sample_id="sample")],
            fraction=fraction,
            seed=0,
        )
