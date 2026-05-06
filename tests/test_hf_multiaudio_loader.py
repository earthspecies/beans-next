"""Unit tests for HuggingFace-backed multi-audio dataset normalization."""

from __future__ import annotations

from typing import Any

import numpy as np

from beans_next.api.types import DatasetExample
from beans_next.datasets.base import ensure_audio_paths_from_sequence
from beans_next.datasets.hf_multiaudio import (
    _strip_audio_placeholders_except_last,
    beans_next_multiaudio_row_filter,
)
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER
from beans_next.prompts.renderer import PromptRenderer, load_builtin_prompt_yaml


def test_ensure_audio_paths_from_sequence_materializes_wavs(
    monkeypatch: Any, tmp_path: Any
) -> None:
    monkeypatch.setenv("BEANS_PRO_HF_AUDIO_CACHE_DIR", str(tmp_path))
    audio_val = [
        {"array": np.zeros(160, dtype=np.float32), "sampling_rate": 16000, "path": ""},
        {"array": np.ones(240, dtype=np.float32), "sampling_rate": 16000, "path": ""},
    ]
    paths = ensure_audio_paths_from_sequence(audio_val, sample_id="hf:multi:0")
    assert paths is not None
    assert len(paths) == 2
    assert all(str(p).endswith(".wav") for p in paths)


def test_multiaudio_prompt_alignment_with_passthrough_prompt(
    monkeypatch: Any, tmp_path: Any
) -> None:
    monkeypatch.setenv("BEANS_PRO_HF_AUDIO_CACHE_DIR", str(tmp_path))
    audio_val = [
        {"array": np.zeros(160, dtype=np.float32), "sampling_rate": 16000, "path": ""},
        {"array": np.ones(240, dtype=np.float32), "sampling_rate": 16000, "path": ""},
    ]
    audio_paths = ensure_audio_paths_from_sequence(audio_val, sample_id="hf:multi:1")
    assert audio_paths is not None

    conversation = (
        f"{AUDIO_PLACEHOLDER}\n"
        "Support A.\n"
        f"{AUDIO_PLACEHOLDER}\n"
        "Query audio.\n"
    )
    ex = DatasetExample(
        sample_id="hf:multi:1",
        split="test",
        labels="A",
        metadata={
            "conversation": conversation,
            "conversation_query_only": _strip_audio_placeholders_except_last(conversation),
            "audio_paths": audio_paths,
            "audio_path": audio_paths[-1],
        },
    )
    spec = load_builtin_prompt_yaml("beans_next_multiaudio_passthrough_v1.yaml")
    req = PromptRenderer(spec).render(ex)
    assert len(req.audio_inputs) == 2


def test_beans_next_multiaudio_row_filter_tier_and_subset() -> None:
    pred = beans_next_multiaudio_row_filter(
        tier="tier_4_in_context",
        subset="crow-4way",
    )
    assert pred({"tier": "tier_4_in_context", "subset": "crow-4way"})
    assert pred({"tier": 4, "subset": "crow-4way"})
    assert pred({"tier": 4, "task": "crow-4way"})
    assert not pred({"tier": "tier_4_in_context", "subset": "zebra-4way"})
    assert not pred({"tier": "tier_1_x", "subset": "crow-4way"})
    assert not pred({"tier": 1, "subset": "crow-4way"})


def test_beans_next_multiaudio_row_filter_task_fallback() -> None:
    pred = beans_next_multiaudio_row_filter(tier="tier_4_in_context", subset="crow-4way")
    assert pred({"tier": "tier_4_in_context", "task": "crow-4way"})
