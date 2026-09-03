"""Tests for eval-task-level `generation_config` overrides.

The benchmark tiers pin their audio length policy in the eval task so a suite
reproduces its published numbers without submit-time environment variables.
"""

from argparse import Namespace
from typing import Any

import pytest

from beans_next.prompts.renderer import PromptSpec
from beans_next.runner.runner import _prompt_spec_from_eval_task

_BASE = {
    "task_type": "captioning",
    "data_source": "esp_data",
    "dataset": "beans_next",
    "prompt": "classification_beans_zero_official_v1",
}


def _spec(eval_task: dict[str, Any]) -> PromptSpec:
    return _prompt_spec_from_eval_task(eval_task, args=Namespace(prompt_yaml=None))


def test_eval_task_generation_config_sets_max_length_seconds() -> None:
    spec = _spec({**_BASE, "generation_config": {"max_length_seconds": 10}})
    assert spec.generation_config.max_length_seconds == 10


def test_absent_generation_config_leaves_prompt_default() -> None:
    assert _spec(dict(_BASE)).generation_config.max_length_seconds is None


def test_eval_task_generation_config_wins_over_beans_zero_subset_hint() -> None:
    """An explicit task cap must not be overwritten by the BEANS-Zero hint."""
    spec = _spec(
        {
            **_BASE,
            "hf_path": "EarthSpeciesProject/BEANS-Zero",
            "subset": "dcase",
            "generation_config": {"max_length_seconds": 30},
        }
    )
    assert spec.generation_config.max_length_seconds == 30


def test_unknown_generation_config_field_raises() -> None:
    with pytest.raises(ValueError, match="Unknown generation_config field"):
        _spec({**_BASE, "generation_config": {"not_a_real_field": 1}})


def test_empty_generation_config_is_ignored() -> None:
    assert _spec({**_BASE, "generation_config": {}}).generation_config is not None
