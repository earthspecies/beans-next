"""Check that the shipped HF suites form a usable evaluation registry."""

from argparse import Namespace
from pathlib import Path

import pytest

from beans_next.api.types import DatasetExample
from beans_next.config.eval_task import EvalTaskConfig
from beans_next.config.run_config import _load_eval_task, _load_suite_eval_tasks
from beans_next.prompts.renderer import PromptRenderer, load_builtin_prompt_yaml
from beans_next.runner.runner import _prompt_spec_from_eval_task

_REGISTRY = Path(__file__).parents[1] / "beans_next" / "registry"


@pytest.mark.parametrize(
    "suite", sorted(p.stem for p in (_REGISTRY / "suite").glob("*.yaml"))
)
def test_canonical_suite_tasks_resolve(suite: str) -> None:
    tasks = _load_suite_eval_tasks(suite)
    assert tasks and len(tasks) == len(set(tasks))
    for task in tasks:
        raw = _load_eval_task(task)
        config = EvalTaskConfig.model_validate(raw)
        if raw.get("dataset") == "birdset":
            assert config.split.endswith("-test_5s")
        else:
            assert config.split == "test"
        assert "data_source" not in raw
        assert load_builtin_prompt_yaml(config.prompt + ".yaml").prompt_id


def test_all_tiers_contains_exactly_the_individual_suites() -> None:
    expected = [
        task
        for tier in range(1, 5)
        for task in _load_suite_eval_tasks(f"beans_next_tier{tier}")
    ]
    assert _load_suite_eval_tasks("beans_next_all_tiers") == expected


def test_structural_captioning_preserves_dataset_question() -> None:
    """The captioning task must ask the row's question with its token budget."""
    task = _load_eval_task("beans_next_t3_structural_captioning")
    spec = _prompt_spec_from_eval_task(task, args=Namespace(prompt_yaml=None))
    question = "<Audio><AudioHere></Audio>\nDescribe the pattern of vocalizations."
    example = DatasetExample(
        sample_id="caption-example",
        task_id="beans_next_t3_structural_captioning",
        metadata={"audio_path": "clip.wav", "instruction": question},
    )
    request = PromptRenderer(spec).render(example)
    assert request.messages[0].content == question
    assert request.generation_config.max_tokens == 256
    assert request.generation_config.max_length_seconds == 30
