"""Check that the shipped HF suites form a usable evaluation registry."""

from pathlib import Path

import pytest

from beans_next.config.eval_task import EvalTaskConfig
from beans_next.config.run_config import _load_eval_task, _load_suite_eval_tasks
from beans_next.prompts.renderer import load_builtin_prompt_yaml

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
