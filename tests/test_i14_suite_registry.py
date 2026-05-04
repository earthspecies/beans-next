"""I14 registry smoke: suite expands and tasks resolve official prompt."""

from __future__ import annotations

from beans_next.config.run_config import _load_eval_task, _load_suite_eval_tasks
from beans_next.prompts import load_builtin_prompt_yaml


def test_i14_suite_registry_expands_to_expected_eval_tasks() -> None:
    assert _load_suite_eval_tasks("beans_zero_watkins_lifestage_calltype") == [
        "beans_zero_watkins",
        "beans_zero_lifestage",
        "beans_zero_call_type",
    ]


def test_i14_eval_tasks_resolve_official_prompt_and_metrics() -> None:
    official = load_builtin_prompt_yaml("classification_beans_zero_official_v1.yaml")
    assert (
        official.prompt_id == "beans_next.prompt.classification_beans_zero_official.v1"
    )

    for eval_task_id in (
        "beans_zero_watkins",
        "beans_zero_lifestage",
        "beans_zero_call_type",
    ):
        cfg = _load_eval_task(eval_task_id)
        assert cfg["task_type"] == "classification"
        assert cfg["prompt"] == "classification_beans_zero_official_v1"

        metrics = cfg["metrics"]
        assert [m["name"] for m in metrics] == [
            "accuracy",
            "f1",
            "recall",
            "precision",
        ]

        for metric in metrics[1:]:
            assert metric["scorer_kwargs"] == {"average": "macro", "zero_division": 0}
