"""Verify dataset revision precedence and task selection."""

from argparse import Namespace
from collections.abc import Iterator
from typing import Any

import pytest

from beans_next.api.types import DatasetExample
from beans_next.datasets import beans_next_hub
from beans_next.runner.runner import _load_examples_for_eval_task


@pytest.mark.parametrize(
    ("cli_revision", "env_revision", "task_revision", "expected"),
    [
        ("cli-pin", "env-pin", "task-pin", "cli-pin"),
        (None, "env-pin", "task-pin", "env-pin"),
        (None, "  ", "task-pin", "task-pin"),
        (None, "", None, "main"),
    ],
)
def test_hf_revision_and_subset_survive_modality_selection(
    monkeypatch: pytest.MonkeyPatch,
    cli_revision: str | None,
    env_revision: str,
    task_revision: str | None,
    expected: str,
) -> None:
    """Preserve task mapping and the selected dataset revision."""
    captured: dict[str, Any] = {}

    def load(repo_id: str, **kwargs: Any) -> Iterator[DatasetExample]:
        captured.update(repo_id=repo_id, **kwargs)
        yield DatasetExample(sample_id="fixture")

    monkeypatch.setenv("BEANS_NEXT_HF_REVISION", env_revision)
    monkeypatch.setattr(beans_next_hub, "iter_hf_beans_next_examples", load)
    rows = _load_examples_for_eval_task(
        {
            "dataset": "beans_next",
            "subset": "v20260823-alarm-call-presence",
            "hf_subset": "alarm-call-presence",
            "revision": task_revision,
        },
        args=Namespace(
            data_source="huggingface",
            hf_revision=cli_revision,
            modality_mode="audio",
            limit=1,
        ),
    )
    assert [row.sample_id for row in rows] == ["fixture"]
    assert captured["revision"] == expected
    assert captured["subset"] == "alarm-call-presence"
    assert captured["load_audio"] is True
