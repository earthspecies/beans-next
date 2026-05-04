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
