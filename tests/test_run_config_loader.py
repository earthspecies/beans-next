"""Tests for Increment 10 (I10-A): run-config YAML schema + loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from beans_next.config.run_config import RunConfigError, load_run_config


def _write(tmp_path: Path, *, text: str) -> Path:
    path = tmp_path / "run_config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_suite_and_tasks_are_mutually_exclusive(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        text="""
models:
  - dummy_local_8000
suite: beans_zero_smoke
tasks:
  - beans_zero_esc50
""",
    )
    with pytest.raises(
        RunConfigError, match=r"Expected exactly one of `suite` or `tasks`"
    ):
        load_run_config(path)


def test_missing_suite_and_tasks_is_validation_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        text="""
models:
  - dummy_local_8000
""",
    )
    with pytest.raises(
        RunConfigError, match=r"Expected exactly one of `suite` or `tasks`"
    ):
        load_run_config(path)


def test_models_list_parses_and_plan_is_deterministic(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        text="""
models:
  - dummy_local_8000
  - naturelm_v1_0_local_8000
suite: beans_zero_smoke
data_source: esp_data
limit: 7
output_dir: results/custom
run_id: test-run
""",
    )
    loaded = load_run_config(path)
    assert loaded.config.data_source == "esp_data"

    assert [m.name for m in loaded.models] == [
        "dummy_local_8000",
        "naturelm_v1_0_local_8000",
    ]
    assert loaded.eval_task_ids == [
        "beans_zero_esc50",
        "beans_zero_enabirds",
        "beans_zero_captioning",
    ]
    assert len(loaded.plan) == 2 * 3

    # Ordering: models-major, tasks-minor.
    assert (loaded.plan[0].model.name, loaded.plan[0].eval_task_id) == (
        "dummy_local_8000",
        "beans_zero_esc50",
    )
    assert (loaded.plan[1].model.name, loaded.plan[1].eval_task_id) == (
        "dummy_local_8000",
        "beans_zero_enabirds",
    )
    assert (loaded.plan[2].model.name, loaded.plan[2].eval_task_id) == (
        "dummy_local_8000",
        "beans_zero_captioning",
    )
    assert (loaded.plan[3].model.name, loaded.plan[3].eval_task_id) == (
        "naturelm_v1_0_local_8000",
        "beans_zero_esc50",
    )


def test_inline_model_config_is_supported(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        text="""
models:
  - id: dummy_local_8000
  - inline:
      name: inline_model
      predict_url: http://localhost:9999/predict
      info_url: http://localhost:9999/info
      health_url: http://localhost:9999/health
      response_schema: predictions_v1
      audio_payload: base64_wav
      auth: {}
      retry_policy:
        max_attempts: 2
        backoff_initial: 0.0
        backoff_max: 0.0
        jitter_fraction: 0.0
tasks:
  - beans_zero_esc50
  - beans_zero_captioning
""",
    )
    loaded = load_run_config(path)
    assert [m.name for m in loaded.models] == ["dummy_local_8000", "inline_model"]
    assert loaded.models[1].retry_policy == {
        "max_attempts": 2,
        "backoff_initial": 0.0,
        "backoff_max": 0.0,
        "jitter_fraction": 0.0,
    }
    assert loaded.eval_task_ids == ["beans_zero_esc50", "beans_zero_captioning"]
    assert len(loaded.plan) == 2 * 2


def test_unknown_suite_id_is_a_clear_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        text="""
models:
  - dummy_local_8000
suite: does_not_exist
""",
    )
    with pytest.raises(RunConfigError, match=r"Unknown suite id: does_not_exist"):
        load_run_config(path)
