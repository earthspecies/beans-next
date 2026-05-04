"""Tests for EvalTaskConfig, load_judge_preset, and related runner helpers."""

from __future__ import annotations

import pytest

from beans_next.config.eval_task import (
    EvalTaskConfig,
    EvalTaskConfigError,
    JudgeRegistryResolutionError,
    load_judge_preset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL = {
    "task_type": "classification",
    "hf_path": "org/dataset",
    "subset": "sub",
    "split": "test",
    "prompt": "some_prompt",
    "metrics": [{"name": "accuracy"}],
}


def _task(**overrides: object) -> dict[str, object]:
    return {**_MINIMAL, **overrides}


# ---------------------------------------------------------------------------
# load_judge_preset
# ---------------------------------------------------------------------------


class TestLoadJudgePreset:
    """Tests for the bundled judge registry loader."""

    def test_known_id_returns_judge_config(self) -> None:
        cfg = load_judge_preset("bioacoustic_open_qa_v1")
        assert cfg.template_id == "bioacoustic_open_qa_v1"

    def test_counting_preset_available(self) -> None:
        cfg = load_judge_preset("bioacoustic_counting_v1")
        assert cfg.template_id == "bioacoustic_counting_v1"

    def test_unknown_id_raises(self) -> None:
        with pytest.raises(JudgeRegistryResolutionError, match="Unknown judge id"):
            load_judge_preset("__no_such_judge_xyz__")


# ---------------------------------------------------------------------------
# EvalTaskConfig — judge field variations
# ---------------------------------------------------------------------------


class TestEvalTaskConfigJudge:
    """Tests for the judge field resolution in EvalTaskConfig."""

    def test_no_judge_field_is_valid(self) -> None:
        cfg = EvalTaskConfig.model_validate(_task())
        assert cfg.judge is None

    def test_judge_as_registry_string(self) -> None:
        cfg = EvalTaskConfig.model_validate(_task(judge="bioacoustic_counting_v1"))
        assert cfg.judge is not None
        assert cfg.judge.template_id == "bioacoustic_counting_v1"

    def test_judge_as_mapping_with_id(self) -> None:
        cfg = EvalTaskConfig.model_validate(
            _task(judge={"id": "bioacoustic_open_qa_v1"})
        )
        assert cfg.judge is not None
        assert cfg.judge.template_id == "bioacoustic_open_qa_v1"

    def test_judge_inline_template_id(self) -> None:
        cfg = EvalTaskConfig.model_validate(
            _task(judge={"template_id": "bioacoustic_counting_v1"})
        )
        assert cfg.judge is not None
        assert cfg.judge.template_id == "bioacoustic_counting_v1"

    def test_judge_inline_block_syntax(self) -> None:
        cfg = EvalTaskConfig.model_validate(
            _task(
                judge={
                    "inline": {
                        "template_id": "bioacoustic_open_qa_v1",
                        "description": "test",
                    }
                }
            )
        )
        assert cfg.judge is not None
        assert cfg.judge.template_id == "bioacoustic_open_qa_v1"

    def test_unknown_registry_id_raises(self) -> None:
        # Pydantic wraps validator ValueError in ValidationError
        from pydantic import ValidationError

        with pytest.raises(
            (EvalTaskConfigError, JudgeRegistryResolutionError, ValidationError)
        ):
            EvalTaskConfig.model_validate(_task(judge="__nonexistent_judge__"))

    def test_judge_integer_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises((EvalTaskConfigError, ValidationError)):
            EvalTaskConfig.model_validate(_task(judge=42))


# ---------------------------------------------------------------------------
# _postprocess_steps_for_examples — open-ended task types
# ---------------------------------------------------------------------------


class TestPostprocessStepsOpenEnded:
    """Tests for open-ended task types skipping label parsing."""

    def _steps(self, task_type: str | None) -> tuple[tuple, tuple]:
        from beans_next.api.types import DatasetExample
        from beans_next.runner.runner import _postprocess_steps_for_examples

        examples = [DatasetExample(sample_id="s1", labels="ref text")]
        return _postprocess_steps_for_examples(examples, task_type=task_type)

    def _names(self, steps: tuple) -> list[str]:
        return [s.name for s in steps]

    def test_open_ended_no_parser_no_fuzzy(self) -> None:
        parsers, cleaners = self._steps("open_ended")
        assert parsers == ()
        assert "parse_labels_comma" not in self._names(cleaners)
        assert "fuzzy_match_to_labels" not in self._names(cleaners)

    def test_counting_no_parser_no_fuzzy(self) -> None:
        parsers, cleaners = self._steps("counting")
        assert parsers == ()
        assert "fuzzy_match_to_labels" not in self._names(cleaners)

    def test_qa_no_parser_no_fuzzy(self) -> None:
        parsers, cleaners = self._steps("qa")
        assert parsers == ()
        assert "fuzzy_match_to_labels" not in self._names(cleaners)

    def test_open_ended_still_runs_normalize_whitespace(self) -> None:
        _, cleaners = self._steps("open_ended")
        assert "normalize_whitespace" in self._names(cleaners)


# ---------------------------------------------------------------------------
# _judge_from_args_and_task
# ---------------------------------------------------------------------------


class TestJudgeFromArgsAndTask:
    """Tests for judge scorer construction from CLI args and eval-task config."""

    def _call(
        self,
        judge_url: str | None,
        task: dict[str, object],
    ) -> object:
        from argparse import Namespace

        from beans_next.runner.runner import _judge_from_args_and_task

        args = Namespace(judge_url=judge_url)
        return _judge_from_args_and_task(args, task)

    def test_no_judge_url_returns_none(self) -> None:
        assert self._call(None, {}) is None

    def test_empty_judge_url_returns_none(self) -> None:
        assert self._call("  ", {}) is None

    def test_with_judge_url_no_task_judge_uses_default_template(self) -> None:
        scorer = self._call("http://judge.local/judge", {})
        assert scorer is not None
        assert scorer.template_id == "bioacoustic_open_qa_v1"

    def test_task_judge_template_id_overrides_default(self) -> None:
        scorer = self._call(
            "http://judge.local/judge",
            {"judge": {"template_id": "bioacoustic_counting_v1"}},
        )
        assert scorer is not None
        assert scorer.template_id == "bioacoustic_counting_v1"
