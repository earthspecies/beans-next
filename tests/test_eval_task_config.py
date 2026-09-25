"""Tests for task-aware post-processing."""

from __future__ import annotations

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
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


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
# ---------------------------------------------------------------------------
