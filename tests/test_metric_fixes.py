"""Tests for metric gap fixes: Levenshtein fuzzy matching and top1_accuracy."""

from __future__ import annotations

import pytest

from beans_next.api.types import DatasetExample
from beans_next.metrics import score_sample
from beans_next.metrics.classification import top1_accuracy
from beans_next.post_process.cleaners import (
    _best_label_by_distance,
    apply_fuzzy_match_to_labels,
)
from beans_next.post_process.pipeline import (
    PostProcessContext,
    PostProcessResult,
)

# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


class TestBestLabelByDistance:
    """Unit tests for label selection by minimum edit distance."""

    def test_tie_breaks_lexicographically(self) -> None:
        label, _ = _best_label_by_distance("x", ["b", "a"])
        assert label == "a"


class TestApplyFuzzyMatchToLabels:
    """Tests for the updated Levenshtein-based fuzzy match cleaner."""

    def _ctx(self, *segments: str) -> PostProcessContext:
        return PostProcessContext(segments=list(segments))

    def test_warning_added_on_threshold_reject(self) -> None:
        ctx = self._ctx("zzzzzzzzz")
        out = apply_fuzzy_match_to_labels(
            ctx,
            labels=["cat"],
            apply_threshold=True,
            max_distance=5,
        )
        assert any("max_distance" in w for w in out.warnings)


# ---------------------------------------------------------------------------
# top1_accuracy scorer
# ---------------------------------------------------------------------------


class TestTop1Accuracy:
    """Tests for the top1_accuracy scorer with multi-reference targets."""

    def test_multi_reference_second_option(self) -> None:
        assert top1_accuracy(["feline"], ["cat, feline"]) == 1.0

    def test_batch_accuracy(self) -> None:
        preds = ["cat", "dog", "bird"]
        targets = ["cat", "dog, canine", "fish"]
        score = top1_accuracy(preds, targets)
        assert score == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# per-dataset label registry
# ---------------------------------------------------------------------------


class TestBeanszeroLabelRegistry:
    """Tests for the beans_zero_labels.json registry loaded by the runner."""

    def test_inline_labels_override_registry(self) -> None:
        from beans_next.runner.runner import _labels_for_eval_task

        result = _labels_for_eval_task(
            {"subset": "esc50", "labels": ["custom_a", "custom_b"]}
        )
        assert result == ["custom_a", "custom_b"]


# ---------------------------------------------------------------------------
# score_sample routing
# ---------------------------------------------------------------------------


class TestScoreSampleRouting:
    """Tests for score_sample task-routing between top1_accuracy and AP."""

    def _post(self, text: str = "") -> PostProcessResult:
        return PostProcessResult(segments=[], text=text, warnings=())

    def _example(self, labels: object, *, task: str | None = None) -> DatasetExample:
        meta: dict[str, object] = {"task": task} if task is not None else {}
        return DatasetExample(sample_id="s0", labels=labels, metadata=meta)

    def test_classification_list_correct(self) -> None:
        ex = self._example(["cat", "feline"], task="classification")
        result = score_sample(ex, post=self._post("cat"), raw_predictions=["cat"])
        assert "top1_accuracy" in result
        assert result["top1_accuracy"] == pytest.approx(1.0)

    def test_detection_list_returns_average_precision(self) -> None:
        ex = self._example(["cat", "dog"], task="detection")
        result = score_sample(ex, post=self._post("cat"), raw_predictions=["cat"])
        assert "average_precision" in result
        assert "top1_accuracy" not in result
