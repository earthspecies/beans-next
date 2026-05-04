"""Tests for metric gap fixes: Levenshtein fuzzy matching and top1_accuracy."""

from __future__ import annotations

import pytest

from beans_next.api.types import DatasetExample
from beans_next.metrics import score_sample
from beans_next.metrics.classification import top1_accuracy
from beans_next.post_process.cleaners import (
    _best_label_by_distance,
    _levenshtein_distance,
    apply_fuzzy_match_to_labels,
)
from beans_next.post_process.pipeline import (
    PostProcessContext,
    PostProcessPipelineError,
    PostProcessResult,
)

# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


class TestLevenshteinDistance:
    """Unit tests for the pure-Python Levenshtein distance helper."""

    def test_identical(self) -> None:
        assert _levenshtein_distance("cat", "cat") == 0

    def test_empty_strings(self) -> None:
        assert _levenshtein_distance("", "") == 0

    def test_one_empty(self) -> None:
        assert _levenshtein_distance("cat", "") == 3
        assert _levenshtein_distance("", "cat") == 3

    def test_insertion(self) -> None:
        assert _levenshtein_distance("cat", "cats") == 1

    def test_deletion(self) -> None:
        assert _levenshtein_distance("cats", "cat") == 1

    def test_substitution(self) -> None:
        assert _levenshtein_distance("cat", "bat") == 1

    def test_completely_different(self) -> None:
        assert _levenshtein_distance("abc", "xyz") == 3

    def test_symmetry(self) -> None:
        assert _levenshtein_distance("kitten", "sitting") == _levenshtein_distance(
            "sitting", "kitten"
        )

    def test_known_case(self) -> None:
        assert _levenshtein_distance("kitten", "sitting") == 3


class TestBestLabelByDistance:
    """Unit tests for label selection by minimum edit distance."""

    def test_exact_match(self) -> None:
        label, dist = _best_label_by_distance("cat", ["cat", "dog", "bird"])
        assert label == "cat"
        assert dist == 0

    def test_nearest(self) -> None:
        label, dist = _best_label_by_distance("kat", ["cat", "dog", "bird"])
        assert label == "cat"
        assert dist == 1

    def test_tie_breaks_lexicographically(self) -> None:
        label, _ = _best_label_by_distance("x", ["b", "a"])
        assert label == "a"

    def test_deduplication(self) -> None:
        label, dist = _best_label_by_distance("cat", ["cat", "cat", "dog"])
        assert label == "cat"
        assert dist == 0


class TestApplyFuzzyMatchToLabels:
    """Tests for the updated Levenshtein-based fuzzy match cleaner."""

    def _ctx(self, *segments: str) -> PostProcessContext:
        return PostProcessContext(segments=list(segments))

    def test_exact_match(self) -> None:
        ctx = self._ctx("cat")
        out = apply_fuzzy_match_to_labels(ctx, labels=["cat", "dog"])
        assert out.segments == ["cat"]

    def test_near_match(self) -> None:
        ctx = self._ctx("kat")
        out = apply_fuzzy_match_to_labels(ctx, labels=["cat", "dog"])
        assert out.segments == ["cat"]

    def test_threshold_reject(self) -> None:
        ctx = self._ctx("zzzzzzzzz")
        out = apply_fuzzy_match_to_labels(
            ctx,
            labels=["cat"],
            apply_threshold=True,
            max_distance=5,
            default_label="None",
        )
        assert out.segments == ["None"]

    def test_threshold_accept(self) -> None:
        ctx = self._ctx("kat")
        out = apply_fuzzy_match_to_labels(
            ctx,
            labels=["cat"],
            apply_threshold=True,
            max_distance=5,
        )
        assert out.segments == ["cat"]

    def test_empty_labels_raises(self) -> None:
        ctx = self._ctx("cat")
        with pytest.raises(PostProcessPipelineError):
            apply_fuzzy_match_to_labels(ctx, labels=[])

    def test_negative_max_distance_raises(self) -> None:
        ctx = self._ctx("cat")
        with pytest.raises(PostProcessPipelineError):
            apply_fuzzy_match_to_labels(ctx, labels=["cat"], max_distance=-1)

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

    def test_single_reference_correct(self) -> None:
        assert top1_accuracy(["cat"], ["cat"]) == 1.0

    def test_single_reference_wrong(self) -> None:
        assert top1_accuracy(["dog"], ["cat"]) == 0.0

    def test_multi_reference_first_option(self) -> None:
        assert top1_accuracy(["cat"], ["cat, feline"]) == 1.0

    def test_multi_reference_second_option(self) -> None:
        assert top1_accuracy(["feline"], ["cat, feline"]) == 1.0

    def test_multi_reference_neither(self) -> None:
        assert top1_accuracy(["dog"], ["cat, feline"]) == 0.0

    def test_batch_accuracy(self) -> None:
        preds = ["cat", "dog", "bird"]
        targets = ["cat", "dog, canine", "fish"]
        score = top1_accuracy(preds, targets)
        assert score == pytest.approx(2 / 3)

    def test_perfect_score(self) -> None:
        preds = ["a", "b", "c"]
        targets = ["a", "b, B", "c"]
        assert top1_accuracy(preds, targets) == 1.0

    def test_length_mismatch_raises(self) -> None:
        from beans_next.metrics.base import MetricsError

        with pytest.raises(MetricsError):
            top1_accuracy(["a"], ["a", "b"])

    def test_empty_raises(self) -> None:
        from beans_next.metrics.base import MetricsError

        with pytest.raises(MetricsError):
            top1_accuracy([], [])

    def test_custom_separator(self) -> None:
        assert top1_accuracy(["cat"], ["cat|feline"], label_separator="|") == 1.0
        assert top1_accuracy(["feline"], ["cat|feline"], label_separator="|") == 1.0


# ---------------------------------------------------------------------------
# per-dataset label registry
# ---------------------------------------------------------------------------


class TestBeanszeroLabelRegistry:
    """Tests for the beans_zero_labels.json registry loaded by the runner."""

    def test_registry_loads_known_subset(self) -> None:
        from beans_next.runner.runner import _labels_for_eval_task

        labels = _labels_for_eval_task({"subset": "esc50"})
        assert isinstance(labels, list)
        assert len(labels) == 50
        assert "dog" in labels

    def test_registry_unknown_subset_returns_none(self) -> None:
        from beans_next.runner.runner import _labels_for_eval_task

        result = _labels_for_eval_task({"subset": "nonexistent_dataset_xyz"})
        assert result is None

    def test_registry_no_subset_returns_none(self) -> None:
        from beans_next.runner.runner import _labels_for_eval_task

        result = _labels_for_eval_task({})
        assert result is None

    def test_inline_labels_override_registry(self) -> None:
        from beans_next.runner.runner import _labels_for_eval_task

        result = _labels_for_eval_task(
            {"subset": "esc50", "labels": ["custom_a", "custom_b"]}
        )
        assert result == ["custom_a", "custom_b"]

    def test_unseen_species_cmn_labels(self) -> None:
        from beans_next.runner.runner import _labels_for_eval_task

        labels = _labels_for_eval_task({"subset": "unseen-species-cmn"})
        assert isinstance(labels, list)
        assert len(labels) == 202


# ---------------------------------------------------------------------------
# score_sample routing
# ---------------------------------------------------------------------------


class TestScoreSampleRouting:
    """Tests for score_sample task-routing between top1_accuracy and AP."""

    def _post(self, text: str = "") -> PostProcessResult:
        return PostProcessResult(segments=[], text=text, warnings=())

    def _example(
        self, labels: object, *, task: str | None = None
    ) -> DatasetExample:
        meta: dict[str, object] = {"task": task} if task is not None else {}
        return DatasetExample(sample_id="s0", labels=labels, metadata=meta)

    def test_classification_list_correct(self) -> None:
        ex = self._example(["cat", "feline"], task="classification")
        result = score_sample(ex, post=self._post("cat"), raw_predictions=["cat"])
        assert "top1_accuracy" in result
        assert result["top1_accuracy"] == pytest.approx(1.0)

    def test_classification_list_wrong(self) -> None:
        ex = self._example(["cat", "feline"], task="classification")
        result = score_sample(ex, post=self._post("dog"), raw_predictions=["dog"])
        assert "top1_accuracy" in result
        assert result["top1_accuracy"] == pytest.approx(0.0)

    def test_detection_list_returns_average_precision(self) -> None:
        ex = self._example(["cat", "dog"], task="detection")
        result = score_sample(ex, post=self._post("cat"), raw_predictions=["cat"])
        assert "average_precision" in result
        assert "top1_accuracy" not in result

    def test_unknown_task_list_falls_back_to_average_precision(self) -> None:
        ex = self._example(["cat", "dog"])
        result = score_sample(ex, post=self._post("cat"), raw_predictions=["cat"])
        assert "average_precision" in result
        assert "top1_accuracy" not in result
