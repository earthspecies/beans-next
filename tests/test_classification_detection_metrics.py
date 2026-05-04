"""Unit tests for classification, detection, and base metric functions."""

from __future__ import annotations

import pytest

from beans_next.metrics.base import (
    MetricsError,
    get_scorer,
    list_scorers,
    validate_equal_length,
)
from beans_next.metrics.classification import accuracy, f1, precision, recall
from beans_next.metrics.detection import average_precision

# Shared 4-sample fixture: preds=[0,1,1,1] vs targets=[1,0,1,1]  (2 classes)
# Class 0: TP=0, FP=1, FN=1  → prec=0.0, rec=0.0, f1=0.0
# Class 1: TP=2, FP=1, FN=1  → prec=2/3, rec=2/3, f1=2/3
# macro=1/3  micro=0.5  weighted=0.5  (class-support: c0=1, c1=3)
_P = [0, 1, 1, 1]
_T = [1, 0, 1, 1]


class TestValidateEqualLength:
    """Tests for validate_equal_length helper."""

    def test_empty_predictions_raises(self) -> None:
        with pytest.raises(MetricsError):
            validate_equal_length([], [1])

    def test_empty_targets_raises(self) -> None:
        with pytest.raises(MetricsError):
            validate_equal_length([1], [])

    def test_length_mismatch_message(self) -> None:
        with pytest.raises(MetricsError, match="2 vs 3"):
            validate_equal_length([1, 2], [1, 2, 3])

    def test_equal_length_passes(self) -> None:
        validate_equal_length([1, 2], [3, 4])  # no exception


class TestRegistry:
    """Tests for the scorer registry (register_scorer, get_scorer, list_scorers)."""

    def test_known_scorer_is_callable(self) -> None:
        assert callable(get_scorer("accuracy"))

    def test_unknown_scorer_raises_lookup_error(self) -> None:
        with pytest.raises(LookupError):
            get_scorer("__no_such_scorer__")

    def test_list_scorers_sorted_and_includes_core(self) -> None:
        names = list_scorers()
        assert names == sorted(names)
        assert {
            "accuracy",
            "cider",
            "f1",
            "precision",
            "recall",
            "average_precision",
            "spider",
        }.issubset(set(names))

    def test_duplicate_registration_raises(self) -> None:
        from beans_next.metrics.base import register_scorer

        def _dup() -> float:  # type: ignore[return]
            ...

        _dup.__name__ = "accuracy"
        with pytest.raises(MetricsError):
            register_scorer(_dup)


class TestAccuracy:
    """Tests for the accuracy scorer (multiclass and multilabel)."""

    def test_multiclass_perfect(self) -> None:
        assert accuracy([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_multiclass_partial(self) -> None:
        assert accuracy([0, 1], [0, 2]) == pytest.approx(0.5)

    def test_multilabel_binary_matrix_exact_match(self) -> None:
        yp = [[1, 0], [0, 1]]
        yt = [[1, 0], [0, 1]]
        assert accuracy(yp, yt) == pytest.approx(1.0)

    def test_multilabel_binary_matrix_partial(self) -> None:
        yp = [[1, 0], [0, 1]]
        yt = [[1, 0], [1, 0]]
        assert accuracy(yp, yt) == pytest.approx(0.5)


class TestPrecision:
    """Tests for precision across all averaging modes."""

    def test_macro(self) -> None:
        assert precision(_P, _T, average="macro") == pytest.approx(1 / 3)

    def test_micro(self) -> None:
        assert precision(_P, _T, average="micro") == pytest.approx(0.5)

    def test_weighted(self) -> None:
        assert precision(_P, _T, average="weighted") == pytest.approx(0.5)

    def test_binary(self) -> None:
        # TP=2, FP=1 → prec=2/3
        assert precision([1, 0, 1, 1], [1, 1, 0, 1], average="binary") == pytest.approx(
            2 / 3
        )

    def test_unsupported_average_raises(self) -> None:
        with pytest.raises(MetricsError, match="Unsupported"):
            precision(_P, _T, average="samples")

    def test_zero_division_no_positive_predictions(self) -> None:
        # TP+FP=0 → precision undefined → zero_division applied
        assert precision(
            [0, 0], [0, 1], average="binary", zero_division=0
        ) == pytest.approx(0.0)
        assert precision(
            [0, 0], [0, 1], average="binary", zero_division=1
        ) == pytest.approx(1.0)


class TestRecall:
    """Tests for recall across all averaging modes."""

    def test_macro(self) -> None:
        assert recall(_P, _T, average="macro") == pytest.approx(1 / 3)

    def test_micro(self) -> None:
        assert recall(_P, _T, average="micro") == pytest.approx(0.5)

    def test_weighted(self) -> None:
        assert recall(_P, _T, average="weighted") == pytest.approx(0.5)

    def test_binary(self) -> None:
        # TP=2, FN=1 → rec=2/3
        assert recall([1, 0, 1, 1], [1, 1, 0, 1], average="binary") == pytest.approx(
            2 / 3
        )


class TestF1:
    """Tests for F1 score across all averaging modes."""

    def test_macro(self) -> None:
        assert f1(_P, _T, average="macro") == pytest.approx(1 / 3)

    def test_micro(self) -> None:
        assert f1(_P, _T, average="micro") == pytest.approx(0.5)

    def test_weighted(self) -> None:
        assert f1(_P, _T, average="weighted") == pytest.approx(0.5)

    def test_binary(self) -> None:
        # prec=2/3, rec=2/3 → f1=2/3
        assert f1([1, 0, 1, 1], [1, 1, 0, 1], average="binary") == pytest.approx(2 / 3)


class TestAveragePrecision:
    """Tests for average_precision (macro/micro, edge cases, label-index format)."""

    def test_perfect_macro(self) -> None:
        y_score = [[0.9, 0.1], [0.1, 0.8]]
        y_true = [[1, 0], [0, 1]]
        assert average_precision(y_score, y_true, average="macro") == pytest.approx(1.0)

    def test_no_positives_in_column_returns_zero(self) -> None:
        y_score = [[0.9], [0.1]]
        y_true = [[0], [0]]
        assert average_precision(y_score, y_true, average="macro") == pytest.approx(0.0)

    def test_micro_vs_macro_differ(self) -> None:
        # Col 0: 1 positive, highest score → col AP=1.0
        # Col 1: no positives → col AP=0.0
        # macro=0.5, micro pools labels so all positives are ranked first → micro=1.0
        y_score = [[0.9, 0.5], [0.1, 0.3]]
        y_true = [[1, 0], [0, 0]]
        assert average_precision(y_score, y_true, average="macro") == pytest.approx(0.5)
        assert average_precision(y_score, y_true, average="micro") == pytest.approx(1.0)

    def test_label_index_targets(self) -> None:
        # Ragged label-index rows (value ≥ 2 prevents binary-matrix interpretation)
        # 3 classes; sample 0 has label 0, sample 1 has label 2
        # binary equiv: [[1,0,0],[0,0,1]]  →  col0 AP=1.0, col1 AP=0.0, col2 AP=1.0
        y_score = [[0.9, 0.2, 0.1], [0.1, 0.8, 0.7]]
        y_true_idx = [[0], [2]]
        assert average_precision(y_score, y_true_idx, average="macro") == pytest.approx(
            2 / 3
        )

    def test_flat_predictions_raises(self) -> None:
        with pytest.raises(MetricsError):
            average_precision([0.9, 0.1], [1, 0])

    def test_unsupported_average_raises(self) -> None:
        with pytest.raises(MetricsError, match="Unsupported"):
            average_precision([[0.9]], [[1]], average="weighted")

    def test_tie_breaking_is_deterministic(self) -> None:
        # Equal scores must produce stable results (lower index ranked first)
        y_score = [[0.5], [0.5]]
        y_true = [[1], [0]]
        assert average_precision(y_score, y_true) == average_precision(y_score, y_true)
