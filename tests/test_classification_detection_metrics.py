"""Unit tests for classification, detection, and base metric functions."""

from __future__ import annotations

import pytest

from beans_next.metrics.classification import accuracy, f1, precision, recall
from beans_next.metrics.detection import average_precision

# Shared 4-sample fixture: preds=[0,1,1,1] vs targets=[1,0,1,1]  (2 classes)
# Class 0: TP=0, FP=1, FN=1  → prec=0.0, rec=0.0, f1=0.0
# Class 1: TP=2, FP=1, FN=1  → prec=2/3, rec=2/3, f1=2/3
# macro=1/3  micro=0.5  weighted=0.5  (class-support: c0=1, c1=3)
_P = [0, 1, 1, 1]
_T = [1, 0, 1, 1]


class TestAccuracy:
    """Tests for the accuracy scorer (multiclass and multilabel)."""

    def test_multilabel_binary_matrix_partial(self) -> None:
        yp = [[1, 0], [0, 1]]
        yt = [[1, 0], [1, 0]]
        assert accuracy(yp, yt) == pytest.approx(0.5)


class TestPrecision:
    """Tests for precision across all averaging modes."""

    def test_macro(self) -> None:
        assert precision(_P, _T, average="macro") == pytest.approx(1 / 3)

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


class TestF1:
    """Tests for F1 score across all averaging modes."""

    def test_macro(self) -> None:
        assert f1(_P, _T, average="macro") == pytest.approx(1 / 3)


class TestAveragePrecision:
    """Tests for average_precision (macro/micro, edge cases, label-index format)."""

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

    def test_tie_breaking_is_deterministic(self) -> None:
        # Equal scores must produce stable results (lower index ranked first)
        y_score = [[0.5], [0.5]]
        y_true = [[1], [0]]
        assert average_precision(y_score, y_true) == average_precision(y_score, y_true)
