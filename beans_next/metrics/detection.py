"""Detection-oriented metrics (multi-label average precision)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from beans_next.metrics.base import MetricsError, register_scorer, validate_equal_length

__all__ = ["average_precision"]


def _as_float_score_matrix(
    predictions: Sequence[Any],
    targets: Sequence[Any],
) -> tuple[list[list[float]], list[list[int]]]:
    """Coerce ``predictions`` to scores and ``targets`` to binary indicators.

    Returns
    -------
    tuple of list of list
        ``(score_matrix, binary_target_matrix)`` with identical ``(n, k)`` shape.

    Raises
    ------
    MetricsError
        If layouts are invalid or cannot be aligned.
    """
    if not predictions or not isinstance(predictions[0], (list, tuple)):
        raise MetricsError(
            "average_precision expects nested predictions (scores or binary rows).",
        )
    if not targets or not isinstance(targets[0], (list, tuple)):
        raise MetricsError(
            "average_precision expects nested targets (binary rows or label indices).",
        )

    pred_rows = [list(r) for r in predictions]
    tgt_rows = [list(r) for r in targets]

    def row_is_binary_ints(row: list[Any]) -> bool:
        return all(isinstance(v, (int, float)) and float(v) in (0.0, 1.0) for v in row)

    tgt_binary = all(row_is_binary_ints(r) for r in tgt_rows)
    lens_t = {len(r) for r in tgt_rows}
    if tgt_binary:
        if len(lens_t) != 1:
            raise MetricsError("Binary target rows must all have the same length.")
        tm = [[int(float(v)) for v in r] for r in tgt_rows]
    else:
        labels: set[int] = set()
        for row in tgt_rows:
            for v in row:
                if isinstance(v, bool) or not isinstance(v, int):
                    raise MetricsError("Label indices in targets must be integers.")
                if v < 0:
                    raise MetricsError("Label indices must be non-negative.")
                labels.add(int(v))
        n_classes = max(labels, default=-1) + 1
        if n_classes <= 0:
            raise MetricsError("Could not infer number of classes from targets.")

        def indices_to_row(indices: list[int]) -> list[int]:
            row = [0] * n_classes
            for i in indices:
                row[i] = 1
            return row

        tm = [indices_to_row([int(x) for x in r]) for r in tgt_rows]

    n_classes = len(tm[0])
    if any(len(r) != n_classes for r in tm):
        raise MetricsError("All target rows must have the same width after coercion.")

    if any(len(r) != n_classes for r in pred_rows):
        raise MetricsError(
            "predictions and targets must have the same number of labels.",
        )

    sm: list[list[float]] = []
    for r in pred_rows:
        row: list[float] = []
        for v in r:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise MetricsError("Prediction scores must be numeric.")
            row.append(float(v))
        sm.append(row)
    return sm, tm


def _average_precision_binary_column(
    y_true: list[int],
    y_score: list[float],
) -> float:
    """Average precision for one binary column (sklearn-style step sum).

    Returns
    -------
    float
        Column AP in ``[0.0, 1.0]`` (``0.0`` when there are no positives).
    """
    total_pos = sum(y_true)
    if total_pos == 0:
        return 0.0

    order = sorted(
        range(len(y_score)),
        key=lambda i: (-y_score[i], i),
    )
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for idx in order:
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_pos
        ap += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return ap


def _average_precision_micro(
    y_true: list[list[int]],
    y_score: list[list[float]],
) -> float:
    flat_t: list[int] = []
    flat_s: list[float] = []
    for row_t, row_s in zip(y_true, y_score, strict=True):
        flat_t.extend(row_t)
        flat_s.extend(row_s)
    return _average_precision_binary_column(flat_t, flat_s)


@register_scorer
def average_precision(  # noqa: DOC501, DOC502
    predictions: Sequence[Any],
    targets: Sequence[Any],
    *,
    average: str = "macro",
) -> float:
    """Average precision for multi-label detection (per-label PR integral).

    ``predictions`` must be a nested sequence of numeric scores (typically
    confidences in ``[0, 1]``) with shape ``(n_samples, n_labels)``. ``targets``
    may be either the same-shaped binary indicator matrix or ragged integer
    label-index rows (converted to multi-hot using the union of indices).

    This implementation matches the common precision--recall step-sum
    formulation used by ``sklearn.metrics.average_precision_score`` for
    multilabel data, with **deterministic** tie-breaking on equal scores
    (lower sample index first after sorting by descending score).

    Parameters
    ----------
    predictions : sequence of sequence of float
        Model scores per label.
    targets : sequence of sequence of int
        Ground-truth binary rows or label-index rows.
    average : {'macro', 'micro'}, optional
        ``macro`` averages AP across labels; ``micro`` pools labels and samples.

    Returns
    -------
    float
        Mean average precision in ``[0.0, 1.0]``.

    Raises
    ------
    MetricsError
        If inputs are empty, mis-shaped, or ``average`` is unsupported.
    """
    validate_equal_length(predictions, targets)
    y_score, y_true = _as_float_score_matrix(predictions, targets)
    if average == "micro":
        return _average_precision_micro(y_true, y_score)
    if average == "macro":
        n_labels = len(y_true[0])
        col_aps: list[float] = []
        for j in range(n_labels):
            col_t = [row[j] for row in y_true]
            col_s = [row[j] for row in y_score]
            col_aps.append(_average_precision_binary_column(col_t, col_s))
        return sum(col_aps) / n_labels
    raise MetricsError(f"Unsupported average mode for AP: {average!r}.")
