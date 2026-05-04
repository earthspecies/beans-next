"""Classification metrics: accuracy, precision, recall, F1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from beans_next.metrics.base import MetricsError, register_scorer, validate_equal_length

__all__ = ["accuracy", "f1", "precision", "recall", "top1_accuracy"]


def _is_nested_sequence(seq: Sequence[Any]) -> bool:
    if not seq:
        return False
    first = seq[0]
    return isinstance(first, (list, tuple))


def _is_binary_matrix(rows: Sequence[Sequence[int | float]]) -> bool:
    if not rows:
        return False
    lens = {len(r) for r in rows}
    if len(lens) != 1:
        return False
    for r in rows:
        for v in r:
            if v not in (0, 1, 0.0, 1.0):
                return False
    return True


def _as_multiclass_ints(seq: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for v in seq:
        if isinstance(v, bool) or not isinstance(v, int):
            raise MetricsError("Multiclass inputs must be a sequence of integers.")
        out.append(int(v))
    return out


def _flat_binary_prf(
    predictions: Sequence[int],
    targets: Sequence[int],
    *,
    zero_division: float | int,
) -> tuple[float, float, float]:
    """Binary precision/recall/F1 for flat 0/1 labels (positive label ``1``).

    Returns
    -------
    tuple of float
        ``(precision, recall, f1)`` for the positive class.
    """
    z = _zero_div_value(zero_division)
    tp = fp = fn = 0
    for p, t in zip(predictions, targets, strict=True):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t != 1:
            fp += 1
        elif p != 1 and t == 1:
            fn += 1
    denom_p = tp + fp
    denom_r = tp + fn
    prec = z if denom_p == 0 else tp / denom_p
    rec = z if denom_r == 0 else tp / denom_r
    if prec + rec == 0:
        f1v = z
    else:
        f1v = 2 * prec * rec / (prec + rec)
    return prec, rec, f1v


def _coerce_to_multilabel_binary(
    predictions: Sequence[Any],
    targets: Sequence[Any],
) -> tuple[list[list[int]], list[list[int]]]:
    """Return ``(y_pred, y_true)`` as matching 0/1 indicator matrices.

    Supports:
    - flat multiclass integer labels (converted to one-hot),
    - fixed-width 0/1 matrices,
    - ragged label-index rows (union of indices defines columns).

    Returns
    -------
    tuple of list of list of int
        ``(predictions_matrix, targets_matrix)`` with identical shape.

    Raises
    ------
    MetricsError
        If layouts are mixed or values cannot be interpreted.
    """
    pred_nested = _is_nested_sequence(predictions)
    tgt_nested = _is_nested_sequence(targets)
    if pred_nested != tgt_nested:
        raise MetricsError(
            "predictions and targets must use the same layout (flat vs nested).",
        )
    if not pred_nested:
        p_int = _as_multiclass_ints(predictions)
        t_int = _as_multiclass_ints(targets)
        n_classes = max(p_int + t_int, default=0) + 1
        if n_classes <= 0:
            raise MetricsError("Could not infer a positive number of classes.")
        pred_m = [[1 if p == c else 0 for c in range(n_classes)] for p in p_int]
        tgt_m = [[1 if t == c else 0 for c in range(n_classes)] for t in t_int]
        return pred_m, tgt_m

    pred_rows = [list(r) for r in predictions]
    tgt_rows = [list(r) for r in targets]

    if _is_binary_matrix(pred_rows) and _is_binary_matrix(tgt_rows):
        if len(pred_rows[0]) != len(tgt_rows[0]):
            raise MetricsError(
                "predictions and targets must have the same number of labels.",
            )
        pm = [[int(v) for v in r] for r in pred_rows]
        tm = [[int(v) for v in r] for r in tgt_rows]
        return pm, tm

    labels: set[int] = set()
    for row in pred_rows + tgt_rows:
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                raise MetricsError("Label indices must be integers.")
            if v < 0:
                raise MetricsError("Label indices must be non-negative.")
            labels.add(int(v))
    n_classes = max(labels, default=-1) + 1
    if n_classes <= 0:
        raise MetricsError(
            "Could not infer a positive number of classes from label indices.",
        )

    def indices_to_row(indices: list[int]) -> list[int]:
        row = [0] * n_classes
        for i in indices:
            row[i] = 1
        return row

    pm = [indices_to_row([int(x) for x in r]) for r in pred_rows]
    tm = [indices_to_row([int(x) for x in r]) for r in tgt_rows]
    return pm, tm


def _zero_div_value(zero_division: float | int) -> float:
    return float(zero_division)


def _precision_recall_f1_per_label(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    *,
    zero_division: float | int,
) -> tuple[list[float], list[float], list[float]]:
    n_labels = len(y_true[0])
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    z = _zero_div_value(zero_division)
    for j in range(n_labels):
        tp = fp = fn = 0
        for i in range(len(y_true)):
            t, p = y_true[i][j], y_pred[i][j]
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1
        denom_p = tp + fp
        denom_r = tp + fn
        prec = z if denom_p == 0 else tp / denom_p
        rec = z if denom_r == 0 else tp / denom_r
        if prec + rec == 0:
            f1 = z
        else:
            f1 = 2 * prec * rec / (prec + rec)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return precs, recs, f1s


def _micro_prf(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    *,
    zero_division: float | int,
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            t, p = y_true[i][j], y_pred[i][j]
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1
    z = _zero_div_value(zero_division)
    denom_p = tp + fp
    denom_r = tp + fn
    prec = z if denom_p == 0 else tp / denom_p
    rec = z if denom_r == 0 else tp / denom_r
    if prec + rec == 0:
        f1 = z
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def _weighted_prf(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    *,
    zero_division: float | int,
) -> tuple[float, float, float]:
    precs, recs, f1s = _precision_recall_f1_per_label(
        y_true,
        y_pred,
        zero_division=zero_division,
    )
    supports = [sum(row[j] for row in y_true) for j in range(len(y_true[0]))]
    total = sum(supports)
    if total == 0:
        z = _zero_div_value(zero_division)
        return z, z, z
    wp = sum(p * s for p, s in zip(precs, supports, strict=True)) / total
    wr = sum(r * s for r, s in zip(recs, supports, strict=True)) / total
    wf = sum(f * s for f, s in zip(f1s, supports, strict=True)) / total
    return wp, wr, wf


@register_scorer
def accuracy(predictions: Sequence[Any], targets: Sequence[Any]) -> float:
    """Subset accuracy (exact row match) or multiclass token accuracy.

    For multilabel/binary matrices, a sample is correct only if all labels match.

    Parameters
    ----------
    predictions : sequence
        Predictions: flat ``int`` labels or nested multilabel layouts (see module doc).
    targets : sequence
        Ground truth in the same layout as ``predictions``.

    Returns
    -------
    float
        Accuracy in ``[0.0, 1.0]``.
    """
    validate_equal_length(predictions, targets)
    pred_nested = _is_nested_sequence(predictions)
    if not pred_nested:
        p_int = _as_multiclass_ints(predictions)
        t_int = _as_multiclass_ints(targets)
        correct = sum(1 for a, b in zip(p_int, t_int, strict=True) if a == b)
        return correct / len(p_int)

    yp, yt = _coerce_to_multilabel_binary(predictions, targets)
    correct = sum(1 for a, b in zip(yp, yt, strict=True) if a == b)
    return correct / len(yp)


@register_scorer
def precision(  # noqa: DOC501, DOC502
    predictions: Sequence[Any],
    targets: Sequence[Any],
    *,
    average: str = "macro",
    zero_division: float | int = 0,
) -> float:
    """Precision for multiclass or multilabel classification.

    Parameters
    ----------
    predictions : sequence
        Predictions in the same supported layouts as ``accuracy``.
    targets : sequence
        Ground truth.
    average : {'macro', 'micro', 'weighted', 'binary'}, optional
        Averaging strategy. ``binary`` uses the positive label ``1`` for flat
        ``0``/``1`` sequences; for nested layouts it uses a single binary column.
    zero_division : float or int, optional
        Value used when precision is undefined (division by zero).

    Returns
    -------
    float
        Precision score.

    Raises
    ------
    MetricsError
        If ``average`` is not supported or inputs are invalid.
    """
    validate_equal_length(predictions, targets)
    if average == "binary" and not _is_nested_sequence(predictions):
        p_int = _as_multiclass_ints(predictions)
        t_int = _as_multiclass_ints(targets)
        if all(x in (0, 1) for x in p_int) and all(x in (0, 1) for x in t_int):
            p, _, _ = _flat_binary_prf(p_int, t_int, zero_division=zero_division)
            return p
    yp, yt = _coerce_to_multilabel_binary(predictions, targets)
    if average == "macro":
        precs, _, _ = _precision_recall_f1_per_label(
            yt,
            yp,
            zero_division=zero_division,
        )
        return sum(precs) / len(precs)
    if average == "micro":
        p, _, _ = _micro_prf(yt, yp, zero_division=zero_division)
        return p
    if average == "weighted":
        p, _, _ = _weighted_prf(yt, yp, zero_division=zero_division)
        return p
    if average == "binary":
        if len(yt[0]) != 1:
            msg = "average='binary' requires single-label binary inputs."
            raise MetricsError(msg)
        precs, _, _ = _precision_recall_f1_per_label(
            yt,
            yp,
            zero_division=zero_division,
        )
        return precs[0]
    raise MetricsError(f"Unsupported average mode: {average!r}.")


@register_scorer
def recall(  # noqa: DOC501, DOC502
    predictions: Sequence[Any],
    targets: Sequence[Any],
    *,
    average: str = "macro",
    zero_division: float | int = 0,
) -> float:
    """Recall for multiclass or multilabel classification.

    Parameters
    ----------
    predictions : sequence
        Predictions.
    targets : sequence
        Ground truth.
    average : {'macro', 'micro', 'weighted', 'binary'}, optional
        Averaging strategy (see ``precision``).
    zero_division : float or int, optional
        Value used when recall is undefined.

    Returns
    -------
    float
        Recall score.

    Raises
    ------
    MetricsError
        If ``average`` is not supported or inputs are invalid.
    """
    validate_equal_length(predictions, targets)
    if average == "binary" and not _is_nested_sequence(predictions):
        p_int = _as_multiclass_ints(predictions)
        t_int = _as_multiclass_ints(targets)
        if all(x in (0, 1) for x in p_int) and all(x in (0, 1) for x in t_int):
            _, r, _ = _flat_binary_prf(p_int, t_int, zero_division=zero_division)
            return r
    yp, yt = _coerce_to_multilabel_binary(predictions, targets)
    if average == "macro":
        _, recs, _ = _precision_recall_f1_per_label(
            yt,
            yp,
            zero_division=zero_division,
        )
        return sum(recs) / len(recs)
    if average == "micro":
        _, r, _ = _micro_prf(yt, yp, zero_division=zero_division)
        return r
    if average == "weighted":
        _, r, _ = _weighted_prf(yt, yp, zero_division=zero_division)
        return r
    if average == "binary":
        if len(yt[0]) != 1:
            msg = "average='binary' requires single-label binary inputs."
            raise MetricsError(msg)
        _, recs, _ = _precision_recall_f1_per_label(
            yt,
            yp,
            zero_division=zero_division,
        )
        return recs[0]
    raise MetricsError(f"Unsupported average mode: {average!r}.")


@register_scorer
def f1(  # noqa: DOC501, DOC502
    predictions: Sequence[Any],
    targets: Sequence[Any],
    *,
    average: str = "macro",
    zero_division: float | int = 0,
) -> float:
    """F1 score for multiclass or multilabel classification.

    Parameters
    ----------
    predictions : sequence
        Predictions.
    targets : sequence
        Ground truth.
    average : {'macro', 'micro', 'weighted', 'binary'}, optional
        Averaging strategy (see ``precision``).
    zero_division : float or int, optional
        Value used when precision and recall are both zero.

    Returns
    -------
    float
        F1 score.

    Raises
    ------
    MetricsError
        If ``average`` is not supported or inputs are invalid.
    """
    validate_equal_length(predictions, targets)
    if average == "binary" and not _is_nested_sequence(predictions):
        p_int = _as_multiclass_ints(predictions)
        t_int = _as_multiclass_ints(targets)
        if all(x in (0, 1) for x in p_int) and all(x in (0, 1) for x in t_int):
            _, _, f = _flat_binary_prf(p_int, t_int, zero_division=zero_division)
            return f
    yp, yt = _coerce_to_multilabel_binary(predictions, targets)
    if average == "macro":
        _, _, f1s = _precision_recall_f1_per_label(
            yt,
            yp,
            zero_division=zero_division,
        )
        return sum(f1s) / len(f1s)
    if average == "micro":
        _, _, f = _micro_prf(yt, yp, zero_division=zero_division)
        return f
    if average == "weighted":
        _, _, f = _weighted_prf(yt, yp, zero_division=zero_division)
        return f
    if average == "binary":
        if len(yt[0]) != 1:
            msg = "average='binary' requires single-label binary inputs."
            raise MetricsError(msg)
        _, _, f1s = _precision_recall_f1_per_label(
            yt,
            yp,
            zero_division=zero_division,
        )
        return f1s[0]
    raise MetricsError(f"Unsupported average mode: {average!r}.")


@register_scorer
def top1_accuracy(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    label_separator: str = ",",
) -> float:
    """Top-1 accuracy allowing comma-separated multi-reference targets.

    A prediction is correct when it exactly matches any option in the
    comma-separated target string. Mirrors BEANS-Zero's
    ``compute_top1_accuracy``.

    Parameters
    ----------
    predictions : sequence of str
        Post-processed model outputs (one per example).
    targets : sequence of str
        Ground-truth labels. Each element may contain multiple acceptable
        answers separated by ``label_separator``.
    label_separator : str, optional
        Delimiter used to split multi-reference targets. Defaults to ``","``
        (comma).

    Returns
    -------
    float
        Accuracy in ``[0.0, 1.0]``.

    Raises
    ------
    MetricsError
        If ``predictions`` and ``targets`` have different lengths or either
        sequence is empty.
    """
    validate_equal_length(predictions, targets)
    if not predictions:
        raise MetricsError("predictions and targets must be non-empty.")
    correct = sum(
        1
        for pred, tgt in zip(predictions, targets, strict=True)
        if str(pred) in [opt.strip() for opt in str(tgt).split(label_separator)]
    )
    return correct / len(predictions)
