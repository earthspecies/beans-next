"""Dataset-level metrics that require all samples simultaneously.

These complement the per-sample metrics in ``classification.py`` and
``detection.py``.  The key difference:

- :func:`compute_macro_f1` aggregates per-class TP/FP/FN across *all* samples
  before computing F1, producing true macro-averaged F1 rather than the mean
  of per-sample binary scores.
- :func:`compute_dataset_map` builds a *global* label vocabulary from every
  sample and computes per-class AP with the full sample pool as context,
  matching the standard multi-label mAP definition.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["compute_dataset_map", "compute_macro_f1"]


def compute_macro_f1(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    zero_division: float = 0.0,
) -> float:
    """Macro-averaged F1 computed from per-class TP/FP/FN across all samples.

    Unlike averaging per-sample binary F1 values (which equals accuracy for
    single-label classification), this aggregates counts globally so that
    classes with fewer samples contribute equally to the final score.

    Parameters
    ----------
    predictions : Sequence[str]
        Post-processed predicted label for each sample (one label per sample).
    targets : Sequence[str]
        Ground-truth label for each sample.
    zero_division : float, optional
        Value returned when precision or recall is undefined (denominator zero).

    Returns
    -------
    float
        Macro-averaged F1 in ``[0.0, 1.0]``.  Returns ``0.0`` when both
        sequences are empty.

    Raises
    ------
    ValueError
        If ``predictions`` and ``targets`` have different lengths.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have the same length, "
            f"got {len(predictions)} vs {len(targets)}"
        )
    if not predictions:
        return 0.0

    vocab = sorted(set(list(predictions) + list(targets)))
    z = float(zero_division)

    tp: dict[str, int] = {c: 0 for c in vocab}
    fp: dict[str, int] = {c: 0 for c in vocab}
    fn: dict[str, int] = {c: 0 for c in vocab}

    for pred, tgt in zip(predictions, targets, strict=False):
        if pred == tgt:
            tp[pred] = tp.get(pred, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[tgt] = fn.get(tgt, 0) + 1

    f1s: list[float] = []
    for c in vocab:
        tp_c = tp.get(c, 0)
        fp_c = fp.get(c, 0)
        fn_c = fn.get(c, 0)
        denom_p = tp_c + fp_c
        denom_r = tp_c + fn_c
        prec = z if denom_p == 0 else tp_c / denom_p
        rec = z if denom_r == 0 else tp_c / denom_r
        if prec + rec == 0:
            f1s.append(z)
        else:
            f1s.append(2 * prec * rec / (prec + rec))

    return sum(f1s) / len(f1s)


def compute_dataset_map(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[Sequence[str]],
) -> float:
    """Mean average precision computed over a global label vocabulary.

    Builds one shared vocabulary from all labels across every sample, then
    constructs ``(n_samples, n_labels)`` score and target matrices and delegates
    to the column-wise AP computation in
    :func:`beans_next.metrics.detection.average_precision`.

    This differs from averaging per-sample AP values, which use a per-sample
    local vocabulary and therefore miss the negative signal from samples where
    a class is absent.

    Parameters
    ----------
    predictions : Sequence[Sequence[str]]
        Predicted label set for each sample (strings, not scores).
    targets : Sequence[Sequence[str]]
        Ground-truth label set for each sample.

    Returns
    -------
    float
        Mean average precision in ``[0.0, 1.0]``.  Returns ``0.0`` when both
        sequences are empty or the global vocabulary is empty.

    Raises
    ------
    ValueError
        If ``predictions`` and ``targets`` have different lengths.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have the same length, "
            f"got {len(predictions)} vs {len(targets)}"
        )
    if not predictions:
        return 0.0

    # Build global vocabulary (deterministic order).
    seen: set[str] = set()
    vocab: list[str] = []
    for preds, tgts in zip(predictions, targets, strict=False):
        for lab in list(preds) + list(tgts):
            if lab and lab not in seen:
                seen.add(lab)
                vocab.append(lab)
    vocab = sorted(vocab)

    if not vocab:
        return 0.0

    tgt_sets = [set(t) for t in targets]
    pred_sets = [set(p) for p in predictions]

    y_true = [[1 if v in ts else 0 for v in vocab] for ts in tgt_sets]
    y_score = [[1.0 if v in ps else 0.0 for v in vocab] for ps in pred_sets]

    from beans_next.metrics.detection import average_precision

    return float(average_precision(y_score, y_true, average="macro"))
