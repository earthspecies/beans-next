"""Deterministic evaluation metrics (classification, detection, captioning)."""

from __future__ import annotations

from collections.abc import Mapping

from beans_next.api.types import DatasetExample
from beans_next.metrics.base import (
    MetricsError,
    get_scorer,
    list_scorers,
    register_scorer,
    validate_equal_length,
)
from beans_next.metrics.captioning import cider, cider_corpus_mean_normalized, spider
from beans_next.metrics.classification import (
    accuracy,
    f1,
    precision,
    recall,
    top1_accuracy,
)
from beans_next.metrics.dataset import compute_dataset_map, compute_macro_f1
from beans_next.metrics.detection import average_precision
from beans_next.post_process.pipeline import PostProcessResult

__all__ = [
    "MetricsError",
    "accuracy",
    "average_precision",
    "compute_dataset_map",
    "compute_macro_f1",
    "f1",
    "get_scorer",
    "list_scorers",
    "precision",
    "recall",
    "register_scorer",
    "score_sample",
    "cider",
    "cider_corpus_mean_normalized",
    "spider",
    "top1_accuracy",
    "validate_equal_length",
]


def _normalize_label_token(s: str) -> str:
    return " ".join(s.strip().split())


def _parse_label_list(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def score_sample(
    example: DatasetExample,
    *,
    post: PostProcessResult,
    raw_predictions: list[str],
    task_type: str | None = None,
) -> Mapping[str, float]:
    """Compute a minimal deterministic metric payload for one example.

    This is a lightweight bridge used by
    :class:`beans_next.runner.runner.BenchmarkRunner` in iteration 1. It infers
    which metric family to apply using `DatasetExample`
    content, preferring explicit metadata when available.

    Parameters
    ----------
    example
        One normalized dataset row.
    post
        A :class:`~beans_next.post_process.pipeline.PostProcessResult`.
    raw_predictions
        Raw decoded prediction strings from the launcher (n-best); the first entry
        is treated as the primary output.
    task_type : str or None, optional
        Explicit task type string (e.g. ``"classification"``, ``"detection"``).
        When provided, takes precedence over ``example.metadata["task"]``.

    Returns
    -------
    Mapping[str, float]
        Per-sample metric values. Empty when inputs do not match any supported
        pattern. For ``task_type`` captioning, returns an empty mapping because
        CIDEr is computed once over the full corpus (see ``mean.cider`` in
        ``summary.json`` from :class:`~beans_next.runner.runner.BenchmarkRunner`).
    """
    pred_text = raw_predictions[0] if raw_predictions else ""
    processed = getattr(post, "text", pred_text) or pred_text
    labels = getattr(example, "labels", None)
    meta = getattr(example, "metadata", {}) or {}
    # Explicit task_type kwarg wins over metadata.
    if task_type is not None:
        task_s = task_type.lower()
    else:
        task = meta.get("task") if isinstance(meta, dict) else None
        task_s = task.lower() if isinstance(task, str) else ""

    if isinstance(labels, str):
        if "caption" in task_s:
            return {}
        y_pred = _normalize_label_token(processed)
        y_true = _normalize_label_token(labels)
        acc = 1.0 if (y_pred == y_true and y_true) else 0.0
        return {
            "accuracy": acc,
            # Match BEANS-Zero evaluator semantics: for single-label classification,
            # top-1 accuracy is identical to exact-match accuracy after postprocess.
            "top1_accuracy": acc,
            "precision": acc,
            "recall": acc,
            "f1": acc,
        }

    if isinstance(labels, list):
        tgt = [_normalize_label_token(x) for x in labels if isinstance(x, str) and x]
        pred_labels = [_normalize_label_token(x) for x in _parse_label_list(processed)]

        if (
            "classification" in task_s or "open_ended" in task_s
        ) and "detection" not in task_s:
            # Multi-reference classification: correct if prediction matches any label.
            # ``open_ended`` is included so BirdSet (and similar open-set tasks that
            # ship multi-species gold) gets top-1 any-of, not multi-label AP.
            norm_pred = _normalize_label_token(processed)
            top1 = 1.0 if (processed.strip() in tgt or norm_pred in tgt) else 0.0
            if tgt:
                try:
                    top1 = float(top1_accuracy([processed], [",".join(tgt)]))
                except Exception:
                    pass
            return {"accuracy": top1, "top1_accuracy": top1}

        # Detection task (or unknown task type with list labels): multi-label AP.
        vocab = sorted(set(tgt) | set(pred_labels))
        if not vocab:
            return {}
        y_true = [[1 if v in tgt else 0 for v in vocab]]
        y_score = [[1.0 if v in pred_labels else 0.0 for v in vocab]]
        ap = float(average_precision(y_score, y_true, average="macro"))
        tp = sum(1 for v in vocab if v in tgt and v in pred_labels)
        fp = sum(1 for v in vocab if v not in tgt and v in pred_labels)
        fn = sum(1 for v in vocab if v in tgt and v not in pred_labels)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1v = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return {
            "average_precision": ap,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
        }

    return {}
