"""Shared aggregation helpers used by runner and rescorer."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from beans_next.api.types import DatasetExample

__all__ = [
    "aggregate_score_means",
    "compute_dataset_level_metrics",
    "per_task_score_means",
]

_logger = logging.getLogger(__name__)


def compute_dataset_level_metrics(
    processed_pairs: list[tuple[str, Any]],
    task_type: str | None,
) -> dict[str, float]:
    """Compute dataset-level metrics from all (processed_prediction, targets) pairs.

    For classification tasks returns ``macro_f1``; for detection returns
    ``dataset_map``; for captioning returns ``cider`` (corpus CIDEr in ``[0,1]``).
    Returns an empty dict when ``processed_pairs`` is empty or the task type does
    not match one of these families.

    Parameters
    ----------
    processed_pairs
        List of ``(processed_prediction_text, targets)`` for non-error samples
        that have non-null targets.
    task_type
        Task type string from runner or rescorer (e.g. ``captioning``).

    Returns
    -------
    dict[str, float]
        Dataset-level metric values (may be empty).
    """
    if not processed_pairs:
        return {}
    task_s = (task_type or "").lower()
    try:
        if "caption" in task_s:
            from beans_next.metrics.captioning import cider_corpus_mean_normalized

            hyps: list[str] = []
            refs: list[str] = []
            for pred_text, tgts in processed_pairs:
                hyp = pred_text if isinstance(pred_text, str) else str(pred_text)
                if isinstance(tgts, str):
                    ref = tgts
                elif isinstance(tgts, list) and tgts:
                    ref = str(tgts[0])
                else:
                    continue
                hyps.append(hyp)
                refs.append(ref)
            if len(hyps) != len(processed_pairs):
                _logger.warning(
                    "Captioning dataset metric: skipped %d row(s) with non-string "
                    "targets.",
                    len(processed_pairs) - len(hyps),
                )
            if not hyps:
                return {}
            return {"cider": cider_corpus_mean_normalized(hyps, refs)}

        if "classification" in task_s and "detection" not in task_s:
            from beans_next.metrics.dataset import compute_macro_f1

            predictions = [p for p, _ in processed_pairs]
            targets: list[str] = []
            for _, t in processed_pairs:
                if isinstance(t, str):
                    targets.append(t)
                elif isinstance(t, list) and t:
                    # Multi-reference: canonical label is the first entry.
                    targets.append(str(t[0]))
                else:
                    targets.append("")
            return {"macro_f1": compute_macro_f1(predictions, targets)}

        if "detection" in task_s:
            from beans_next.metrics.dataset import compute_dataset_map

            pred_sets: list[list[str]] = []
            tgt_sets: list[list[str]] = []
            for pred_text, tgts in processed_pairs:
                pred_labels = [
                    p.strip() for p in pred_text.split(",") if p.strip()
                ]
                pred_sets.append(pred_labels)
                if isinstance(tgts, list):
                    tgt_sets.append([str(t) for t in tgts])
                elif isinstance(tgts, str):
                    tgt_sets.append(
                        [t.strip() for t in tgts.split(",") if t.strip()]
                    )
                else:
                    tgt_sets.append([])
            return {"dataset_map": compute_dataset_map(pred_sets, tgt_sets)}
    except Exception:
        _logger.warning(
            "Dataset-level metric computation failed; skipping.",
            exc_info=True,
        )
    return {}


def aggregate_score_means(score_rows: list[Mapping[str, float]]) -> dict[str, float]:
    """Return per-key mean across score rows, skipping missing keys.

    Parameters
    ----------
    score_rows
        List of per-sample score dicts. Empty dicts are skipped per key.

    Returns
    -------
    dict[str, float]
        Sorted-key mean values.
    """
    if not score_rows:
        return {}
    keys: set[str] = set()
    for row in score_rows:
        keys.update(row.keys())
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals = [float(row[key]) for row in score_rows if key in row]
        if vals:
            out[key] = sum(vals) / float(len(vals))
    return out


def per_task_score_means(
    examples: list[DatasetExample],
    score_rows: list[Mapping[str, float]],
) -> dict[str, Any]:
    """Return per-task mean scores keyed by task_id.

    Parameters
    ----------
    examples
        Dataset examples parallel to ``score_rows``.
    score_rows
        Per-sample score dicts.

    Returns
    -------
    dict[str, Any]
        Mapping of task label to mean score dict. ``None`` task_id becomes
        ``"default"``.
    """
    buckets: dict[str | None, list[Mapping[str, float]]] = {}
    for ex, scores in zip(examples, score_rows, strict=True):
        buckets.setdefault(ex.task_id, []).append(scores)
    return {
        (tid if tid is not None else "default"): aggregate_score_means(rows)
        for tid, rows in sorted(
            buckets.items(),
            key=lambda kv: (kv[0] is None, kv[0] or ""),
        )
    }
