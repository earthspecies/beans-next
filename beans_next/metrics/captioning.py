"""Corpus CIDEr captioning metrics.

**Primary metric (no Java):** :func:`cider` / :func:`cider_corpus_mean_normalized` —
full-corpus CIDEr with IDF over all references, normalized to ``[0, 1]`` (internal
×10 scale divided by 10). Per-sample CIDEr with a one-item corpus is degenerate,
so ``beans_next.metrics.score_sample`` returns no keys for captioning; the runner
adds ``summary.metrics.mean.cider`` from all non-error rows.

"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from beans_next.metrics._cider import Cider
from beans_next.metrics.base import MetricsError, register_scorer, validate_equal_length

__all__ = [
    "cider",
    "cider_corpus_mean_normalized",
]

_logger = logging.getLogger(__name__)

_MIN_CORPUS_FOR_CIDER = 2


def cider_corpus_mean_normalized(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> float:
    """Corpus-level mean CIDEr in ``[0.0, 1.0]`` (×10 CIDEr divided by 10).

    Uses a single :class:`~beans_next.metrics._cider.Cider` pass so document
    frequency is taken over all references. With fewer than two pairs, IDF is
    degenerate and this returns ``0.0``.

    Parameters
    ----------
    predictions
        Model captions (one per example).
    targets
        Reference captions (one per example).

    Returns
    -------
    float
        Mean CIDEr, normalized.

    Raises
    ------
    MetricsError
        If lengths differ, the sequences are empty, or entries are not strings.
    """
    validate_equal_length(predictions, targets)
    if not predictions:
        raise MetricsError("cider_corpus_mean_normalized requires at least one pair.")
    if not all(isinstance(p, str) for p in predictions) or not all(
        isinstance(t, str) for t in targets
    ):
        raise MetricsError("predictions and targets must be str for CIDEr.")

    if len(predictions) < _MIN_CORPUS_FOR_CIDER:
        return 0.0

    refs = {f"sample{i:08d}": [targets[i]] for i in range(len(targets))}
    hyps = {f"sample{i:08d}": [predictions[i]] for i in range(len(predictions))}
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(refs, hyps)
    return float(cider_score / 10.0)


@register_scorer
def cider(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Registered alias for :func:`cider_corpus_mean_normalized`.

    Returns
    -------
    float
        Same as :func:`cider_corpus_mean_normalized`.
    """
    return cider_corpus_mean_normalized(predictions, targets)
