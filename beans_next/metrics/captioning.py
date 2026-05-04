"""Captioning metrics: corpus CIDEr (default) and optional SPIDEr (CIDEr + SPICE).

**Primary metric (no Java):** :func:`cider` / :func:`cider_corpus_mean_normalized` —
full-corpus CIDEr with IDF over all references, normalized to ``[0, 1]`` (internal
×10 scale divided by 10). Per-sample CIDEr with a one-item corpus is degenerate,
so ``beans_next.metrics.score_sample`` returns no keys for captioning; the runner
adds ``summary.metrics.mean.cider`` from all non-error rows.

**Optional SPIDEr:** :func:`spider` also averages in SPICE, which requires Java 8+
and Stanford CoreNLP 3.6.0 JARs (``beans-next setup-spice``). When SPICE is
unavailable, :exc:`~beans_next.metrics._spice.SpiceUnavailableError` is caught
and SPICE is treated as 0.0.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from beans_next.metrics._cider import Cider
from beans_next.metrics._spice import Spice, SpiceUnavailableError
from beans_next.metrics.base import MetricsError, register_scorer, validate_equal_length

__all__ = [
    "SpiceUnavailableError",
    "cider",
    "cider_corpus_mean_normalized",
    "spider",
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


@register_scorer
def spider(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compute the SPIDEr metric: ``(CIDEr/10 + SPICE) / 2``.

    Requires Java 8+ on ``PATH`` and Stanford CoreNLP 3.6.0 JARs installed via
    ``beans-next setup-spice``.

    Parameters
    ----------
    predictions : sequence of str
        Model captions (one per example).
    targets : sequence of str
        Reference captions (one per example).

    Returns
    -------
    float
        SPIDEr score in ``[0.0, 1.0]``.

    Raises
    ------
    MetricsError
        If inputs are not non-empty strings or lengths differ.
    """
    validate_equal_length(predictions, targets)
    if not all(isinstance(p, str) for p in predictions) or not all(
        isinstance(t, str) for t in targets
    ):
        raise MetricsError("predictions and targets must be str for spider().")

    refs = {f"sample{i:08d}": [t] for i, t in enumerate(targets)}
    hyps = {f"sample{i:08d}": [p] for i, p in enumerate(predictions)}

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(refs, hyps)
    cider_score /= 10.0  # CIDEr is reported ×10 internally; normalise to [0,1]

    try:
        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(refs, hyps)
    except SpiceUnavailableError as exc:
        # On clusters without Java, treat SPICE as unavailable but keep the run alive.
        # This makes captioning runs usable in CPU-only environments.
        _logger.warning("SPICE unavailable; using 0.0 for SPICE: %s", exc)
        spice_score = 0.0

    return float((cider_score + spice_score) / 2.0)
