"""CIDEr (Consensus-Based Image Description Evaluation) metric.

Reference implementation adapted from:
    Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
    https://github.com/salaniz/pycocoevalcap
"""

from __future__ import annotations

import numpy as np

from beans_next.metrics._cider.cider_scorer import CiderScorer


class Cider:
    """Compute the CIDEr metric over a set of hypothesis/reference pairs.

    Parameters
    ----------
    n : int
        Maximum n-gram order (1 through *n*). Default is 4.
    sigma : float
        Standard deviation for the Gaussian length penalty. Default is 6.0.
    """

    def __init__(self, n: int = 4, sigma: float = 6.0) -> None:
        self._n = n
        self._sigma = sigma

    def compute_score(
        self,
        gts: dict[str, list[str]],
        res: dict[str, list[str]],
    ) -> tuple[float, np.ndarray]:
        """Compute CIDEr score for a corpus.

        Parameters
        ----------
        gts : dict[str, list[str]]
            Ground-truth captions, keyed by sample id. Each value is a list of
            one or more reference strings.
        res : dict[str, list[str]]
            Hypothesis captions, keyed by sample id. Each value must be a
            one-element list (the model's predicted caption).

        Returns
        -------
        tuple[float, np.ndarray]
            ``(mean_score, per_sample_scores)`` where ``mean_score`` is the
            corpus-level CIDEr (×10 scale) and ``per_sample_scores`` is the
            per-sample array at the same scale.
        """
        assert gts.keys() == res.keys()
        img_ids = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for img_id in img_ids:
            hypo = res[img_id]
            ref = gts[img_id]
            assert isinstance(hypo, list)
            assert len(hypo) == 1
            assert isinstance(ref, list)
            assert len(ref) > 0
            cider_scorer += (hypo[0], ref)

        return cider_scorer.compute_score()

    def method(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            Always ``"CIDEr"``.
        """
        return "CIDEr"
