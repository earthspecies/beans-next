"""CIDEr scorer internals.

Adapted from https://github.com/salaniz/pycocoevalcap (MIT licence).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np


def precook(s: str, n: int = 4) -> dict[tuple[str, ...], int]:
    """Compute n-gram counts for a string.

    Parameters
    ----------
    s : str
        Input string (space-tokenized).
    n : int
        Maximum n-gram order.

    Returns
    -------
    dict[tuple[str, ...], int]
        N-gram → count mapping for orders 1 through *n*.
    """
    words = s.split()
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs: list[str], n: int = 4) -> list[dict[tuple[str, ...], int]]:
    """Compute n-gram counts for a list of reference strings.

    Parameters
    ----------
    refs : list[str]
        Reference captions.
    n : int
        Maximum n-gram order.

    Returns
    -------
    list[dict[tuple[str, ...], int]]
        One n-gram count dict per reference.
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test: str, n: int = 4) -> dict[tuple[str, ...], int]:
    """Compute n-gram counts for a hypothesis string.

    Parameters
    ----------
    test : str
        Hypothesis caption.
    n : int
        Maximum n-gram order.

    Returns
    -------
    dict[tuple[str, ...], int]
        N-gram → count mapping.
    """
    return precook(test, n)


class CiderScorer:
    """Accumulate hypothesis/reference pairs and compute CIDEr.

    Parameters
    ----------
    test : str or None
        Optional initial hypothesis string.
    refs : list[str] or None
        Optional initial reference strings.
    n : int
        Maximum n-gram order.
    sigma : float
        Standard deviation for the Gaussian length penalty.
    """

    def __init__(
        self,
        test: str | None = None,
        refs: list[str] | None = None,
        n: int = 4,
        sigma: float = 6.0,
    ) -> None:
        self.n = n
        self.sigma = sigma
        self.crefs: list[list[dict[tuple[str, ...], int]]] = []
        self.ctest: list[dict[tuple[str, ...], int] | None] = []
        self.document_frequency: dict[tuple[str, ...], float] = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len: float | None = None

    def copy(self) -> "CiderScorer":
        """Return a shallow copy of this scorer.

        Returns
        -------
        CiderScorer
            New instance with copied ``crefs`` and ``ctest`` lists.
        """
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def cook_append(
        self,
        test: str | None,
        refs: list[str] | None,
    ) -> None:
        """Append one hypothesis/reference pair.

        Parameters
        ----------
        test : str or None
            Hypothesis string, or ``None`` as a placeholder.
        refs : list[str] or None
            Reference strings. When ``None``, nothing is appended.
        """
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            self.ctest.append(cook_test(test) if test is not None else None)

    def size(self) -> int:
        """Return the number of accumulated pairs.

        Returns
        -------
        int
            Number of (hypothesis, references) pairs added so far.
        """
        assert len(self.crefs) == len(self.ctest), (
            "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        )
        return len(self.crefs)

    def __iadd__(
        self,
        other: "tuple[str, list[str]] | CiderScorer",
    ) -> "CiderScorer":
        """Add a pair or another scorer in-place.

        Parameters
        ----------
        other : tuple[str, list[str]] or CiderScorer
            Either a ``(hypothesis, references)`` tuple or another scorer whose
            accumulated data is merged into this one.

        Returns
        -------
        CiderScorer
            ``self``, updated.
        """
        if isinstance(other, tuple):
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self) -> None:
        """Compute document frequency for all n-grams in the reference corpus.

        Populates ``self.document_frequency`` in-place. Call before
        :meth:`compute_cider`.
        """
        for refs in self.crefs:
            for ngram in {
                ngram for ref in refs for ngram in ref
            }:
                self.document_frequency[ngram] += 1

    def compute_cider(self) -> list[float]:
        """Compute per-sample CIDEr scores (×10 scale).

        Returns
        -------
        list[float]
            One score per accumulated sample.
        """

        def counts2vec(
            cnts: dict[tuple[str, ...], int],
        ) -> tuple[list[dict[tuple[str, ...], float]], list[float], int]:
            vec: list[dict[tuple[str, ...], float]] = [
                defaultdict(float) for _ in range(self.n)
            ]
            length = 0
            norm = [0.0] * self.n
            for ngram, term_freq in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                assert self.ref_len is not None
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += vec[n][ngram] ** 2
                if n == 1:
                    length += term_freq
            norm = [math.sqrt(v) for v in norm]
            return vec, norm, length

        def sim(
            vec_hyp: list[dict[tuple[str, ...], float]],
            vec_ref: list[dict[tuple[str, ...], float]],
            norm_hyp: list[float],
            norm_ref: list[float],
            length_hyp: int,
            length_ref: int,
        ) -> np.ndarray:
            delta = float(length_hyp - length_ref)
            val = np.zeros(self.n)
            for n in range(self.n):
                for ngram in vec_hyp[n]:
                    overlap = min(vec_hyp[n][ngram], vec_ref[n][ngram])
                    val[n] += overlap * vec_ref[n][ngram]
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= norm_hyp[n] * norm_ref[n]
                assert not math.isnan(val[n])
                val[n] *= math.exp(-(delta**2) / (2 * self.sigma**2))
            return val

        self.ref_len = math.log(float(len(self.crefs)))

        scores: list[float] = []
        for test, refs in zip(self.ctest, self.crefs, strict=False):
            assert test is not None
            vec, norm, length = counts2vec(test)
            score = np.zeros(self.n)
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = float(np.mean(score)) / len(refs) * 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self) -> tuple[float, np.ndarray]:
        """Compute corpus-level CIDEr score.

        Returns
        -------
        tuple[float, np.ndarray]
            ``(mean_cider, per_sample_scores)`` both on the ×10 scale.
        """
        self.compute_doc_freq()
        assert len(self.ctest) >= max(self.document_frequency.values())
        score = self.compute_cider()
        arr = np.array(score)
        return float(np.mean(arr)), arr
