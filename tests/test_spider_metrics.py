"""Tests for the real SPIDEr metric implementation (CIDEr + SPICE).

CIDEr is pure Python + NumPy and runs without any external dependencies.
SPICE requires Java and Stanford CoreNLP JARs; tests that exercise SPICE are
skipped when the prerequisites are absent.
"""

from __future__ import annotations

import pytest

from beans_next.metrics._cider import Cider
from beans_next.metrics._cider.cider_scorer import precook
from beans_next.metrics._spice import SpiceUnavailableError, check_spice_available
from beans_next.metrics._spice._download import stanford_jars_present
from beans_next.metrics.captioning import (
    cider,
    cider_corpus_mean_normalized,
    spider,
)
from beans_next.runner._utils import compute_dataset_level_metrics

# ---------------------------------------------------------------------------
# CIDEr unit tests (always run)
# ---------------------------------------------------------------------------


class TestPrecook:
    """Unit tests for `precook` n-gram helper."""

    def test_unigrams(self) -> None:
        counts = precook("a b c")
        assert counts[("a",)] == 1
        assert counts[("b",)] == 1
        assert counts[("c",)] == 1

    def test_bigrams(self) -> None:
        counts = precook("a b c")
        assert counts[("a", "b")] == 1
        assert counts[("b", "c")] == 1

    def test_repeated(self) -> None:
        counts = precook("a a b")
        assert counts[("a",)] == 2
        assert counts[("a", "a")] == 1

    def test_empty(self) -> None:
        counts = precook("")
        assert len(counts) == 0


class TestCider:
    """Unit tests for the `Cider` metric class."""

    def _corpus(
        self,
        hyps: list[str],
        refs: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        keys = [f"sample{i:04d}" for i in range(len(hyps))]
        gts = {k: [r] for k, r in zip(keys, refs, strict=True)}
        res = {k: [h] for k, h in zip(keys, hyps, strict=True)}
        return gts, res

    def test_identical_multi_sample_returns_high_score(self) -> None:
        # CIDEr needs >1 sample for meaningful IDF; with 1 sample ref_len=log(1)=0
        captions = [
            "a cat is sitting on the mat",
            "the dog runs across the green field",
        ]
        gts, res = self._corpus(captions, captions)
        score, per_sample = Cider().compute_score(gts, res)
        assert score > 5.0  # ×10 scale; perfect match ≈ 10.0

    def test_score_is_float(self) -> None:
        gts, res = self._corpus(["hello world", "foo bar"], ["hello world", "foo bar"])
        score, per_sample = Cider().compute_score(gts, res)
        assert isinstance(score, float)

    def test_per_sample_length(self) -> None:
        gts, res = self._corpus(["a b c", "d e f"], ["a b c", "d e f"])
        _, per_sample = Cider().compute_score(gts, res)
        assert len(per_sample) == 2

    def test_unrelated_captions_lower_score(self) -> None:
        # Compare perfect-match vs mismatched captions in a 2-sample corpus
        caption_a = "a cat is sitting on the mat"
        caption_b = "the dog runs across the green field"
        unrelated = "distant galaxy emits radio waves at night"

        gts_match, res_match = self._corpus(
            [caption_a, caption_b], [caption_a, caption_b]
        )
        gts_miss, res_miss = self._corpus(
            [unrelated, caption_b], [caption_a, caption_b]
        )
        score_match, _ = Cider().compute_score(gts_match, res_match)
        score_miss, _ = Cider().compute_score(gts_miss, res_miss)
        assert score_match > score_miss

    def test_multi_sample(self) -> None:
        captions = ["a dog runs fast", "birds sing at dawn"]
        gts, res = self._corpus(captions, captions)
        score, per_sample = Cider().compute_score(gts, res)
        assert score > 0.0
        assert len(per_sample) == 2


class TestCiderCorpusNormalized:
    """Corpus CIDEr used for captioning benchmarks (no Java)."""

    def test_single_pair_returns_zero(self) -> None:
        assert cider_corpus_mean_normalized(["hello world"], ["hello world"]) == 0.0

    def test_two_sample_identical_positive(self) -> None:
        val = cider_corpus_mean_normalized(
            ["a cat sits", "birds fly"],
            ["a cat sits", "birds fly"],
        )
        assert 0.0 < val <= 1.0

    def test_cider_registered_matches_helper(self) -> None:
        from beans_next.metrics import get_scorer

        hy = ["a b c", "d e f"]
        rf = ["a b c", "d e f"]
        assert get_scorer("cider")(hy, rf) == cider(hy, rf)
        assert cider(hy, rf) == cider_corpus_mean_normalized(hy, rf)

    def test_compute_dataset_level_metrics_captioning(self) -> None:
        pairs = [
            ("the dog runs fast", "a dog running"),
            ("bird sings at dawn", "morning songbird"),
        ]
        out = compute_dataset_level_metrics(pairs, "captioning")
        assert "cider" in out
        assert 0.0 <= out["cider"] <= 1.0


# ---------------------------------------------------------------------------
# SPICE availability check
# ---------------------------------------------------------------------------


_spice_available = False
try:
    check_spice_available()
    _spice_available = True
except SpiceUnavailableError:
    pass

_skip_spice = pytest.mark.skipif(
    not _spice_available,
    reason="SPICE unavailable: Java absent or Stanford JARs not installed",
)


class TestSpiceAvailability:
    """Tests for SPICE availability checks and `SpiceUnavailableError`."""

    def test_unavailable_raises_spice_error(self) -> None:
        """SpiceUnavailableError raised when prerequisites absent."""
        if _spice_available:
            pytest.skip("SPICE is available on this machine")
        with pytest.raises(SpiceUnavailableError):
            check_spice_available()

    def test_spice_unavailable_error_is_metrics_error(self) -> None:
        from beans_next.metrics.base import MetricsError

        err = SpiceUnavailableError("test")
        assert isinstance(err, MetricsError)

    def test_stanford_jars_present_returns_bool(self) -> None:
        result = stanford_jars_present()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# spider() scorer tests
# ---------------------------------------------------------------------------


class TestSpiderScorer:
    """Tests for the `spider()` scorer function."""

    @_skip_spice
    def test_identical_returns_near_one(self) -> None:
        caption = "a cat is sitting on the mat"
        score = spider([caption], [caption])
        assert 0.0 <= score <= 1.0
        assert score > 0.8

    @_skip_spice
    def test_returns_float(self) -> None:
        score = spider(["hello world"], ["hello world"])
        assert isinstance(score, float)

    @_skip_spice
    def test_multi_sample(self) -> None:
        predictions = ["a dog barks loudly", "a bird sings at dawn"]
        targets = ["a dog barks loudly", "a bird sings at dawn"]
        score = spider(predictions, targets)
        assert 0.0 <= score <= 1.0

    def test_raises_on_length_mismatch(self) -> None:
        from beans_next.metrics.base import MetricsError

        with pytest.raises(MetricsError):
            spider(["a"], ["a", "b"])

    def test_raises_on_non_string_input(self) -> None:
        from beans_next.metrics.base import MetricsError

        with pytest.raises(MetricsError):
            spider([1], ["a"])  # type: ignore[list-item]

    def test_spice_unavailable_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When SPICE is unavailable, spider() degrades gracefully."""
        import beans_next.metrics._spice.spice as spice_mod

        def _fake_check() -> None:
            raise SpiceUnavailableError("no java")

        monkeypatch.setattr(spice_mod, "check_spice_available", _fake_check)
        score = spider(["a cat"], ["a cat"])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# captioning task skips fuzzy matching in postprocess
# ---------------------------------------------------------------------------


class TestCaptioningPostprocess:
    """Tests that captioning tasks skip fuzzy label matching in postprocessing."""

    def test_captioning_skips_fuzzy_match(self) -> None:
        from beans_next.api.types import DatasetExample
        from beans_next.runner.runner import _postprocess_steps_for_examples

        example = DatasetExample(
            sample_id="s1",
            labels="a cat is on the mat",
        )
        _, cleaners = _postprocess_steps_for_examples([example], task_type="captioning")
        names = [s.name for s in cleaners]
        assert "fuzzy_match_to_labels" not in names

    def test_classification_includes_extract_label_from_text(self) -> None:
        """Classification tasks use extract_label_from_text (not fuzzy_match_to_labels)
        so that free-form model output is handled via exact/substring/Levenshtein."""
        from beans_next.api.types import DatasetExample
        from beans_next.runner.runner import _postprocess_steps_for_examples

        example = DatasetExample(
            sample_id="s1",
            labels="cat,dog",
        )
        _, cleaners = _postprocess_steps_for_examples(
            [example], task_type="classification"
        )
        names = [s.name for s in cleaners]
        assert "extract_label_from_text" in names
        assert "fuzzy_match_to_labels" not in names

    def test_none_task_type_includes_fuzzy_match_when_labels_present(self) -> None:
        from beans_next.api.types import DatasetExample
        from beans_next.runner.runner import _postprocess_steps_for_examples

        example = DatasetExample(
            sample_id="s1",
            labels="hawk",
        )
        _, cleaners = _postprocess_steps_for_examples([example], task_type=None)
        names = [s.name for s in cleaners]
        assert "conservative_match_to_labels" in names

    def test_cli_build_postprocess_tuples_delegates_captioning(self) -> None:
        """Legacy CLI postprocess builder must not comma-split caption targets."""
        from beans_next.api.types import DatasetExample
        from beans_next.cli import _build_postprocess_tuples

        example = DatasetExample(
            sample_id="s1",
            labels="a long, human caption with commas",
        )
        parsers, cleaners = _build_postprocess_tuples(
            [example],
            task_type="captioning",
        )
        assert parsers == ()
        names = [s.name for s in cleaners]
        assert "fuzzy_match_to_labels" not in names
