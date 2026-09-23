"""Regression tests for all non-classification task types introduced in this session.

Covers per-sample score_sample routing and dataset-level aggregation for:
  frequency_range, species_set, species_count_dict, species_order,
  species_name, presence_binary, regression, captioning.
"""

from __future__ import annotations

import pytest

from beans_next.api.types import DatasetExample
from beans_next.metrics import score_sample
from beans_next.metrics.base import MetricsError
from beans_next.metrics.regression import extract_frequency_range
from beans_next.post_process.pipeline import PostProcessResult
from beans_next.runner._utils import compute_dataset_level_metrics


def _post(text: str) -> PostProcessResult:
    return PostProcessResult(segments=[], text=text, warnings=())


def _ex(labels: object, task: str) -> DatasetExample:
    return DatasetExample(sample_id="s0", labels=labels, metadata={"task": task})


# ---------------------------------------------------------------------------
# extract_frequency_range
# ---------------------------------------------------------------------------


class TestExtractFrequencyRange:
    """Unit tests for extract_frequency_range helper."""
    def test_hz_range(self) -> None:
        assert extract_frequency_range("200-8000 Hz") == (200.0, 8000.0)

    def test_khz_range_converted(self) -> None:
        low, high = extract_frequency_range("1.5-4 kHz")
        assert low == pytest.approx(1500.0)
        assert high == pytest.approx(4000.0)

    def test_single_value_becomes_point(self) -> None:
        assert extract_frequency_range("3140 Hz") == (3140.0, 3140.0)

    def test_range_normalised_low_high(self) -> None:
        # values given high-first should still return (low, high)
        low, high = extract_frequency_range("8000-200 Hz")
        assert low == 200.0
        assert high == 8000.0

    def test_to_separator(self) -> None:
        low, high = extract_frequency_range("200 to 8000 Hz")
        assert low == 200.0
        assert high == 8000.0

    def test_empty_raises(self) -> None:
        with pytest.raises(MetricsError):
            extract_frequency_range("")

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(MetricsError):
            extract_frequency_range("unknown range")


# ---------------------------------------------------------------------------
# frequency_range task type
# ---------------------------------------------------------------------------


class TestFrequencyRangeScoring:
    """Per-sample scoring for frequency_range task type."""
    def test_exact_range_match(self) -> None:
        r = score_sample(
            _ex("200-8000 Hz", "frequency_range"),
            post=_post("200-8000 Hz"),
            raw_predictions=["200-8000 Hz"],
        )
        assert r["parse_success"] == pytest.approx(1.0)
        assert r["absolute_error_low"] == pytest.approx(0.0)
        assert r["absolute_error_high"] == pytest.approx(0.0)
        assert r["iou"] == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        r = score_sample(
            _ex("200-8000 Hz", "frequency_range"),
            post=_post("500-6000 Hz"),
            raw_predictions=["500-6000 Hz"],
        )
        assert r["absolute_error_low"] == pytest.approx(300.0)
        assert r["absolute_error_high"] == pytest.approx(2000.0)
        assert r["iou"] == pytest.approx(5500.0 / 7800.0)

    def test_no_overlap(self) -> None:
        r = score_sample(
            _ex("200-500 Hz", "frequency_range"),
            post=_post("600-1000 Hz"),
            raw_predictions=["600-1000 Hz"],
        )
        assert r["iou"] == pytest.approx(0.0)

    def test_point_targets_same_value(self) -> None:
        r = score_sample(
            _ex("3140 Hz", "frequency_range"),
            post=_post("3140"),
            raw_predictions=["3140"],
        )
        assert r["iou"] == pytest.approx(1.0)
        assert r["absolute_error_low"] == pytest.approx(0.0)

    def test_point_targets_different_value(self) -> None:
        r = score_sample(
            _ex("3140 Hz", "frequency_range"),
            post=_post("1200"),
            raw_predictions=["1200"],
        )
        assert r["iou"] == pytest.approx(0.0)
        assert r["absolute_error_low"] == pytest.approx(1940.0)

    def test_parse_failure_returns_parse_success_zero(self) -> None:
        r = score_sample(
            _ex("200-8000 Hz", "frequency_range"),
            post=_post("unknown"),
            raw_predictions=["unknown"],
        )
        assert r == {"parse_success": 0.0}


# ---------------------------------------------------------------------------
# species_set task type
# ---------------------------------------------------------------------------


class TestSpeciesSetScoring:
    """Per-sample scoring for species_set task type."""
    def test_perfect_match(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_set"),
            post=_post("Thrush nightingale, Common chaffinch"),
            raw_predictions=["Thrush nightingale, Common chaffinch"],
        )
        assert r["f1"] == pytest.approx(1.0)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)

    def test_reversed_order_full_credit(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_set"),
            post=_post("Common chaffinch, Thrush nightingale"),
            raw_predictions=["Common chaffinch, Thrush nightingale"],
        )
        assert r["f1"] == pytest.approx(1.0)

    def test_case_insensitive(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_set"),
            post=_post("THRUSH NIGHTINGALE, COMMON CHAFFINCH"),
            raw_predictions=["THRUSH NIGHTINGALE, COMMON CHAFFINCH"],
        )
        assert r["f1"] == pytest.approx(1.0)

    def test_missing_one_species(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_set"),
            post=_post("Thrush nightingale"),
            raw_predictions=["Thrush nightingale"],
        )
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(2 / 3)

    def test_extra_species_penalises_precision(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_set"),
            post=_post("Thrush nightingale, Common chaffinch, European robin"),
            raw_predictions=["Thrush nightingale, Common chaffinch, European robin"],
        )
        assert r["recall"] == pytest.approx(1.0)
        assert r["precision"] == pytest.approx(2 / 3)

    def test_one_wrong(self) -> None:
        # tp=1, fp=1, fn=1 → precision=0.5, recall=0.5, f1=0.5
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_set"),
            post=_post("Thrush nightingale, Blackbird"),
            raw_predictions=["Thrush nightingale, Blackbird"],
        )
        assert r["f1"] == pytest.approx(0.5)

    def test_all_wrong(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_set"),
            post=_post("Blackbird, Wren"),
            raw_predictions=["Blackbird, Wren"],
        )
        assert r["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# species_count_dict task type
# ---------------------------------------------------------------------------


class TestSpeciesCountDictScoring:
    """Per-sample scoring for species_count_dict task type."""
    def test_perfect(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3, Common chaffinch: 2", "species_count_dict"),
            post=_post("Thrush nightingale: 3, Common chaffinch: 2"),
            raw_predictions=["Thrush nightingale: 3, Common chaffinch: 2"],
        )
        assert r["species_f1"] == pytest.approx(1.0)
        assert r["count_mae"] == pytest.approx(0.0)

    def test_reversed_output_order(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3, Common chaffinch: 2", "species_count_dict"),
            post=_post("Common chaffinch: 2, Thrush nightingale: 3"),
            raw_predictions=["Common chaffinch: 2, Thrush nightingale: 3"],
        )
        assert r["species_f1"] == pytest.approx(1.0)
        assert r["count_mae"] == pytest.approx(0.0)

    def test_swapped_counts(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3, Common chaffinch: 2", "species_count_dict"),
            post=_post("Thrush nightingale: 2, Common chaffinch: 3"),
            raw_predictions=["Thrush nightingale: 2, Common chaffinch: 3"],
        )
        assert r["species_f1"] == pytest.approx(1.0)
        assert r["count_mae"] == pytest.approx(1.0)

    def test_missing_species(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3, Common chaffinch: 2", "species_count_dict"),
            post=_post("Thrush nightingale: 3"),
            raw_predictions=["Thrush nightingale: 3"],
        )
        assert r["species_f1"] == pytest.approx(2 / 3)
        # union has 2 species; errors = [0, 2]
        assert r["count_mae"] == pytest.approx(1.0)

    def test_extra_species(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3, Common chaffinch: 2", "species_count_dict"),
            post=_post("Thrush nightingale: 3, Common chaffinch: 2, European robin: 1"),
            raw_predictions=["Thrush nightingale: 3, Common chaffinch: 2, European robin: 1"],
        )
        assert r["species_f1"] == pytest.approx(0.8)
        # union has 3; errors = [0, 0, 1]
        assert r["count_mae"] == pytest.approx(1 / 3)

    def test_all_wrong(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3, Common chaffinch: 2", "species_count_dict"),
            post=_post("Blackbird: 4, Wren: 1"),
            raw_predictions=["Blackbird: 4, Wren: 1"],
        )
        assert r["species_f1"] == pytest.approx(0.0)
        assert r["count_mae"] == pytest.approx(2.5)

    def test_empty_prediction(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3, Common chaffinch: 2", "species_count_dict"),
            post=_post(""),
            raw_predictions=[""],
        )
        assert r["species_f1"] == pytest.approx(0.0)
        assert r["count_mae"] == pytest.approx(2.5)

    def test_case_insensitive_species(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale: 3", "species_count_dict"),
            post=_post("THRUSH NIGHTINGALE: 3"),
            raw_predictions=["THRUSH NIGHTINGALE: 3"],
        )
        assert r["species_f1"] == pytest.approx(1.0)
        assert r["count_mae"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# species_order task type
# ---------------------------------------------------------------------------


class TestSpeciesOrderScoring:
    """Per-sample scoring for species_order task type."""
    def test_correct_order(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_order"),
            post=_post("Thrush nightingale, Common chaffinch"),
            raw_predictions=["Thrush nightingale, Common chaffinch"],
        )
        assert r["top1_accuracy"] == pytest.approx(1.0)

    def test_wrong_order(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_order"),
            post=_post("Common chaffinch, Thrush nightingale"),
            raw_predictions=["Common chaffinch, Thrush nightingale"],
        )
        assert r["top1_accuracy"] == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_order"),
            post=_post("THRUSH NIGHTINGALE, COMMON CHAFFINCH"),
            raw_predictions=["THRUSH NIGHTINGALE, COMMON CHAFFINCH"],
        )
        assert r["top1_accuracy"] == pytest.approx(1.0)

    def test_partial_answer(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale, Common chaffinch", "species_order"),
            post=_post("Thrush nightingale"),
            raw_predictions=["Thrush nightingale"],
        )
        assert r["top1_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# species_name task type
# ---------------------------------------------------------------------------


class TestSpeciesNameScoring:
    """Per-sample scoring for species_name task type."""
    def test_exact(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale", "species_name"),
            post=_post("Thrush nightingale"),
            raw_predictions=["Thrush nightingale"],
        )
        assert r["top1_accuracy"] == pytest.approx(1.0)

    def test_case_insensitive(self) -> None:
        for pred in ("Thrush Nightingale", "THRUSH NIGHTINGALE", "thrush nightingale"):
            r = score_sample(
                _ex("Thrush nightingale", "species_name"),
                post=_post(pred),
                raw_predictions=[pred],
            )
            assert r["top1_accuracy"] == pytest.approx(1.0), pred

    def test_wrong_species(self) -> None:
        r = score_sample(
            _ex("Thrush nightingale", "species_name"),
            post=_post("Common chaffinch"),
            raw_predictions=["Common chaffinch"],
        )
        assert r["top1_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# presence_binary task type
# ---------------------------------------------------------------------------


class TestPresenceBinaryScoring:
    """Per-sample scoring for presence_binary task type."""
    def test_yes_match(self) -> None:
        for pred in ("Yes", "yes", "YES"):
            r = score_sample(
                _ex("Yes", "presence_binary"),
                post=_post(pred),
                raw_predictions=[pred],
            )
            assert r["top1_accuracy"] == pytest.approx(1.0), pred

    def test_no_match(self) -> None:
        for pred in ("No", "no", "NO"):
            r = score_sample(
                _ex("No", "presence_binary"),
                post=_post(pred),
                raw_predictions=[pred],
            )
            assert r["top1_accuracy"] == pytest.approx(1.0), pred

    def test_mismatch(self) -> None:
        r = score_sample(
            _ex("Yes", "presence_binary"),
            post=_post("No"),
            raw_predictions=["No"],
        )
        assert r["top1_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# regression task type
# ---------------------------------------------------------------------------


class TestRegressionScoring:
    """Per-sample scoring for regression task type."""
    def test_exact_integer(self) -> None:
        r = score_sample(
            _ex("2", "regression"),
            post=_post("2"),
            raw_predictions=["2"],
        )
        assert r["numeric_parse_success"] == pytest.approx(1.0)
        assert r["absolute_error"] == pytest.approx(0.0)
        assert r["signed_error"] == pytest.approx(0.0)

    def test_off_by_one(self) -> None:
        r = score_sample(
            _ex("3", "regression"),
            post=_post("2"),
            raw_predictions=["2"],
        )
        assert r["absolute_error"] == pytest.approx(1.0)
        assert r["signed_error"] == pytest.approx(-1.0)

    def test_with_db_units(self) -> None:
        r = score_sample(
            _ex("39 dB", "regression"),
            post=_post("39 dB"),
            raw_predictions=["39 dB"],
        )
        assert r["absolute_error"] == pytest.approx(0.0)

    def test_db_off_by_8(self) -> None:
        r = score_sample(
            _ex("55 dB", "regression"),
            post=_post("47 dB"),
            raw_predictions=["47 dB"],
        )
        assert r["absolute_error"] == pytest.approx(8.0)

    def test_hz_value(self) -> None:
        r = score_sample(
            _ex("3100 Hz", "regression"),
            post=_post("2900 Hz"),
            raw_predictions=["2900 Hz"],
        )
        assert r["absolute_error"] == pytest.approx(200.0)

    def test_parse_failure(self) -> None:
        r = score_sample(
            _ex("39 dB", "regression"),
            post=_post("I don't know"),
            raw_predictions=["I don't know"],
        )
        assert r == {"numeric_parse_success": 0.0}

    def test_squared_error(self) -> None:
        r = score_sample(
            _ex("5", "regression"),
            post=_post("2"),
            raw_predictions=["2"],
        )
        assert r["squared_error"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# captioning task type
# ---------------------------------------------------------------------------


class TestCaptiongTaskType:
    """Captioning task type: empty per-sample scores, corpus CIDEr at dataset level."""
    def test_per_sample_returns_empty(self) -> None:
        r = score_sample(
            _ex("A bird sings a clear melodic phrase.", "captioning"),
            post=_post("A bird sings a clear melodic phrase."),
            raw_predictions=["A bird sings a clear melodic phrase."],
        )
        assert r == {}

    def test_dataset_level_cider_returns_score(self) -> None:
        # CIDEr is corpus-level and TF-IDF based; score varies with corpus.
        # We just assert the key is present and the value is a finite float.
        hyps = [
            "A thrush nightingale sings at high pitch.",
            "Common chaffinch vocalizes repeatedly in the background.",
            "A woodpecker drums against a tree trunk.",
        ]
        refs = [
            "A nightingale produces high-pitched melodic calls.",
            "A chaffinch calls repeatedly from a perch.",
            "Rhythmic drumming from a woodpecker is audible.",
        ]
        result = compute_dataset_level_metrics(list(zip(hyps, refs, strict=False)), "captioning")
        assert "cider" in result
        assert isinstance(result["cider"], float)

    def test_dataset_level_cider_all_wrong(self) -> None:
        result = compute_dataset_level_metrics(
            [("zzz zzz zzz", "A bird sings a clear melodic phrase.")],
            "captioning",
        )
        assert "cider" in result

    def test_dataset_level_no_cider_for_classification(self) -> None:
        result = compute_dataset_level_metrics(
            [("cat", "cat"), ("dog", "dog")], "classification"
        )
        assert "cider" not in result


# ---------------------------------------------------------------------------
# dataset-level: regression and frequency_range use aggregate_score_means
# ---------------------------------------------------------------------------


class TestDatasetLevelRegressionAndFrequencyRange:
    """Dataset-level compute returns empty for regression and frequency_range; aggregation is done by aggregate_score_means."""
    def test_regression_dataset_level_empty(self) -> None:
        result = compute_dataset_level_metrics(
            [("2", "3"), ("5", "5")], "regression"
        )
        # regression per-sample aggregation done by aggregate_score_means, not here
        assert result == {}

    def test_frequency_range_dataset_level_empty(self) -> None:
        result = compute_dataset_level_metrics(
            [("200-8000 Hz", "200-8000 Hz")], "frequency_range"
        )
        assert result == {}


# ---------------------------------------------------------------------------
# MCQ full-label content matching (_mcq_content_match fallback)
# ---------------------------------------------------------------------------


class TestMcqContentMatch:
    """MCQ tasks with full-label GT like '(A) Galerida theklae': all realistic
    model output formats should score 1.0; wrong answers must score 0.0.
    Only fires for full-label GT; bare-letter GT tasks are unaffected."""

    def _ex_mcq(self, label: str) -> DatasetExample:
        return DatasetExample(
            sample_id="s0", labels=label, metadata={"task": "classification"}
        )

    def _score(self, label: str, pred: str) -> float:
        r = score_sample(
            self._ex_mcq(label),
            post=_post(pred),
            raw_predictions=[pred],
        )
        return r["top1_accuracy"]

    # correct predictions — all should be 1.0
    def test_bare_letter(self) -> None:
        assert self._score("(A) Galerida theklae", "A") == pytest.approx(1.0)

    def test_letter_in_parens(self) -> None:
        assert self._score("(A) Galerida theklae", "(A)") == pytest.approx(1.0)

    def test_letter_dot(self) -> None:
        assert self._score("(A) Galerida theklae", "A.") == pytest.approx(1.0)

    def test_full_label(self) -> None:
        assert self._score("(A) Galerida theklae", "(A) Galerida theklae") == pytest.approx(1.0)

    def test_name_only(self) -> None:
        assert self._score("(A) Galerida theklae", "Galerida theklae") == pytest.approx(1.0)

    def test_letter_space_name(self) -> None:
        assert self._score("(A) Galerida theklae", "A Galerida theklae") == pytest.approx(1.0)

    def test_letter_dot_name(self) -> None:
        assert self._score("(A) Galerida theklae", "A. Galerida theklae") == pytest.approx(1.0)

    def test_letter_colon_name(self) -> None:
        assert self._score("(A) Galerida theklae", "A: Galerida theklae") == pytest.approx(1.0)

    def test_trailing_period(self) -> None:
        assert self._score("(A) Galerida theklae", "Galerida theklae.") == pytest.approx(1.0)

    def test_sentence_wrapping(self) -> None:
        assert self._score(
            "(A) Galerida theklae", "The answer is Galerida theklae"
        ) == pytest.approx(1.0)

    def test_case_insensitive_lowercase(self) -> None:
        assert self._score("(A) Galerida theklae", "galerida theklae") == pytest.approx(1.0)

    def test_case_insensitive_uppercase(self) -> None:
        assert self._score("(A) Galerida theklae", "GALERIDA THEKLAE") == pytest.approx(1.0)

    # wrong predictions — must be 0.0
    def test_wrong_species_name(self) -> None:
        assert self._score("(A) Galerida theklae", "Sylvia communis") == pytest.approx(0.0)

    def test_wrong_letter_wrong_name(self) -> None:
        assert self._score("(A) Galerida theklae", "B Sylvia communis") == pytest.approx(0.0)

    def test_wrong_letter(self) -> None:
        assert self._score("(A) Galerida theklae", "B") == pytest.approx(0.0)

    def test_empty(self) -> None:
        assert self._score("(A) Galerida theklae", "") == pytest.approx(0.0)

    # numeric MCQ content
    def test_numeric_bare_letter(self) -> None:
        assert self._score("(A) 3", "A") == pytest.approx(1.0)

    def test_numeric_count_only(self) -> None:
        assert self._score("(A) 3", "3") == pytest.approx(1.0)

    def test_numeric_letter_plus_count(self) -> None:
        assert self._score("(A) 3", "A 3") == pytest.approx(1.0)

    def test_numeric_no_false_positive_inside_larger_number(self) -> None:
        assert self._score("(A) 3", "32") == pytest.approx(0.0)

    # bare-letter GT must be completely unaffected
    def test_bare_letter_gt_correct(self) -> None:
        assert self._score("C", "C") == pytest.approx(1.0)

    def test_bare_letter_gt_case_insensitive(self) -> None:
        assert self._score("C", "c") == pytest.approx(1.0)

    def test_bare_letter_gt_wrong_letter(self) -> None:
        assert self._score("C", "A") == pytest.approx(0.0)

    def test_bare_letter_gt_species_name_no_credit(self) -> None:
        # bare-letter GT has no content to match against
        assert self._score("C", "Galerida theklae") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# species_freq_range task type
# ---------------------------------------------------------------------------


class TestSpeciesFreqRangeScoring:
    """Per-sample scoring for species_freq_range task type (frequency_range_description)."""

    def _score(self, gt: str, pred: str) -> dict:
        ex = DatasetExample(
            sample_id="s0", labels=gt, metadata={"task": "species_freq_range"}
        )
        return score_sample(ex, post=_post(pred), raw_predictions=[pred])

    def test_perfect_two_species(self) -> None:
        gt = "Chloris chloris: 2440-5130 Hz, Cyanopica cooki: 1510-10370 Hz"
        r = self._score(gt, gt)
        assert r["parse_success"] == pytest.approx(1.0)
        assert r["species_f1"] == pytest.approx(1.0)
        assert r["freq_mae_low"] == pytest.approx(0.0)
        assert r["freq_mae_high"] == pytest.approx(0.0)
        assert r["freq_mean_iou"] == pytest.approx(1.0)

    def test_missing_one_species(self) -> None:
        gt = "Chloris chloris: 2440-5130 Hz, Cyanopica cooki: 1510-10370 Hz"
        pred = "Chloris chloris: 2440-5130 Hz"
        r = self._score(gt, pred)
        assert r["species_recall"] == pytest.approx(0.5)
        # Only matched species contribute to freq metrics — perfect for the one match
        assert r["freq_mae_low"] == pytest.approx(0.0)
        assert r["freq_mean_iou"] == pytest.approx(1.0)

    def test_freq_error(self) -> None:
        gt = "Chloris chloris: 2000-8000 Hz"
        pred = "Chloris chloris: 2500-7000 Hz"
        r = self._score(gt, pred)
        assert r["freq_mae_low"] == pytest.approx(500.0)
        assert r["freq_mae_high"] == pytest.approx(1000.0)
        assert r["freq_mean_iou"] == pytest.approx(4500.0 / 6000.0)

    def test_unknown_excluded(self) -> None:
        gt = "Chloris chloris: 2440-5130 Hz, Unknown: 1600-1820 Hz"
        pred = "Chloris chloris: 2440-5130 Hz"
        r = self._score(gt, pred)
        # Unknown stripped from GT → only Chloris chloris in true set → perfect
        assert r["species_f1"] == pytest.approx(1.0)

    def test_no_species_both_empty(self) -> None:
        r = self._score("None", "None")
        assert r["parse_success"] == pytest.approx(1.0)
        assert r["species_f1"] == pytest.approx(1.0)

    def test_empty_pred_against_nonempty_gt(self) -> None:
        r = self._score("Chloris chloris: 2440-5130 Hz", "None")
        assert r["species_f1"] == pytest.approx(0.0)

    def test_wrong_species(self) -> None:
        r = self._score(
            "Chloris chloris: 2440-5130 Hz",
            "Luscinia megarhynchos: 1000-6000 Hz",
        )
        assert r["species_f1"] == pytest.approx(0.0)
        assert "freq_mae_low" not in r  # no matched species


# ---------------------------------------------------------------------------
# species_summary task type
# ---------------------------------------------------------------------------


class TestSpeciesSummaryScoring:
    """Per-sample scoring for species_summary task type (ordered_species_summary)."""

    def _score(self, gt: str, pred: str) -> dict:
        ex = DatasetExample(
            sample_id="s0", labels=gt, metadata={"task": "species_summary"}
        )
        return score_sample(ex, post=_post(pred), raw_predictions=[pred])

    def test_perfect(self) -> None:
        gt = "Pipilo erythrophthalmus: 2 calls, 2330-5150 Hz; Setophaga virens: 2 calls, 4650-7320 Hz"
        r = self._score(gt, gt)
        assert r["species_f1"] == pytest.approx(1.0)
        assert r["count_mae"] == pytest.approx(0.0)
        assert r["freq_mae_low"] == pytest.approx(0.0)
        assert r["freq_mean_iou"] == pytest.approx(1.0)

    def test_wrong_count(self) -> None:
        gt = "Pipilo erythrophthalmus: 2 calls, 2330-5150 Hz"
        pred = "Pipilo erythrophthalmus: 5 calls, 2330-5150 Hz"
        r = self._score(gt, pred)
        assert r["species_f1"] == pytest.approx(1.0)
        assert r["count_mae"] == pytest.approx(3.0)
        assert r["freq_mae_low"] == pytest.approx(0.0)

    def test_singular_call(self) -> None:
        gt = "Seiurus aurocapilla: 1 call, 3170-5680 Hz"
        r = self._score(gt, gt)
        assert r["count_mae"] == pytest.approx(0.0)
        assert r["species_f1"] == pytest.approx(1.0)

    def test_missing_species_no_count_metric(self) -> None:
        gt = "Pipilo erythrophthalmus: 2 calls, 2330-5150 Hz; Setophaga virens: 2 calls, 4650-7320 Hz"
        pred = "Pipilo erythrophthalmus: 2 calls, 2330-5150 Hz"
        r = self._score(gt, pred)
        assert r["species_recall"] == pytest.approx(0.5)
        assert r["count_mae"] == pytest.approx(0.0)  # only matched species

    def test_no_species_both_empty(self) -> None:
        r = self._score("None", "None")
        assert r["species_f1"] == pytest.approx(1.0)

    def test_unknown_excluded(self) -> None:
        gt = "Galerida theklae: 3 calls, 2190-5570 Hz; Unknown: 1 call, 3100-5410 Hz"
        pred = "Galerida theklae: 3 calls, 2190-5570 Hz"
        r = self._score(gt, pred)
        assert r["species_f1"] == pytest.approx(1.0)

    def test_no_matched_species_omits_freq_metrics(self) -> None:
        r = self._score(
            "Pipilo erythrophthalmus: 2 calls, 2330-5150 Hz",
            "Luscinia megarhynchos: 4 calls, 1000-6000 Hz",
        )
        assert r["species_f1"] == pytest.approx(0.0)
        assert "freq_mae_low" not in r
        assert "count_mae" not in r


# ---------------------------------------------------------------------------
# species_name task type — sci/common name lookup
# ---------------------------------------------------------------------------


class TestSpeciesNameLookup:
    """species_name task type accepts both scientific and common names via
    the bundled t3_species_names.json lookup. Only applies to the 46 species
    in the T3 OE single-species tasks — all other species still require exact
    match. Both directions (sci→common and common→sci) are supported."""

    def _score(self, gt: str, pred: str) -> float:
        ex = DatasetExample(
            sample_id="s0", labels=gt, metadata={"task": "species_name"}
        )
        return score_sample(ex, post=_post(pred), raw_predictions=[pred])["top1_accuracy"]

    # sci GT → common name prediction
    def test_sci_gt_exact_sci_pred(self) -> None:
        assert self._score("Sturnus unicolor", "Sturnus unicolor") == pytest.approx(1.0)

    def test_sci_gt_common_name_pred(self) -> None:
        assert self._score("Sturnus unicolor", "Spotless Starling") == pytest.approx(1.0)

    def test_sci_gt_common_name_lowercase(self) -> None:
        assert self._score("Sturnus unicolor", "spotless starling") == pytest.approx(1.0)

    def test_sci_gt_common_name_uppercase(self) -> None:
        assert self._score("Sturnus unicolor", "SPOTLESS STARLING") == pytest.approx(1.0)

    def test_sci_gt_wrong_common_name(self) -> None:
        # Common Starling is Sturnus vulgaris, not unicolor
        assert self._score("Sturnus unicolor", "Common Starling") == pytest.approx(0.0)

    def test_sci_gt_multiple_common_variants(self) -> None:
        assert self._score("Galerida theklae", "Thekla Lark") == pytest.approx(1.0)
        assert self._score("Galerida theklae", "Thekla's Lark") == pytest.approx(1.0)

    def test_sci_gt_short_common_name(self) -> None:
        assert self._score("Luscinia megarhynchos", "Nightingale") == pytest.approx(1.0)

    # Correct rejections
    def test_wrong_species_name(self) -> None:
        assert self._score("Corvus brachyrhynchos", "Raven") == pytest.approx(0.0)

    def test_similar_species_rejected(self) -> None:
        # Blue Tit ≠ Great Tit
        assert self._score("Parus major", "Blue Tit") == pytest.approx(0.0)

    def test_unknown_species_falls_back_to_exact(self) -> None:
        # Species not in lookup — still works via exact match
        assert self._score("Corvus corax", "Corvus corax") == pytest.approx(1.0)
        assert self._score("Corvus corax", "Common Raven") == pytest.approx(0.0)
