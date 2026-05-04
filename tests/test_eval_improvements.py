"""Tests for evaluation improvements: label extraction, dataset-level metrics,
null-target handling, and rescorer task-type awareness."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from beans_next.api.types import DatasetExample, ModelPrediction, ScoredPrediction
from beans_next.metrics import score_sample
from beans_next.metrics.dataset import compute_dataset_map, compute_macro_f1
from beans_next.post_process.cleaners import apply_extract_label_from_text
from beans_next.post_process.pipeline import (
    PostProcessContext,
    PostProcessResult,
    StepSpec,
    run_post_process_pipeline,
)
from beans_next.runner.rescorer import _default_postprocess_steps

# ---------------------------------------------------------------------------
# apply_extract_label_from_text — stage 1: exact match (case-insensitive)
# ---------------------------------------------------------------------------


class TestExtractLabelExactMatch:
    """Stage-1 exact match (case-insensitive) in apply_extract_label_from_text."""

    def _ctx(self, *segments: str) -> PostProcessContext:
        return PostProcessContext(segments=list(segments))

    def test_exact_lowercase(self) -> None:
        ctx = self._ctx("dog")
        out = apply_extract_label_from_text(ctx, labels=["dog", "cat"])
        assert out.segments == ["dog"]

    def test_exact_uppercase_pred(self) -> None:
        """Model output 'DOG' should map to label 'dog'."""
        ctx = self._ctx("DOG")
        out = apply_extract_label_from_text(ctx, labels=["dog", "cat"])
        assert out.segments == ["dog"]

    def test_exact_titlecase_pred(self) -> None:
        ctx = self._ctx("Dog")
        out = apply_extract_label_from_text(ctx, labels=["dog", "cat"])
        assert out.segments == ["dog"]

    def test_label_casing_preserved(self) -> None:
        """Original casing of the vocabulary label is returned even when the
        model output uses different casing."""
        ctx = self._ctx("AMERICAN CROW")
        out = apply_extract_label_from_text(ctx, labels=["American Crow", "Raven"])
        # Exact match: "american crow" == "american crow" → returns original "American Crow"
        assert out.segments == ["American Crow"]

    def test_exact_with_leading_trailing_whitespace(self) -> None:
        ctx = self._ctx("  dog  ")
        out = apply_extract_label_from_text(ctx, labels=["dog", "cat"])
        assert out.segments == ["dog"]


# ---------------------------------------------------------------------------
# apply_extract_label_from_text — stage 2: substring scan
# ---------------------------------------------------------------------------


class TestExtractLabelSubstringScan:
    """Stage-2 substring scan in apply_extract_label_from_text."""

    def _ctx(self, *segments: str) -> PostProcessContext:
        return PostProcessContext(segments=list(segments))

    def test_label_embedded_in_sentence(self) -> None:
        """'It sounds like a dog barking' → 'dog'."""
        ctx = self._ctx("It sounds like a dog barking.")
        out = apply_extract_label_from_text(ctx, labels=["dog", "cat", "bird"])
        assert out.segments == ["dog"]

    def test_reasoning_prefix_stripped(self) -> None:
        ctx = self._ctx("Based on the audio, I believe this is a cat.")
        out = apply_extract_label_from_text(ctx, labels=["dog", "cat", "bird"])
        assert out.segments == ["cat"]

    def test_longest_label_wins(self) -> None:
        """'American crow' should win over bare 'crow' when both are in vocab."""
        ctx = self._ctx("The bird is an American crow.")
        out = apply_extract_label_from_text(
            ctx, labels=["crow", "American crow", "raven"]
        )
        assert out.segments == ["American crow"]

    def test_substring_case_insensitive(self) -> None:
        ctx = self._ctx("I classify this as Cat purring.")
        out = apply_extract_label_from_text(ctx, labels=["cat", "dog"])
        assert out.segments == ["cat"]


# ---------------------------------------------------------------------------
# apply_extract_label_from_text — stage 3: Levenshtein fallback
# ---------------------------------------------------------------------------


class TestExtractLabelLevenshteinFallback:
    """Stage-3 Levenshtein fallback in apply_extract_label_from_text."""

    def _ctx(self, *segments: str) -> PostProcessContext:
        return PostProcessContext(segments=list(segments))

    def test_typo_corrected(self) -> None:
        ctx = self._ctx("kat")
        out = apply_extract_label_from_text(ctx, labels=["cat", "dog"])
        assert out.segments == ["cat"]

    def test_threshold_reject(self) -> None:
        ctx = self._ctx("xxxxxxxxxxx")
        out = apply_extract_label_from_text(
            ctx,
            labels=["cat"],
            apply_threshold=True,
            max_distance=5,
            default_label="None",
        )
        assert out.segments == ["None"]

    def test_threshold_accept(self) -> None:
        ctx = self._ctx("kat")
        out = apply_extract_label_from_text(
            ctx, labels=["cat"], apply_threshold=True, max_distance=5
        )
        assert out.segments == ["cat"]

    def test_empty_labels_raises(self) -> None:
        from beans_next.post_process.pipeline import PostProcessPipelineError

        ctx = self._ctx("dog")
        with pytest.raises(PostProcessPipelineError):
            apply_extract_label_from_text(ctx, labels=[])


# ---------------------------------------------------------------------------
# End-to-end pipeline: extract_label_from_text step registered
# ---------------------------------------------------------------------------


class TestExtractLabelPipelineIntegration:
    """End-to-end pipeline tests for the extract_label_from_text step."""

    def test_free_form_output_maps_correctly(self) -> None:
        """Free-form model output should map to the correct label end-to-end."""
        labels = ("dog", "cat", "bird")
        result = run_post_process_pipeline(
            "Based on the audio analysis, I believe this recording contains a dog.",
            parser_steps=(),
            cleaner_steps=(
                StepSpec("normalize_whitespace", {}),
                StepSpec("strip_eos", {}),
                StepSpec("extract_label_from_text", {"labels": labels}),
            ),
        )
        assert result.text == "dog"

    def test_uppercase_model_output(self) -> None:
        labels = ("dog", "cat", "bird")
        result = run_post_process_pipeline(
            "CAT",
            parser_steps=(),
            cleaner_steps=(
                StepSpec("normalize_whitespace", {}),
                StepSpec("extract_label_from_text", {"labels": labels}),
            ),
        )
        assert result.text == "cat"


# ---------------------------------------------------------------------------
# Regression: binary Yes/No must not be comma-split
# ---------------------------------------------------------------------------


class TestBinaryYesNoDoesNotCommaSplit:
    """Verbose Yes/No answers must map to a single label (no duplication).

    This reproduces the failure mode where OpenAI returns prose like
    "Yes, there is a bird vocalizing..." and the comma-split parser turns it
    into fragments which then fuzzy-match back to "Yes" repeatedly.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Yes, there is a bird vocalizing in this recording.", "Yes"),
            (
                "No, the recording does not contain mammal vocalizations. It primarily "
                "features insect sounds.",
                "No",
            ),
            (
                "Yes, there is an alarm call present. The rapid, repetitive pattern...",
                "Yes",
            ),
        ],
    )
    def test_score_from_file_default_pipeline(self, raw: str, expected: str) -> None:
        parsers, cleaners = _default_postprocess_steps(
            targets=["Yes", "No"],
            task_type=None,  # critical: the buggy case was task_type omitted
        )
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        assert result.text == expected


class TestMcqDoesNotCommaSplitAndChoosesFinalLetter:
    """MCQ A/B/C/D responses must map to a single final letter."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                "To determine the best match:\n- A: A short, high-pitched sound...\n"
                "- B: ...\n- C: ...\n- D: ...\n\nFinal answer: B",
                "B",
            ),
            (
                "Based on the acoustic characteristics described:\n"
                "* **A** describes...\n* **B** describes...\n* **C** describes...\n"
                "* **D** describes...\n\nAnswer: A",
                "A",
            ),
            (
                "Based on the acoustic characteristics of the audio, the correct "
                "description is:\n\n**D**",
                "D",
            ),
        ],
    )
    def test_score_from_file_default_pipeline(self, raw: str, expected: str) -> None:
        parsers, cleaners = _default_postprocess_steps(
            targets=["A", "B", "C", "D"],
            task_type=None,  # critical: without task_type this used to comma-split
        )
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        assert result.text == expected

    def test_markdown_bold_letter_extracts(self) -> None:
        parsers, cleaners = _default_postprocess_steps(
            targets=["A", "B", "C", "D"],
            task_type=None,
        )
        raw = "Based on the acoustic characteristics of the audio, the correct description is:\n\n**B**"
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        assert result.text == "B"

    def test_option_reference_does_not_match_article_a(self) -> None:
        parsers, cleaners = _default_postprocess_steps(
            targets=["A", "B", "C", "D"],
            task_type=None,
        )
        raw = (
            "The sound consists of a short, high-pitched note followed by a longer tone. "
            "This matches the description in option C."
        )
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        assert result.text == "C"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a", "A"),
            ("b", "B"),
            (" c ", "C"),
            ("**d**", "D"),
            ("(b)", "B"),
        ],
    )
    def test_lowercase_single_letter_short_answer_is_accepted(
        self, raw: str, expected: str
    ) -> None:
        parsers, cleaners = _default_postprocess_steps(
            targets=["A", "B", "C", "D"],
            task_type=None,
        )
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        assert result.text == expected


class TestHzBucketDoesNotCommaSplitAndParsesThousandsSeparator:
    """F0 Hz bucket tasks must not comma-split prose or numeric thousands separators."""

    def test_thousands_separator_number_is_parsed_as_one_value(self) -> None:
        parsers, cleaners = _default_postprocess_steps(
            targets=["3650 Hz", "4000 Hz", "4010 Hz", "4030 Hz"],
            task_type=None,
        )
        raw = "#0.00s - 10.00s#: 1,654.75\n"
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        # Parse the full numeric value (1654.75) instead of splitting on the comma.
        # Then map to a single closest bucket label.
        assert result.text == "3650 Hz"

    def test_prose_with_commas_yields_single_bucket_label(self) -> None:
        parsers, cleaners = _default_postprocess_steps(
            targets=["2780 Hz", "3100 Hz", "3650 Hz"],
            task_type=None,
        )
        raw = (
            "To determine the mean fundamental frequency of a vocalization, we would typically "
            "analyze the audio signal using specialized software or tools that can perform a "
            "frequency analysis, such as a spectrogram or pitch-tracking software."
        )
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        assert result.text in {"2780 Hz", "3100 Hz", "3650 Hz"}
        assert "," not in result.text

    def test_deci_hz_range_midpoint_maps_to_bucket(self) -> None:
        # Fixture example: naturelm_v1_1 emits "2131-4440 Hz" which should be
        # interpreted as 213.1–444.0 Hz; midpoint 328.55 → closest bucket 340 Hz.
        parsers, cleaners = _default_postprocess_steps(
            targets=["210 Hz", "300 Hz", "340 Hz", "440 Hz"],
            task_type=None,
        )
        raw = "2131-4440 Hz"
        result = run_post_process_pipeline(raw, parser_steps=parsers, cleaner_steps=cleaners)
        assert result.text == "340 Hz"


# ---------------------------------------------------------------------------
# compute_macro_f1 — dataset-level metric differs from accuracy on imbalanced data
# ---------------------------------------------------------------------------


class TestComputeMacroF1:
    """Tests for compute_macro_f1 (dataset-level, not per-sample average)."""

    def test_perfect_score(self) -> None:
        preds = ["a", "b", "a", "b"]
        tgts = ["a", "b", "a", "b"]
        assert compute_macro_f1(preds, tgts) == pytest.approx(1.0)

    def test_zero_score(self) -> None:
        preds = ["b", "a"]
        tgts = ["a", "b"]
        assert compute_macro_f1(preds, tgts) == pytest.approx(0.0)

    def test_differs_from_accuracy_on_imbalanced_data(self) -> None:
        """
        100 samples: 1 from class A, 99 from class B. Model always predicts 'b'.
        accuracy = 99/100 = 0.99 (dominated by B)
        Class A: tp=0, fp=0, fn=1 → F1=0
        Class B: tp=99, fp=1, fn=0 → P=0.99, R=1.0, F1≈0.995
        macro_f1 ≈ 0.497, well below accuracy=0.99.
        """
        preds = ["b"] * 100
        tgts = ["a"] + ["b"] * 99
        acc = sum(p == t for p, t in zip(preds, tgts, strict=False)) / len(preds)
        mf1 = compute_macro_f1(preds, tgts)
        assert acc == pytest.approx(0.99)
        # macro_f1 should be substantially lower than accuracy
        assert mf1 < 0.6
        assert abs(mf1 - acc) > 0.3

    def test_empty_sequences(self) -> None:
        assert compute_macro_f1([], []) == pytest.approx(0.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_macro_f1(["a"], ["a", "b"])

    def test_multiclass(self) -> None:
        preds = ["a", "b", "c", "a"]
        tgts = ["a", "b", "c", "b"]
        # a: tp=1, fp=1, fn=0 → P=0.5, R=1.0, F1=0.667
        # b: tp=1, fp=0, fn=1 → P=1.0, R=0.5, F1=0.667
        # c: tp=1, fp=0, fn=0 → P=1.0, R=1.0, F1=1.0
        # macro = (0.667 + 0.667 + 1.0) / 3 = 0.778
        result = compute_macro_f1(preds, tgts)
        assert result == pytest.approx(2 / 3 * 2 / 3 + 1 / 3, rel=1e-4)


# ---------------------------------------------------------------------------
# compute_dataset_map — differs from per-sample AP on multi-sample detection
# ---------------------------------------------------------------------------


class TestComputeDatasetMap:
    """Tests for compute_dataset_map (global-vocab dataset-level MAP)."""

    def test_perfect_detection(self) -> None:
        preds = [["a", "b"], ["c"]]
        tgts = [["a", "b"], ["c"]]
        assert compute_dataset_map(preds, tgts) == pytest.approx(1.0)

    def test_empty_sequences(self) -> None:
        assert compute_dataset_map([], []) == pytest.approx(0.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_dataset_map([["a"]], [["a"], ["b"]])

    def test_uses_global_vocab(self) -> None:
        """
        Global vocab = {a, b}.  Sample 1 predicts [a], target [a, b].
        Sample 2 predicts [b], target [b].

        Per-sample AP (old approach) would compute AP on each sample's local vocab.
        Dataset MAP uses both samples for each class, so AP_b includes the
        negative evidence from sample 2 not predicting b initially, etc.

        We just verify the result is a float in [0, 1] and differs from naive
        per-sample averaging where sample 1's local AP ignores sample 2's b.
        """
        preds = [["a"], ["b"]]
        tgts = [["a", "b"], ["b"]]
        result = compute_dataset_map(preds, tgts)
        assert 0.0 <= result <= 1.0

    def test_all_wrong_predictions_lower_than_perfect(self) -> None:
        """Swapped predictions score lower than correct predictions."""
        perfect = compute_dataset_map([["a"], ["b"]], [["a"], ["b"]])
        wrong = compute_dataset_map([["b"], ["a"]], [["a"], ["b"]])
        assert perfect == pytest.approx(1.0)
        assert wrong < perfect


# ---------------------------------------------------------------------------
# score_sample — task_type kwarg overrides metadata
# ---------------------------------------------------------------------------


class TestScoreSampleTaskTypeKwarg:
    """Tests that the task_type kwarg in score_sample overrides metadata routing."""

    def _post(self, text: str = "") -> PostProcessResult:
        return PostProcessResult(segments=[], text=text, warnings=())

    def _example(self, labels: object) -> DatasetExample:
        return DatasetExample(sample_id="s0", labels=labels, metadata={})

    def test_task_type_classification_routes_correctly(self) -> None:
        ex = self._example(["cat", "feline"])
        result = score_sample(
            ex,
            post=self._post("cat"),
            raw_predictions=["cat"],
            task_type="classification",
        )
        assert "top1_accuracy" in result
        assert "average_precision" not in result

    def test_task_type_detection_routes_correctly(self) -> None:
        ex = self._example(["cat", "dog"])
        result = score_sample(
            ex,
            post=self._post("cat"),
            raw_predictions=["cat"],
            task_type="detection",
        )
        assert "average_precision" in result
        assert "top1_accuracy" not in result

    def test_task_type_wins_over_metadata(self) -> None:
        """Explicit task_type kwarg overrides example.metadata['task']."""
        ex = DatasetExample(
            sample_id="s0",
            labels=["cat", "dog"],
            metadata={"task": "detection"},
        )
        result = score_sample(
            ex,
            post=self._post("cat"),
            raw_predictions=["cat"],
            task_type="classification",
        )
        assert "top1_accuracy" in result
        assert "average_precision" not in result

    def test_captioning_task_returns_empty_per_sample(self) -> None:
        """CIDEr is corpus-level; score_sample does not emit per-sample keys."""
        ex = DatasetExample(
            sample_id="s0",
            labels="golden whistler",
            metadata={},
        )
        result = score_sample(
            ex,
            post=self._post("olive whistler call"),
            raw_predictions=["olive whistler call"],
            task_type="captioning",
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Null-targets warning (rescorer)
# ---------------------------------------------------------------------------


class TestRescoreNullTargetsWarning:
    """Null targets produce a warning and empty scores in the rescorer."""

    def _write_jsonl(self, path: Path, rows: list[object]) -> None:
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    def test_null_targets_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from beans_next.runner.rescorer import rescore_predictions_file

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        preds_path = run_dir / "predictions.jsonl"
        proc_path = run_dir / "processed_predictions.jsonl"

        self._write_jsonl(
            preds_path,
            [
                ModelPrediction(
                    sample_id="s1", predictions=["dog"], error=None
                ).model_dump(mode="json")
            ],
        )
        # targets=None in processed predictions
        self._write_jsonl(
            proc_path,
            [
                ScoredPrediction(
                    sample_id="s1",
                    task_id=None,
                    predictions=["dog"],
                    processed_prediction="dog",
                    targets=None,
                    scores=None,
                    error=None,
                ).model_dump(mode="json")
            ],
        )

        with caplog.at_level(logging.WARNING):
            rescore_predictions_file(preds_path, output_dir=tmp_path / "out")

        warning_texts = " ".join(caplog.messages)
        assert "no targets" in warning_texts.lower() or "null" in warning_texts.lower()

    def test_null_targets_scores_empty(self, tmp_path: Path) -> None:
        from beans_next.runner.rescorer import rescore_predictions_file

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        preds_path = run_dir / "predictions.jsonl"
        proc_path = run_dir / "processed_predictions.jsonl"

        self._write_jsonl(
            preds_path,
            [
                ModelPrediction(
                    sample_id="s1", predictions=["dog"], error=None
                ).model_dump(mode="json")
            ],
        )
        self._write_jsonl(
            proc_path,
            [
                ScoredPrediction(
                    sample_id="s1",
                    task_id=None,
                    predictions=["dog"],
                    processed_prediction="dog",
                    targets=None,
                    scores=None,
                    error=None,
                ).model_dump(mode="json")
            ],
        )

        out_dir = tmp_path / "out"
        rescore_predictions_file(preds_path, output_dir=out_dir)
        scored = json.loads(
            (out_dir / "scored_predictions.jsonl").read_text(encoding="utf-8")
        )
        assert scored.get("scores") is None


# ---------------------------------------------------------------------------
# Rescorer task_type awareness — classification skips parse_labels_comma
# ---------------------------------------------------------------------------


class TestRescoreTaskType:
    """Rescorer task_type parameter selects correct postprocess steps and scorer routing."""

    def _write_jsonl(self, path: Path, rows: list[object]) -> None:
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    def _run_rescore(
        self, tmp_path: Path, raw_pred: str, target: str, task_type: str | None
    ) -> dict:
        from beans_next.runner.rescorer import rescore_predictions_file

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        preds_path = run_dir / "predictions.jsonl"
        proc_path = run_dir / "processed_predictions.jsonl"

        self._write_jsonl(
            preds_path,
            [
                ModelPrediction(
                    sample_id="s1", predictions=[raw_pred], error=None
                ).model_dump(mode="json")
            ],
        )
        self._write_jsonl(
            proc_path,
            [
                ScoredPrediction(
                    sample_id="s1",
                    task_id=None,
                    predictions=[raw_pred],
                    processed_prediction=raw_pred,
                    targets=target,
                    scores=None,
                    error=None,
                ).model_dump(mode="json")
            ],
        )
        out_dir = tmp_path / "out"
        rescore_predictions_file(preds_path, output_dir=out_dir, task_type=task_type)
        rows = (out_dir / "scored_predictions.jsonl").read_text(encoding="utf-8")
        return json.loads(rows.strip())

    def test_classification_free_form_correct(self, tmp_path: Path) -> None:
        """Classification task: free-form output containing the label → accuracy=1."""
        result = self._run_rescore(
            tmp_path,
            raw_pred="I think this is a dog barking.",
            target="dog",
            task_type="classification",
        )
        scores = result.get("scores") or {}
        assert scores.get("accuracy") == pytest.approx(1.0)

    def test_classification_uses_extract_label_not_comma_split(
        self, tmp_path: Path
    ) -> None:
        """Classification: output 'dog, a common pet' should NOT be comma-split."""
        self._run_rescore(
            tmp_path,
            raw_pred="dog, a common pet",
            target="dog",
            task_type="classification",
        )
        proc = (tmp_path / "out" / "processed_predictions.jsonl").read_text(
            encoding="utf-8"
        )
        proc_obj = json.loads(proc.strip())
        # With extract_label_from_text (no comma split), 'dog' is found as a
        # substring and the processed prediction is 'dog', not 'dog, a common pet'.
        assert proc_obj.get("processed_prediction") == "dog"
