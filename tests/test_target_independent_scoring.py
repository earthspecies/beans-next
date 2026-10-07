"""Scientific invariants for paper scoring, with hand-calculated expectations."""

from pathlib import Path

import pytest

from beans_next.api.types import DatasetExample
from beans_next.metrics import score_sample
from beans_next.metrics.base import MetricsError
from beans_next.metrics.regression import extract_numeric_value
from beans_next.post_process.answers import parse_choice_set, question_options
from beans_next.post_process.pipeline import (
    PostProcessResult,
    run_post_process_pipeline,
)
from beans_next.runner._utils import compute_dataset_level_metrics
from beans_next.runner.rescorer import _default_postprocess_steps
from beans_next.runner.runner import _postprocess_steps_for_examples


def score(pred: str, target: str, kind: str, question: str = "") -> dict[str, float]:
    return dict(
        score_sample(
            DatasetExample(
                sample_id="x", labels=target, metadata={"instruction": question}
            ),
            post=PostProcessResult([pred], pred),
            raw_predictions=[pred],
            task_type=kind,
        )
    )


@pytest.mark.parametrize("target", [30, 300, 3000, 30000])
def test_frequency_interpretation_independent_of_target(target: int) -> None:
    assert extract_numeric_value("2000-4000 Hz", unit="hz", target_value=target) == 3000


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.5 kHz to 4.5 kHz", 3000),
        ("-10 dB", -10),
        ("1,654.75 Hz", 1654.75),
        ("2.5", 2.5),
        ("three calls", 3),
        ("#0.0s - 10.0s#: 350 Hz", 350),
    ],
)
def test_unambiguous_numeric_formats(text: str, expected: float) -> None:
    assert extract_numeric_value(text) == expected


@pytest.mark.parametrize(
    "text", ["2 species and 7 calls", "unknown", "1 or 2", "10 seconds"]
)
def test_reject_ambiguous_or_unrelated_numbers(text: str) -> None:
    with pytest.raises(MetricsError):
        extract_numeric_value(text)


@pytest.mark.parametrize(
    "kind,targets,pred",
    [
        ("regression", ["100 Hz", "300 Hz"], "200 Hz"),
        ("regression", ["2", "3"], "2.5"),
        ("species_count_dict", ["Parus major: 2", "Parus major: 3"], "Parus major: 7"),
    ],
)
def test_online_offline_preserve_answer(
    kind: str, targets: list[str], pred: str
) -> None:
    examples = [
        DatasetExample(sample_id=str(i), labels=t) for i, t in enumerate(targets)
    ]
    for parsers, cleaners in [
        _postprocess_steps_for_examples(examples, kind),
        _default_postprocess_steps(targets=targets, task_type=kind),
    ]:
        assert (
            run_post_process_pipeline(
                pred, parser_steps=parsers, cleaner_steps=cleaners
            ).text
            == pred
        )


@pytest.mark.parametrize(
    "pred", ["gibberish", "Parus major", "Parus major: 2 calls", "Unknown"]
)
def test_invalid_or_nonempty_summary_not_correct_absence(pred: str) -> None:
    assert score(pred, "None", "species_summary")["species_f1"] == 0


def test_explicit_absence_is_valid() -> None:
    assert (
        score("No species are present.", "None", "species_summary")["species_f1"] == 1
    )


def test_species_score_independent_of_attribute_format() -> None:
    target = "Parus major: 2 calls, 100-200 Hz; Sturnus unicolor: 3 calls, 200-300 Hz"
    assert score("Parus major", target, "species_summary")[
        "species_f1"
    ] == pytest.approx(2 / 3)
    assert (
        score("Great Tit", "Parus major: 2 calls, 100-200 Hz", "species_summary")[
            "species_f1"
        ]
        == 1
    )


def test_unknown_only_target_excluded_not_empty_correct() -> None:
    scores = score("gibberish", "Unknown: 1 call, 100-200 Hz", "species_summary")
    assert scores == {"target_parse_success": 0}


QUESTION = "Which species?\nOptions: a) Galerida theklae, b) Parus major"


@pytest.mark.parametrize(
    "pred", ["B", "(B)", "Parus major", "The answer is Parus major", "B: Parus major"]
)
def test_mcq_letter_and_option_text(pred: str) -> None:
    assert score(pred, "b", "classification", QUESTION)["accuracy"] == 1


@pytest.mark.parametrize(
    "pred",
    [
        "Not Galerida theklae. Choose B.",
        "B: Galerida theklae",
        "Galerida theklae or Parus major",
        "A, B",
    ],
)
def test_mcq_reject_conflict_negation_and_multiple_options(pred: str) -> None:
    assert score(pred, "a", "classification", QUESTION)["accuracy"] == 0


def test_option_permutation_preserves_semantic_answer() -> None:
    other = "Which species?\nOptions: a) Parus major, b) Galerida theklae"
    assert score("Parus major", "a", "classification", other)["accuracy"] == 1
    assert score("Parus major", "b", "classification", QUESTION)["accuracy"] == 1


def test_mcq_never_reads_assistant_as_option() -> None:
    assert (
        question_options({"messages": [{"role": "assistant", "content": QUESTION}]})
        == {}
    )
    assert parse_choice_set("Parus major") is None


def test_multilabel_accuracy_preserves_complete_set() -> None:
    assert score("C, A", "A, C", "multilabel_accuracy")["accuracy"] == 1
    assert score("A", "A, C", "multilabel_accuracy")["accuracy"] == 0
    assert score("None", "None", "multilabel_accuracy")["accuracy"] == 1
    assert score("gibberish", "None", "multilabel_accuracy")["accuracy"] == 0


def test_macro_f1_is_over_five_call_types() -> None:
    # song: TP=1,FN=1 -> 2/3; call: TP=1 -> 1; three absent classes -> 0.
    result = compute_dataset_level_metrics(
        [("song, call", "song, call"), ("None", "song")], "multilabel_classification"
    )
    assert result["macro_f1"] == pytest.approx((2 / 3 + 1) / 5)


def test_specific_call_type_is_not_generic_call() -> None:
    result = score("alarm call", "alarm call, call", "multilabel_classification")
    assert result["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "answer",
    ["Great Tit", "The highest pitch is produced by b) Parus major. It sings."],
)
def test_mcq_alias_and_explicit_prose(answer: str) -> None:
    assert score(answer, "b", "classification", QUESTION)["accuracy"] == 1


def test_prose_with_conflicting_letter_and_species_rejected() -> None:
    assert (
        score(
            "The highest pitch is produced by a) Parus major.",
            "a",
            "classification",
            QUESTION,
        )["accuracy"]
        == 0
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Species B",
        "The recording best matches Call type B.",
        "The recording contains sound B.",
        "Final Answer: B",
        "Therefore, B is the best match.",
    ],
)
def test_audio_mcq_explicit_selection_in_prose(answer: str) -> None:
    question = "A: <Audio><AudioHere></Audio>\nB: <Audio><AudioHere></Audio>"
    assert score(answer, "B", "classification", question)["accuracy"] == 1


def test_multilabel_does_not_drop_recognized_labels_due_to_one_bad_item() -> None:
    from beans_next.post_process.answers import INVALID_ANSWER, parse_call_types

    assert parse_call_types("song, call, unrecognized") == {
        "song",
        "call",
        INVALID_ANSWER,
    }
    result = compute_dataset_level_metrics(
        [("song, call, unrecognized", "song, call")], "multilabel_classification"
    )
    assert result["macro_f1"] == pytest.approx(2 / 5)


def test_f0_symbol_is_not_a_zero_prediction() -> None:
    with pytest.raises(MetricsError):
        extract_numeric_value("I cannot estimate f0 from this audio", unit="hz")
    assert extract_numeric_value("f0 is 200 Hz", unit="hz") == 200


def test_offline_rescorer_preserves_question_and_version(tmp_path: Path) -> None:
    import json

    from beans_next.api.types import ModelPrediction, ScoredPrediction
    from beans_next.post_process.answers import SCORING_VERSION
    from beans_next.runner.rescorer import rescore_predictions_file

    raw = ModelPrediction(sample_id="x", predictions=["Great Tit"])
    row = ScoredPrediction(
        sample_id="x",
        predictions=["Great Tit"],
        processed_prediction="a",
        targets="b",
        question=QUESTION,
    )
    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(raw.model_dump(mode="json")) + "\n")
    (tmp_path / "processed_predictions.jsonl").write_text(
        json.dumps(row.model_dump(mode="json")) + "\n"
    )
    summary = rescore_predictions_file(
        source, output_dir=tmp_path / "out", task_type="classification"
    )
    saved = json.loads((tmp_path / "out/scored_predictions.jsonl").read_text())
    assert saved["scores"]["accuracy"] == 1
    assert saved["question"] == QUESTION
    assert summary.scorer_versions == {"deterministic": SCORING_VERSION}


def test_detection_macro_f1_is_per_label_not_exact_set() -> None:
    # A: TP=1,FN=1 -> 2/3; B: TP=1,FP=1 -> 2/3.
    pairs = [("A, B", "A"), ("B", "A, B"), ("None", "None")]
    assert compute_dataset_level_metrics(pairs, "multilabel_detection")[
        "macro_f1"
    ] == pytest.approx(2 / 3)
    assert compute_dataset_level_metrics(list(reversed(pairs)), "multilabel_detection")[
        "macro_f1"
    ] == pytest.approx(2 / 3)


def test_detection_none_is_not_a_positive_class() -> None:
    pairs = [("None", "A"), ("None", "B"), ("None", "None")]
    assert compute_dataset_level_metrics(pairs, "multilabel_detection")["macro_f1"] == 0


def test_detection_task_preserves_all_predicted_labels() -> None:
    assert score("A, C", "A", "multilabel_detection")["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "text,unit,expected",
    [
        ("There is only one bird heard in this clip.", None, 1),
        ("There are 2 different species.\n1. Sparrow\n2. Robin", None, 2),
        ("There are no bird flight calls in this clip.", None, 0),
        ("twentyseven", None, 27),
        ("-33,29 dB", "db", -33.29),
        ("1,200 Hz", "hz", 1200),
        ("F0: 1.5 Hz, F1: 6.5 Hz, F2: 7.5 Hz", "hz", 1.5),
    ],
)
def test_explicit_numeric_answers(text: str, unit: str | None, expected: float) -> None:
    for target in (1, 50, 10000):
        assert extract_numeric_value(text, unit=unit, target_value=target) == expected


@pytest.mark.parametrize(
    "text,unit",
    [
        ("There are 2 species and 7 calls.", None),
        ("The fundamental frequency is 120 Hz or 200 Hz.", "hz"),
        ("The SNR is high.", "db"),
        ("Use Fourier analysis to estimate the frequency.", "hz"),
    ],
)
def test_ambiguous_or_missing_numeric_answers(text: str, unit: str | None) -> None:
    with pytest.raises(MetricsError):
        extract_numeric_value(text, unit=unit)


@pytest.mark.parametrize(
    "text",
    [
        "To calculate SNR use 10 * log10(Psignal/Pnoise).",
        "I cannot estimate SNR. 1. Obtain the recording.",
    ],
)
def test_formula_and_instruction_numbers_are_not_measurements(text: str) -> None:
    with pytest.raises(MetricsError):
        extract_numeric_value(text, unit="db")


@pytest.mark.parametrize(
    "text,unit,expected",
    [
        ("105 Hertz (Hz)", "hz", 105),
        ("SNR = 10 (poor)", "db", 10),
    ],
)
def test_spelled_units_and_labeled_scalar(
    text: str, unit: str | None, expected: float
) -> None:
    assert extract_numeric_value(text, unit=unit) == expected


@pytest.mark.parametrize("text", ["F1", "A1", "SNR 80%"])
def test_label_digits_and_percent_are_not_snr_or_f0(text: str) -> None:
    with pytest.raises(MetricsError):
        extract_numeric_value(text, unit="db")


@pytest.mark.parametrize(
    "text",
    [
        "The SNR could range from 0 dB to",
        "The SNR could range from 0 dB",
        "The SNR is above 10 dB.",
        "There are at least 3 species.",
    ],
)
def test_incomplete_ranges_and_bounds(text: str) -> None:
    with pytest.raises(MetricsError):
        extract_numeric_value(text, unit="db" if "SNR" in text else None)


def test_complete_answer_survives_truncated_explanation() -> None:
    assert (
        extract_numeric_value(
            "The fundamental frequency is 120 Hz. Explanation: 1. Pitch and", unit="hz"
        )
        == 120
    )
