"""Formatting extensions retain submission scoring and run-specific choices."""

import json
from pathlib import Path

import pytest

from beans_next.api.types import ModelPrediction, ScoredPrediction
from beans_next.post_process.answers import (
    INVALID_ANSWER,
    is_presence_nonanswer,
    parse_call_types,
    parse_choice_set,
    question_options,
)
from beans_next.runner.rescorer import rescore_predictions_file


@pytest.mark.parametrize(
    "answer",
    [
        "alarm call, call, begging call",
        "The correct answer is:\n- alarm call\n- call\n- begging call",
        "**Answer:**\n* alarm call\n* call\n* begging call",
        "alarm call\ncall\nbegging call",
    ],
)
def test_equivalent_call_type_formats(answer: str) -> None:
    assert parse_call_types(answer) == {"alarm call", "call", "begging call"}


@pytest.mark.parametrize(
    "answer", ["The correct answer is: call.", "Answer: call", "call"]
)
def test_prefixed_single_call(answer: str) -> None:
    assert parse_call_types(answer) == {"call"}


@pytest.mark.parametrize(
    "answer",
    [
        "No alarm call is present",
        "I cannot hear the recording. Alarm call: a warning sound.",
        "- Alarm call: a warning sound\n- Flight call: a sound during flight",
        "alarm call or flight call",
    ],
)
def test_prose_is_not_a_call_type_selection(answer: str) -> None:
    assert parse_call_types(answer) is None


def test_unknown_list_item_and_specific_call_semantics_unchanged() -> None:
    assert parse_call_types("alarm call") == {"alarm call"}
    assert parse_call_types("song, call, unrecognized") == {
        "song",
        "call",
        INVALID_ANSWER,
    }
    assert parse_call_types("None") == set()


@pytest.mark.parametrize(
    "answer",
    [
        "I'm sorry, but I cannot determine if there is a bird without listening.",
        "I cannot tell whether there is a bird in this recording.",
        "To answer your question, I would need more information about the recording. "
        "Please provide more context.",
        "You haven't provided a recording. Please provide the recording.",
        "However, I don't see any recording. Please provide more context.",
        "To answer your question, I would need more information. "
        "Please provide the recording.",
        "To determine this, I would need audio. Since I can't directly access files, "
        "please describe it.",
        "Unfortunately, I don't have the ability to access or listen to "
        "audio recordings.",
    ],
)
def test_explicit_refusal_does_not_become_negative_label(answer: str) -> None:
    assert is_presence_nonanswer(answer, "presence_binary")
    assert not is_presence_nonanswer(answer, "classification")


@pytest.mark.parametrize(
    "answer",
    [
        "Yes",
        "No.",
        "No, there is no bird in the recording.",
        "There is a bird vocalizing in the recording.",
        "I cannot hear clearly. My best guess is: Yes.",
        "I cannot determine the species. Yes, a bird is audible.",
    ],
)
def test_ordinary_binary_answers_keep_existing_extraction(answer: str) -> None:
    assert not is_presence_nonanswer(answer, "presence_binary")


def test_rescore_refusal_wrong_and_normal_yes_no_preserved(tmp_path: Path) -> None:
    answers = ["I cannot determine whether a bird is present.", "No", "Yes"]
    raw = [
        ModelPrediction(sample_id=str(i), predictions=[a])
        for i, a in enumerate(answers)
    ]
    processed = [
        ScoredPrediction(
            sample_id=str(i), predictions=[a], targets=["No", "No", "Yes"][i]
        )
        for i, a in enumerate(answers)
    ]
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        "".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in raw)
    )
    (tmp_path / "processed_predictions.jsonl").write_text(
        "".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in processed)
    )
    result = rescore_predictions_file(
        source, output_dir=tmp_path / "out", task_type="presence_binary"
    )
    assert result.metrics["mean"]["accuracy"] == pytest.approx(2 / 3)
    scored = [
        json.loads(line)
        for line in (tmp_path / "out/scored_predictions.jsonl").read_text().splitlines()
    ]
    assert scored[0]["processed_prediction"] == INVALID_ANSWER
    assert scored[0]["scores"]["accuracy"] == 0
    assert result.n_errors == 0  # Refusal is a model response, not transport failure.


def test_saved_options_not_current_shuffled_options() -> None:
    old = {"instruction": "Which species?\nA: Parus major\nB: Galerida theklae"}
    shuffled = {"instruction": "Which species?\nA: Galerida theklae\nB: Parus major"}
    assert parse_choice_set("Parus major", question_options(old)) == {"a"}
    assert parse_choice_set("Parus major", question_options(shuffled)) == {"b"}
    assert parse_choice_set("A", question_options(old)) == {"a"}
