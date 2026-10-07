"""Hand-calculated coverage and missing-species checks."""

import pytest

from beans_next.api.types import DatasetExample
from beans_next.metrics import score_sample
from beans_next.metrics.species_counts import parse_species_counts
from beans_next.post_process.pipeline import PostProcessResult


def score(raw: str, target: str = "Parus major: 3, Turdus merula: 1") -> dict:
    return dict(
        score_sample(
            DatasetExample(sample_id="x", labels=target),
            post=PostProcessResult([raw], raw),
            raw_predictions=[raw],
            task_type="species_count_dict",
        )
    )


@pytest.mark.parametrize(
    "raw",
    [
        "unrelated response",
        "I cannot count these",
        "3",
        "Parus major",
        "",
        "Parus major: -2",
        "Parus major: 1 or 2",
        "Parus major: 1, Parus major: 2",
    ],
)
def test_invalid_is_not_zero(raw: str) -> None:
    assert parse_species_counts(raw) is None
    result = score(raw)
    assert result["parse_success"] == 0
    assert "count_mae" not in result


@pytest.mark.parametrize(
    "raw",
    [
        "None",
        "none.",
        "No calls",
        "There are no vocalizations in this recording.",
        "There are no vocalizations from any species in this recording.",
        "{}",
    ],
)
def test_explicit_absence_is_valid_but_can_be_wrong(raw: str) -> None:
    assert parse_species_counts(raw) == {}
    assert score(raw)["count_mae"] == 2
    assert score(raw)["parse_success"] == 1


def test_missing_and_hallucinated_species_are_penalized() -> None:
    assert score("Parus major: 3")["count_mae"] == 0.5
    assert score("Parus major: 3, Extra species: 2")["count_mae"] == 1
    assert score("None", "None")["count_mae"] == 0


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Parus major: 3, Turdus merula: 1", {"parus major": 3, "turdus merula": 1}),
        (
            "Based on the audio:\n* **Parus major** (Great Tit): 3 calls",
            {"parus major": 3},
        ),
        ("Great Tit (Parus major): three calls", {"parus major": 3}),
        (
            "There is one call from a Great Tit (Parus major) and "
            "2 calls from a Blackbird (Turdus merula).",
            {"parus major": 1, "turdus merula": 2},
        ),
        (
            "There is only one call heard in this recording: "
            "that of a Great Tit (Parus major).",
            {"parus major": 1},
        ),
    ],
)
def test_readable_formats(raw: str, expected: dict) -> None:
    assert parse_species_counts(raw) == expected


def test_target_does_not_resolve_unlabeled_count() -> None:
    for target in ["Parus major: 3", "Turdus merula: 3"]:
        assert score("3", target)["parse_success"] == 0


def test_comma_separated_prose_preserves_every_species() -> None:
    assert parse_species_counts(
        "2 calls from Parus major, 1 call from Turdus merula."
    ) == {"parus major": 2, "turdus merula": 1}


def test_dash_count_and_explicit_audible_absence() -> None:
    assert parse_species_counts("Parus major (Great Tit) - 1 vocalization") == {
        "parus major": 1
    }
    assert (
        parse_species_counts(
            "There are no calls from any species audible in the audio."
        )
        == {}
    )
