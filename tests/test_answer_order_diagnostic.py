from __future__ import annotations

from argparse import Namespace

from beans_next.api.types import DatasetExample
from beans_next.diagnostics import permute_answer_order
from beans_next.runner.runner import (
    _finalize_loaded_examples,
    _sample_examples_stratified_by_label,
)


def test_permute_inline_options_remaps_reference() -> None:
    example = DatasetExample(
        sample_id="inline-1",
        labels="b",
        metadata={
            "instruction": "Question?\nOptions: a) alpha, b) beta, c) gamma"
        },
    )
    result = permute_answer_order(example, seed=7)

    assert result.metadata["instruction"] != example.metadata["instruction"]
    assert result.metadata["answer_order_permutation"] != [0, 1, 2]
    options = result.metadata["instruction"].split("Options: ", 1)[1]
    correct_text = options.split(", ")[ord(str(result.labels)) - ord("a")]
    assert correct_text.endswith("beta")


def test_permute_multiline_options_preserves_suffix_and_remaps() -> None:
    example = DatasetExample(
        sample_id="lines-1",
        labels="C",
        metadata={
            "instruction": (
                "Question?\n\nA: alpha\nB: beta\nC: gamma\nD: delta\n\n"
                "Answer with the letter of the correct choice (A, B, C, D)."
            )
        },
    )
    result = permute_answer_order(example, seed=11)

    assert "Answer with the letter" in result.metadata["instruction"]
    option_lines = [
        line
        for line in result.metadata["instruction"].splitlines()
        if len(line) > 2 and line[1:3] == ": "
    ]
    correct_line = option_lines[ord(str(result.labels)) - ord("A")]
    assert correct_line.endswith("gamma")


def test_audio_only_choices_are_unchanged() -> None:
    example = DatasetExample(
        sample_id="audio-only",
        labels="B",
        metadata={"instruction": "A: <AudioHere>\nB: <AudioHere>"},
    )
    assert permute_answer_order(example, seed=0) == example


def test_stratified_sample_is_exact_deterministic_and_balanced() -> None:
    examples = [
        DatasetExample(sample_id=f"a-{index}", labels="a") for index in range(80)
    ] + [DatasetExample(sample_id=f"b-{index}", labels="b") for index in range(20)]

    first = _sample_examples_stratified_by_label(examples, fraction=0.1, seed=9)
    second = _sample_examples_stratified_by_label(examples, fraction=0.1, seed=9)

    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert len(first) == 10
    assert sum(row.labels == "a" for row in first) == 8
    assert sum(row.labels == "b" for row in first) == 2


def test_finalize_applies_sampling_before_permutation() -> None:
    examples = [
        DatasetExample(
            sample_id=f"s-{index}",
            labels="a",
            metadata={"instruction": "Q?\nOptions: a) yes, b) no"},
        )
        for index in range(20)
    ]
    args = Namespace(
        sample_fraction=0.1,
        seed=3,
        permute_answer_order=True,
        stratify_by_label=True,
    )
    results = _finalize_loaded_examples(examples, args=args)

    assert len(results) == 2
    assert all("answer_order_permutation" in row.metadata for row in results)
