"""Tests for answer-first MCQ extraction.

Some models state the chosen option before explaining it
(``"C: A crow cawing in flight."``). The trailing-letter heuristic reads that
as "A", from the article in "A crow". These tests pin the leading-label rule
and, just as importantly, that the styles other models use are unchanged.
"""

from __future__ import annotations

from beans_next.post_process.cleaners import apply_extract_mcq_choice_from_text
from beans_next.post_process.pipeline import PostProcessContext

_LABELS = ["A", "B", "C", "D"]


def _pick(text: str) -> str:
    ctx = apply_extract_mcq_choice_from_text(
        PostProcessContext(segments=[text], warnings=()), labels=_LABELS
    )
    return ctx.segments[0]


def test_answer_first_with_article_a_in_prose() -> None:
    """The regression this rule exists for."""
    assert _pick("C: A crow cawing in flight.") == "C"
    assert _pick("D: A crow cawing in the distance.") == "D"
    assert _pick("B: A dog is barking nearby.") == "B"


def test_answer_first_variants() -> None:
    assert _pick("C. A bird call.") == "C"
    assert _pick("**B**: dog barking") == "B"
    assert _pick("A) Some description") == "A"


def test_bare_letter_unchanged() -> None:
    """NatureLM-style terse answers."""
    assert _pick("B") == "B"
    assert _pick("**D**") == "D"
    assert _pick("c") == "C"


def test_explicit_marker_still_wins() -> None:
    """Qwen-style chain-of-thought ending in an explicit answer."""
    text = (
        "The recording contains rapid barks.\n\nAmong the four call types:\n"
        "- **A**: a growl\n- **B**: a yelp\n- **C**: a howl\n- **D**: rapid barks\n\n"
        "✅ **Answer: D**"
    )
    assert _pick(text) == "D"


def test_trailing_bold_letter_unchanged() -> None:
    """Qwen-style answer with no explicit marker word."""
    text = (
        "The recording features a series of rapid, high-pitched barks, "
        "which most closely matches:\n\n**D**."
    )
    assert _pick(text) == "D"


def test_enumeration_starting_with_a_label_does_not_take_the_first() -> None:
    """An enumeration must not be read as an answer-first response."""
    text = "A: growl\nB: yelp\nC: howl\nD: barks\n\nThe correct match is D."
    assert _pick(text) == "D"


def test_prose_first_output_is_unaffected_by_the_new_rule() -> None:
    text = "The recording contains a dog bark, which matches option C."
    assert _pick(text) == "C"
