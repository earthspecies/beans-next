"""T3 post-processing fixtures: raw model output → processed_prediction.

Each test case pins the pipeline output for a specific model output format
against a known T3 task type (binary, MCQ, count OE). The step specs mirror
what ``_postprocess_steps_for_examples`` configures for each vocab shape.
"""

from __future__ import annotations

import pytest

from beans_next.post_process.pipeline import StepSpec, run_post_process_pipeline

# ---------------------------------------------------------------------------
# Shared step specs (mirroring _postprocess_steps_for_examples logic)
# ---------------------------------------------------------------------------

_BINARY_CLEANERS = (
    StepSpec("normalize_whitespace", {}),
    StepSpec("strip_eos", {}),
    StepSpec("extract_label_from_text", {"labels": ("Yes", "No")}),
)

_MCQ_ABCD_CLEANERS = (
    StepSpec("normalize_whitespace", {}),
    StepSpec("strip_eos", {}),
    StepSpec("extract_mcq_choice_from_text", {"labels": ("a", "b", "c", "d")}),
)

_COUNT_CLEANERS = (
    StepSpec("normalize_whitespace", {}),
    StepSpec("strip_eos", {}),
    StepSpec(
        "extract_label_from_text",
        {"labels": ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")},
    ),
)


@pytest.mark.parametrize(
    ("raw", "parser_steps", "cleaner_steps", "expected"),
    [
        # ── NatureLM v1.0 ────────────────────────────────────────────────────
        # v1.0 prefixes every answer with a timestamp segment marker.
        (
            "#0.00s - 10.00s#: No\n",
            (),
            _BINARY_CLEANERS,
            "No",
        ),
        (
            "#0.00s - 10.00s#: Yes\n",
            (),
            _BINARY_CLEANERS,
            "Yes",
        ),
        # MCQ: timestamp prefix + "letter) species" — letter extracted.
        (
            "#0.00s - 10.00s#: b) Cisticola juncidis\n",
            (),
            _MCQ_ABCD_CLEANERS,
            "b",
        ),
        (
            "#0.00s - 10.00s#: a) Galerida theklae\n",
            (),
            _MCQ_ABCD_CLEANERS,
            "a",
        ),
        # Count OE: timestamp prefix + bare number.
        (
            "#0.00s - 10.00s#: 3\n",
            (),
            _COUNT_CLEANERS,
            "3",
        ),
        # ── NatureLM v1.1 ────────────────────────────────────────────────────
        # v1.1 emits "None" for tasks it cannot parse (e.g. binary presence).
        # Levenshtein('none', 'no') = 2 < Levenshtein('none', 'yes') = 3.
        (
            "None",
            (),
            _BINARY_CLEANERS,
            "No",
        ),
        # "None" for count tasks: all single-digit labels have the same distance
        # from "none" (=3 deletions); lexicographically smallest wins → "1".
        (
            "None",
            (),
            _COUNT_CLEANERS,
            "1",
        ),
        # MCQ without timestamp: "letter) species" — letter extracted.
        (
            "c) Cisticola juncidis",
            (),
            _MCQ_ABCD_CLEANERS,
            "c",
        ),
        # ── Audio Flamingo 3 (AF3) ────────────────────────────────────────────
        # AF3 emits lowercase binary answers.
        (
            "no",
            (),
            _BINARY_CLEANERS,
            "No",
        ),
        (
            "yes",
            (),
            _BINARY_CLEANERS,
            "Yes",
        ),
        # MCQ: clean "letter) species" format.
        (
            "a) Galerida theklae",
            (),
            _MCQ_ABCD_CLEANERS,
            "a",
        ),
        # Count OE: bare number.
        (
            "1",
            (),
            _COUNT_CLEANERS,
            "1",
        ),
        # ── Qwen3-Omni ───────────────────────────────────────────────────────
        # Qwen3-Omni produces verbose reasoning; binary label extracted via
        # substring scan (the first word of the answer before the comma).
        (
            (
                "No, there are no bird flight calls present in this recording. "
                "The audio consists of a continuous, low-frequency buzzing sound."
            ),
            (),
            _BINARY_CLEANERS,
            "No",
        ),
        (
            (
                "Yes, the calls of both Catharus ustulatus and Vireo olivaceus "
                "can be heard simultaneously in the audio clip."
            ),
            (),
            _BINARY_CLEANERS,
            "Yes",
        ),
        # MCQ: same "letter) species" pattern as other models.
        (
            "b) Cisticola juncidis",
            (),
            _MCQ_ABCD_CLEANERS,
            "b",
        ),
        # Count OE: number embedded in a full sentence.
        (
            "There are 2 species audible in this recording.",
            (),
            _COUNT_CLEANERS,
            "2",
        ),
        # Count with markdown bold around the number.
        (
            (
                "There are **3** distinct species of birds in this clip.\n\n"
                "1. American Crow (Corvus brachyrhynchos)\n"
                "2. Tufted Titmouse (Baeolophus bicolor)\n"
                "3. Blue Jay (Cyanocitta cristata)"
            ),
            (),
            _COUNT_CLEANERS,
            "3",
        ),
    ],
    ids=[
        "nlm_v1_0/binary_no",
        "nlm_v1_0/binary_yes",
        "nlm_v1_0/mcq_b",
        "nlm_v1_0/mcq_a",
        "nlm_v1_0/count_3",
        "nlm_v1_1/binary_none_to_no",
        "nlm_v1_1/count_none_to_1",
        "nlm_v1_1/mcq_c",
        "af3/binary_no_lowercase",
        "af3/binary_yes_lowercase",
        "af3/mcq_a",
        "af3/count_1",
        "qwen3_omni/binary_no_verbose",
        "qwen3_omni/binary_yes_verbose",
        "qwen3_omni/mcq_b",
        "qwen3_omni/count_2_sentence",
        "qwen3_omni/count_3_markdown_bold",
    ],
)
def test_t3_postprocess_raw_to_processed(
    raw: str,
    parser_steps: tuple[StepSpec, ...],
    cleaner_steps: tuple[StepSpec, ...],
    expected: str,
) -> None:
    result = run_post_process_pipeline(
        raw, parser_steps=parser_steps, cleaner_steps=cleaner_steps
    )
    assert result.text == expected
