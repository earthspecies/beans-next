"""Tests for conservative detection label post-processing."""

from __future__ import annotations

from beans_next.post_process.pipeline import StepSpec, run_post_process_pipeline


def test_conservative_detection_match_drops_unmatched_and_lowercases() -> None:
    labels = ("Hermit Thrush", "Northern Cardinal", "None")
    post = run_post_process_pipeline(
        "hermit thrush, definitely a northern cardnal, totally unknown thing",
        parser_steps=(StepSpec("parse_labels_comma", {}),),
        cleaner_steps=(
            StepSpec(
                "conservative_match_to_labels",
                {
                    "labels": labels,
                    "max_distance": 2,
                    "allow_fuzzy": True,
                    "drop_unmatched": True,
                },
            ),
        ),
    )
    assert post.segments == ["Hermit Thrush", "Northern Cardinal"]
