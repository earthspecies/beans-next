"""Tests for BirdSet scientific-name extraction post-processing."""

from __future__ import annotations

from beans_next.post_process.cleaners import apply_extract_scientific_name_from_text
from beans_next.post_process.pipeline import PostProcessContext


def test_extract_scientific_name_from_text_handles_timestamps_and_prose() -> None:
    labels = [
        "Turdus philomelos",
        "Jynx torquilla",
        "Dendrocopos major",
    ]
    ctx = PostProcessContext(
        segments=[
            "#0.00s - 10.00s#: The focal species is *Jynx torquilla* (Eurasian Wryneck)."
        ]
    )
    out = apply_extract_scientific_name_from_text(ctx, labels=labels)
    assert out.segments == ["Jynx torquilla"]


def test_extract_scientific_name_from_text_preserves_when_no_confident_match() -> None:
    labels = ["Turdus philomelos", "Jynx torquilla"]
    ctx = PostProcessContext(segments=["I think it's a Common Cuckoo (Cuculus canorus)."])
    out = apply_extract_scientific_name_from_text(ctx, labels=labels)
    # Not in labels, so we should not coerce into a wrong label.
    assert out.segments == [ctx.segments[0].strip()]
