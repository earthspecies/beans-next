"""Prevent ID reuse when source keys point to different evaluation content."""

from copy import deepcopy

import pytest

from beans_next.datasets.hub_identifiers import content_example_id


def example() -> dict:
    return {
        "source_key": "t2-behavior:000000",
        "task": "t2-behavior",
        "audio_paths": ["audio/abc.wav"],
        "messages": [
            {"role": "user", "content": "Which call?"},
            {"role": "assistant", "content": "a"},
        ],
    }


def test_id_is_deterministic_and_ignores_dictionary_order() -> None:
    row = example()
    other = deepcopy(row)
    other["messages"] = [dict(reversed(list(m.items()))) for m in row["messages"]]
    assert content_example_id(**row) == content_example_id(**other)


@pytest.mark.parametrize("change", ["audio", "prompt", "target", "source_key", "task"])
def test_reused_source_key_cannot_hide_changed_content(change: str) -> None:
    before = example()
    after = deepcopy(before)
    if change == "audio":
        after["audio_paths"] = ["audio/def.wav"]
    elif change in ("prompt", "target"):
        after["messages"][0 if change == "prompt" else 1]["content"] += " changed"
    else:
        after[change] += "-different"
    assert content_example_id(**before) != content_example_id(**after)


def test_audio_order_and_repeats_are_significant() -> None:
    row = example()
    ids = {
        content_example_id(**{**row, "audio_paths": paths})
        for paths in (["a", "b"], ["b", "a"], ["a", "a", "b"])
    }
    assert len(ids) == 3
