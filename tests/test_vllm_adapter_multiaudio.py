"""Tests for placeholder-aware audio injection in the vLLM adapter.

Tier 4 (in-context multi-audio) encodes the position of every clip with an
`<Audio><AudioHere></Audio>` placeholder, so support audios must be interleaved
with the text that labels them. Prepending them as a block loses that binding.

The adapter imports `httpx`/`fastapi`, which the library test environment does
not install, so the placeholder helpers are exec'd directly from source.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_ADAPTER = Path(__file__).resolve().parents[1] / "examples/servers/vllm/adapter.py"

_CONVERSATION = (
    "Here are four call types.\n\n"
    "A: <Audio><AudioHere></Audio>\n"
    "B: <Audio><AudioHere></Audio>\n"
    "C: <Audio><AudioHere></Audio>\n"
    "D: <Audio><AudioHere></Audio>\n\n"
    "Which call type best matches the following recording?\n"
    "<Audio><AudioHere></Audio>"
)


def _inject_audio_items() -> Callable[..., list[dict[str, Any]]]:
    """Exec the adapter's placeholder helpers without its heavy imports.

    Returns
    -------
    collections.abc.Callable
        The adapter's `_inject_audio_items` function.
    """
    src = _ADAPTER.read_text()
    start = src.index("_AUDIO_TAG_PATTERN = re.compile")
    end = src.index("def _call_upstream_chat_completion(")
    ns: dict[str, Any] = {"re": re, "Any": Any}
    exec(src[start:end], ns)  # noqa: S102
    return ns["_inject_audio_items"]


def _audio(n: int) -> list[dict[str, Any]]:
    return [
        {"type": "audio_url", "audio_url": {"url": f"data:{i}"}} for i in range(n)
    ]


def test_audios_are_interleaved_at_placeholder_positions() -> None:
    parts = _inject_audio_items()(
        [{"role": "user", "content": _CONVERSATION}], _audio(5)
    )[0]["content"]
    order = [p["audio_url"]["url"] for p in parts if p["type"] == "audio_url"]
    assert order == [f"data:{i}" for i in range(5)]
    # Each choice label must immediately precede its own clip.
    assert parts[0]["text"].strip().endswith("A:")
    assert parts[2]["text"].strip() == "B:"


def test_no_placeholders_still_prepends_to_last_user_message() -> None:
    """Single-audio tiers 1-3 must be unaffected by the tier 4 change."""
    content = _inject_audio_items()(
        [{"role": "user", "content": "Classify this."}], _audio(1)
    )[0]["content"]
    assert content[0]["type"] == "audio_url"
    assert content[1]["text"] == "Classify this."


def test_too_few_audios_for_placeholders_raises() -> None:
    with pytest.raises(ValueError, match="not enough audio_inputs"):
        _inject_audio_items()([{"role": "user", "content": _CONVERSATION}], _audio(3))


def test_unused_audios_after_substitution_raises() -> None:
    with pytest.raises(ValueError, match="unused audio_inputs"):
        _inject_audio_items()([{"role": "user", "content": _CONVERSATION}], _audio(7))
