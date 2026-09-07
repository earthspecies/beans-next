"""Tests for AF-Next multi-turn conversation construction.

AF-Next's canonical multi-audio form is one audio per conversation turn: a
single turn carrying N clips renders only one `<sound>` placeholder and the
rest are dropped. These tests pin the turn-splitting behaviour without needing
the model or its processor.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_SERVE = Path(__file__).resolve().parents[1] / "examples/servers/af3/serve.py"
_PH = "<Audio><AudioHere></Audio>"


@dataclass
class _Msg:
    role: str
    content: str


def _build() -> Callable[..., list[dict[str, Any]]]:
    """Exec `_build_conversation` from the af3 launcher source.

    The launcher imports fastapi, which the library test environment does not
    install, so the function is exec'd in isolation with only the names it
    actually needs at runtime.

    Returns
    -------
    collections.abc.Callable
        The launcher's `_build_conversation` function.
    """
    src = _SERVE.read_text()
    start = src.index("def _build_conversation(")
    end = src.index("\ndef ", start)
    ns: dict[str, Any] = {
        "re": re,
        "Any": Any,
        "_AUDIO_PLACEHOLDER": re.compile(r"<Audio><AudioHere></Audio>"),
        "annotations": __import__("__future__").annotations,
    }
    exec(compile("from __future__ import annotations\n" + src[start:end],
                 "serve_extract", "exec"), ns)  # noqa: S102
    return ns["_build_conversation"]


def test_multi_audio_message_becomes_one_turn_per_clip() -> None:
    build = _build()
    text = f"Here are four.\n\nA: {_PH}\nB: {_PH}\nC: {_PH}\nD: {_PH}\n\nWhich?\n{_PH}"
    conv = build([_Msg("user", text)], [f"/tmp/{i}.wav" for i in range(5)])
    assert len(conv) == 5
    for turn in conv:
        kinds = [c["type"] for c in turn["content"]]
        assert kinds.count("audio") == 1, kinds
    # the label preceding each clip stays with it, in order
    assert conv[0]["content"][0]["text"].endswith("A: ")
    assert conv[1]["content"][0]["text"].strip() == "B:"
    order = [c["path"] for t in conv for c in t["content"] if c["type"] == "audio"]
    assert order == [f"/tmp/{i}.wav" for i in range(5)]


def test_single_audio_message_keeps_one_turn() -> None:
    """Tiers 1-3 send one clip; that shape must be untouched."""
    build = _build()
    conv = build([_Msg("user", f"Classify this.\n{_PH}")], ["/tmp/a.wav"])
    assert len(conv) == 1
    assert [c["type"] for c in conv[0]["content"]] == ["text", "audio"]


def test_message_with_no_placeholder_is_preserved() -> None:
    build = _build()
    conv = build([_Msg("user", "No audio here.")], [])
    assert len(conv) == 1
    assert conv[0]["content"] == [{"type": "text", "text": "No audio here."}]


def test_more_placeholders_than_audio_raises() -> None:
    """Silently dropping clips would leave the prompt referring to unheard audio."""
    build = _build()
    text = f"A: {_PH}\nB: {_PH}\nC: {_PH}"
    with pytest.raises(ValueError, match="refusing to drop clips"):
        build([_Msg("user", text)], ["/tmp/a.wav"])


def test_surplus_audio_gets_its_own_turns() -> None:
    build = _build()
    conv = build([_Msg("user", f"One.\n{_PH}")], ["/tmp/a.wav", "/tmp/b.wav"])
    audio = [c["path"] for t in conv for c in t["content"] if c["type"] == "audio"]
    assert audio == ["/tmp/a.wav", "/tmp/b.wav"]
    assert all(
        [c["type"] for c in t["content"]].count("audio") <= 1 for t in conv
    )
