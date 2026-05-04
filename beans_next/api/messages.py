"""Utilities for chat-style ``messages`` used in ``predictions_v1`` requests."""

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class _MessageWithRoleContent(Protocol):
    """Object carrying ``role`` and ``content`` as strings."""

    role: str
    content: str


def messages_to_dicts(messages: Iterable[object]) -> list[dict[str, str]]:
    """Convert message objects to `predictions_v1` wire dicts.

    Each output item is `{"role": <str>, "content": <str>}` as required by
    `PredictionsV1RequestItem` in `beans_next.api.http_schemas`.

    Accepts:

    * Mappings with string `role` and `content` keys (e.g. plain `dict`).
    * Objects exposing string attributes `role` and `content` (e.g. small
      dataclasses or Pydantic models).

    Parameters
    ----------
    messages
        Sequence of user/system/assistant messages to serialize.

    Returns
    -------
    list[dict[str, str]]
        List of `{"role", "content"}` dicts in input order.

    Raises
    ------
    TypeError
        If an element is neither a suitable mapping nor an object with
        `role` / `content` string attributes.
    KeyError
        If a mapping is missing `role` or `content`.

    """
    out: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            try:
                role = msg["role"]
                content = msg["content"]
            except KeyError as exc:
                raise KeyError(
                    "Each mapping message must include 'role' and 'content' keys."
                ) from exc
        elif isinstance(msg, _MessageWithRoleContent):
            role = msg.role
            content = msg.content
        else:
            msg_type = type(msg).__name__
            raise TypeError(
                "Each message must be a mapping with 'role' and 'content' keys, "
                "or an object with 'role' and 'content' str attributes; "
                f"got {msg_type!r}."
            )
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("'role' and 'content' must be strings.")
        out.append({"role": role, "content": content})
    return out


__all__ = ["messages_to_dicts"]
