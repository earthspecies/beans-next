"""Checkpoint helpers for benchmark resume.

This module is intentionally dependency-light and only deals with reading and
validating the on-disk ``checkpoint.json`` payload written by the runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "completed_sample_ids_from_checkpoint_json",
    "read_checkpoint_json",
]

_CHECKPOINT_SCHEMA_VERSION_V1 = "beans_next.checkpoint.v1"


def read_checkpoint_json(path: Path) -> Mapping[str, Any]:
    """Read and validate a ``checkpoint.json`` payload as a mapping.

    Parameters
    ----------
    path
        Path to a ``checkpoint.json`` file.

    Returns
    -------
    collections.abc.Mapping[str, typing.Any]
        Parsed checkpoint payload.

    Raises
    ------
    ValueError
        If the file cannot be parsed as JSON or the root is not a mapping.
    """
    raw_text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(raw_text)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid JSON in checkpoint: {path}") from exc
    if not isinstance(obj, Mapping):
        raise ValueError(
            f"Checkpoint root must be a mapping, got {type(obj).__name__} ({path})"
        )
    return obj


def completed_sample_ids_from_checkpoint_json(path: Path) -> set[str]:
    """Return completed sample ids from a checkpoint payload.

    Parameters
    ----------
    path
        Path to ``checkpoint.json``.

    Returns
    -------
    set[str]
        Set of completed ``sample_id`` strings.

    Raises
    ------
    ValueError
        If the payload is missing required fields or they are invalid.
    """
    payload = read_checkpoint_json(path)
    schema = payload.get("schema_version")
    if schema != _CHECKPOINT_SCHEMA_VERSION_V1:
        raise ValueError(
            f"Unsupported checkpoint schema_version={schema!r} (expected "
            f"{_CHECKPOINT_SCHEMA_VERSION_V1!r})"
        )
    ids = payload.get("completed_sample_ids")
    if ids is None:
        ids = payload.get("completed_ids")
    if not isinstance(ids, list):
        raise ValueError("Checkpoint must contain `completed_sample_ids` list.")
    out: set[str] = set()
    for item in ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Checkpoint `completed_sample_ids` must contain strings.")
        out.add(item)
    return out
