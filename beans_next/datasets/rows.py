"""Convert Hub metadata rows into evaluation examples without storage access."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from beans_next.api.types import DatasetExample

_SAMPLE_ID_PREFIX = "beans_next:hf:"
_AUDIO_PLACEHOLDER_TAG = "<Audio><AudioHere></Audio>"


def synthesize_row_sample_id(
    *,
    dataset: str,
    subset: str,
    split: str,
    ordinal: int,
) -> str:
    """Build a deterministic `sample_id` for metadata rows missing a stable id.

    Parameters
    ----------
    dataset
        Dataset family identifier (for this module: typically `"beans_zero"`).
    subset
        Subset name (e.g. `"esc50"`).
    split
        Split name (e.g. `"test"`).
    ordinal
        Zero-based ordinal in the yielded stream.

    Returns
    -------
    str
        Stable synthetic sample id.
    """
    parts = (dataset, subset, split, str(int(ordinal)))
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{_SAMPLE_ID_PREFIX}{digest}"


def _resolve_row_id(row: Mapping[str, object]) -> str | None:
    raw = row.get("id") or row.get("sample_id") or row.get("uuid")
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            return stripped
    return None


def _labels_from_row(
    row: Mapping[str, object],
) -> str | list[str] | dict[str, object] | None:
    val = row.get("output") if "output" in row else row.get("labels")
    if val is None:
        return None
    if isinstance(val, str | list | dict):
        return val
    return str(val)


def _build_dataset_example(
    row: Mapping[str, object],
    *,
    sample_id: str,
    audio_path: str | None,
    split: str,
    task_id: str | None,
) -> DatasetExample:
    """Assemble a `DatasetExample` from a metadata row and a resolved audio path.

    Accepts dedicated legacy ``instruction`` and ``output`` columns, or the
    ``messages`` column used by every tier in the compact Hub schema.

    Parameters
    ----------
    row
        Raw metadata row from the dataset loader.
    sample_id
        Stable sample identifier.
    audio_path
        Absolute local WAV path, or ``None`` when audio is unavailable.
    split
        Dataset split stored on the example.
    task_id
        Optional eval-task id stored on the example.

    Returns
    -------
    DatasetExample
        Fully assembled example ready for the runner.
    """
    meta: dict[str, object] = {}
    if isinstance(audio_path, str) and audio_path.strip():
        meta["audio_path"] = audio_path
    for key in (
        "file_name",
        "source_dataset",
        "dataset_name",
        "task",
        "license",
        "created_at",
    ):
        val = row.get(key)
        if isinstance(val, str | int | float | bool):
            meta[key] = val

    instruction_raw = row.get("instruction") or row.get("instruction_text")
    instruction: str | None = (
        instruction_raw.strip()
        if isinstance(instruction_raw, str) and instruction_raw.strip()
        else None
    )
    labels = _labels_from_row(row)

    # Compact Hub rows store prompts and answers in ``messages`` for all tiers.
    if instruction is None or labels is None:
        messages_raw = row.get("messages")
        if isinstance(messages_raw, list):
            for msg in messages_raw:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if (
                    role == "user"
                    and instruction is None
                    and isinstance(content, str)
                    and content.strip()
                ):
                    instruction = content.strip()
                elif (
                    role == "assistant"
                    and labels is None
                    and isinstance(content, str)
                    and content.strip()
                ):
                    labels = content.strip()

    if instruction is not None:
        meta["instruction"] = instruction

    return DatasetExample(
        sample_id=sample_id,
        task_id=task_id,
        split=split,
        labels=labels,
        metadata=meta,
    )


def _strip_audio_placeholders_except_last(conversation: str) -> str:
    """Replace all but the last ``<Audio><AudioHere></Audio>`` with ``[audio]``.

    Used to build a single-audio reformulation of multi-audio prompts for
    launchers that support only one audio input (e.g. NatureLM v1.1).

    Parameters
    ----------
    conversation
        User message text containing one or more ``<Audio><AudioHere></Audio>``
        placeholders.

    Returns
    -------
    str
        Modified conversation with all but the last placeholder replaced by
        ``[audio]``.
    """
    tag = _AUDIO_PLACEHOLDER_TAG
    idx = conversation.rfind(tag)
    if idx == -1:
        return conversation
    prefix = conversation[:idx].replace(tag, "[audio]")
    return prefix + conversation[idx:]


def _build_multiaudio_dataset_example(
    row: Mapping[str, object],
    *,
    sample_id: str,
    audio_paths: list[str],
    query_audio_path: str | None,
    split: str,
    task_id: str | None,
) -> DatasetExample:
    """Assemble a ``DatasetExample`` from a BEANSNextMultiAudio row.

    Parameters
    ----------
    row
        Raw metadata row from the dataset loader.
    sample_id
        Stable sample identifier.
    audio_paths
        Resolved local WAV paths for all ``audio_paths`` entries.
    query_audio_path
        Legacy fallback when the complete ordered audio list is unavailable.
        Otherwise the final list entry is the query, including for single-audio
        prompt specs, so its waveform matches full multi-audio evaluation.
    split
        Dataset split stored on the example.
    task_id
        Optional eval-task id stored on the example.

    Returns
    -------
    DatasetExample
        Fully assembled example ready for the runner.
    """
    meta: dict[str, object] = {}

    if audio_paths:
        meta["audio_paths"] = audio_paths
        meta["n_audios"] = len(audio_paths)

    effective_query = audio_paths[-1] if audio_paths else query_audio_path
    if effective_query:
        meta["audio_path"] = effective_query

    conversation = ""
    labels: str | None = None
    messages_raw = row.get("messages")
    if isinstance(messages_raw, list):
        for msg in messages_raw:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and isinstance(content, str):
                conversation = content
            elif role == "assistant" and isinstance(content, str):
                labels = content.strip() or None

    if conversation:
        meta["conversation"] = conversation
        meta["conversation_query_only"] = _strip_audio_placeholders_except_last(
            conversation
        )

    for key in (
        "task",
        "dataset_name",
        "source_dataset",
        "license",
        "template_path",
    ):
        val = row.get(key)
        if isinstance(val, str | int | float | bool):
            meta[key] = val

    return DatasetExample(
        sample_id=sample_id,
        task_id=task_id,
        split=split,
        labels=labels,
        metadata=meta,
    )
