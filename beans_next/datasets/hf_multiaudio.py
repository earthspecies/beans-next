"""HuggingFace `datasets` loader for multi-audio BeansPro-style rows.

This module mirrors :mod:`beans_next.datasets.hf_streaming`, but targets datasets
whose examples contain **multiple** audio slots aligned with multiple
``<Audio><AudioHere></Audio>`` placeholders in a single user message.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from beans_next.api.types import DatasetExample
from beans_next.datasets.base import (
    ensure_audio_paths_from_sequence,
    require_datasets,
    resolve_hf_sample_id,
)
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER


def _strip_audio_placeholders_except_last(conversation: str) -> str:
    idx = conversation.rfind(AUDIO_PLACEHOLDER)
    if idx == -1:
        return conversation
    prefix = conversation[:idx].replace(AUDIO_PLACEHOLDER, "[audio]")
    return prefix + conversation[idx:]


def _conversation_and_labels_from_messages(
    row: Mapping[str, Any],
) -> tuple[str, str | None]:
    messages_raw = row.get("messages")
    if not isinstance(messages_raw, list):
        return "", None
    conversation = ""
    labels: str | None = None
    for msg in messages_raw:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "user" and isinstance(content, str):
            conversation = content
        elif role == "assistant" and isinstance(content, str):
            stripped = content.strip()
            labels = stripped or None
    return conversation, labels


def _count_placeholders(conversation: str) -> int:
    if not conversation:
        return 0
    return conversation.count(AUDIO_PLACEHOLDER)


def beans_next_multiaudio_row_filter(
    *,
    tier: str | None,
    subset: str | None,
) -> Callable[[Mapping[str, Any]], bool]:
    """Build a row predicate for unified BEANS-Next tables (``tier`` / ``subset`` columns).

    Rows from the Hub may use ``subset`` (preferred) or legacy ``task`` for the
    multiaudio subset id (e.g. ``\"crow-4way\"``).

    Parameters
    ----------
    tier
        When non-empty, require ``row[\"tier\"] == tier``.
    subset
        When non-empty, require ``row[\"subset\"] == subset`` or
        ``row[\"task\"] == subset``.

    Returns
    -------
    Callable[[Mapping[str, Any]], bool]
        Predicate suitable for ``row_filter`` on streaming loaders.
    """
    t = tier.strip() if isinstance(tier, str) and tier.strip() else None
    s = subset.strip() if isinstance(subset, str) and subset.strip() else None

    def _pred(row: Mapping[str, Any]) -> bool:
        if t is not None:
            if row.get("tier") != t:
                return False
        if s is not None:
            return row.get("subset") == s or row.get("task") == s
        return True

    return _pred


def iter_hf_streaming_multiaudio_examples(
    path_or_id: str,
    *,
    split: str,
    config_name: str | None = None,
    revision: str | None = None,
    task_id: str | None = None,
    id_field: str = "id",
    row_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    load_dataset_kwargs: Mapping[str, Any] | None = None,
    audio_field: str = "audio",
) -> Iterator[DatasetExample]:
    """Yield ``DatasetExample`` rows from a streaming multi-audio Hub dataset.

    The loader emits the same metadata keys as the esp_data-backed multiaudio path
    in :func:`beans_next.datasets.esp_data.iter_esp_data_beans_next_multiaudio_examples`:

    - ``metadata["conversation"]``: full multi-audio user prompt
    - ``metadata["conversation_query_only"]``: single-audio reformulation
    - ``metadata["audio_paths"]``: list of local WAV paths (one per slot)
    - ``metadata["audio_path"]``: alias to the query audio (last element)

    Parameters
    ----------
    path_or_id
        Hub dataset id or path for ``datasets.load_dataset``.
    split
        Hub split name (typically ``"test"``).
    config_name
        Optional Hub builder config. Use ``None`` for a **single-config** dataset
        (e.g. ``EarthSpeciesProject/BEANS-Next``) where ``tier`` and ``subset`` are
        columns on each row. Pass a string only for legacy multi-config repos.
    revision
        Optional Hub revision.
    task_id
        Optional eval-task id stored on each yielded example.
    id_field
        Row key inspected for a stable string id.
    row_filter
        Optional predicate to skip rows client-side.
    load_dataset_kwargs
        Extra kwargs forwarded to ``datasets.load_dataset``. This iterator always
        forces ``streaming=True``.
    audio_field
        Column name storing the list of decoded HF audio entries (recommended:
        ``"audio"``).
    """
    kwargs = dict(load_dataset_kwargs or ())
    if kwargs.get("streaming") is False:
        msg = "non-streaming loads belong in beans_next.datasets.hf (map-style)"
        raise ValueError(msg)
    kwargs["streaming"] = True
    datasets = require_datasets()

    if config_name is None:
        loaded = datasets.load_dataset(path_or_id, split=split, revision=revision, **kwargs)
    else:
        loaded = datasets.load_dataset(
            path_or_id, config_name, split=split, revision=revision, **kwargs
        )

    yield_ordinal = 0
    for row in loaded:
        if row_filter is not None and not row_filter(row):
            continue
        mapping: Mapping[str, Any] = dict(row) if not isinstance(row, Mapping) else row

        sample_id = resolve_hf_sample_id(
            mapping,
            path_or_id=path_or_id,
            split=split,
            revision=revision,
            ordinal=yield_ordinal,
            id_field=id_field,
        )

        conversation, labels = _conversation_and_labels_from_messages(mapping)
        if not conversation:
            conv_val = mapping.get("conversation")
            if isinstance(conv_val, str):
                conversation = conv_val
        if labels is None:
            out_val = mapping.get("output")
            if isinstance(out_val, str):
                labels = out_val.strip() or None

        audio_paths_val = mapping.get("audio_paths")
        if isinstance(audio_paths_val, list) and all(
            isinstance(x, str) and x.strip() for x in audio_paths_val
        ):
            audio_paths = [str(x).strip() for x in audio_paths_val]
        else:
            audio_paths = ensure_audio_paths_from_sequence(
                mapping.get(audio_field), sample_id=sample_id
            )
            if audio_paths is None:
                msg = (
                    "Missing multi-audio data for HuggingFace row. "
                    f"sample_id={sample_id!r} field={audio_field!r}"
                )
                raise ValueError(msg)

        n_placeholders = _count_placeholders(conversation)
        if n_placeholders and n_placeholders != len(audio_paths):
            msg = (
                "Audio placeholder count does not match audio list length. "
                f"sample_id={sample_id!r} placeholders={n_placeholders} "
                f"n_audios={len(audio_paths)}"
            )
            raise ValueError(msg)

        meta: dict[str, Any] = {
            "audio_paths": audio_paths,
            "n_audios": len(audio_paths),
        }
        if audio_paths:
            meta["audio_path"] = audio_paths[-1]
        if conversation:
            meta["conversation"] = conversation
            meta["conversation_query_only"] = _strip_audio_placeholders_except_last(
                conversation
            )

        for key in (
            "tier",
            "subset",
            "task",
            "dataset_name",
            "source_dataset",
            "license",
            "template_path",
        ):
            v = mapping.get(key)
            if isinstance(v, str | int | float | bool):
                meta[key] = v

        yield DatasetExample(
            sample_id=sample_id,
            task_id=task_id,
            split=split,
            labels=labels,
            metadata=meta,
        )
        yield_ordinal += 1
