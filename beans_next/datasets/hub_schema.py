"""Compact the Hub metadata while preserving an exact provenance delta."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

MAIN_COLUMNS = (
    "id",
    "sample_id",
    "tier",
    "task",
    "messages",
    "file_name",
    "context_audio_paths",
    "query_audio_path",
    "source_dataset",
    "source_id",
    "license",
    "metadata",
)
_CONSTRUCTION_KEYS = frozenset(
    {
        "audio_id",
        "synthesis_config",
        "source_split",
        "source_file",
        "source_uri",
        "beanspro_category",
        "beanspro_format",
        "beanspro_task",
        "difficulty_score",
        "sampling_cap",
        "sampling_seed",
        "quality_scores",
        "quality_thresholds",
        "source_conversation_key",
        "source_version",
        "source_dataset",
        "source_id",
    }
)
_ANNOTATION_COLUMNS = (
    "crop_start",
    "crop_end",
    "duration_sec",
    "sample_rate",
    "selection_table_tsv",
    "species_count",
    "species_list",
    "total_call_count",
    "skills",
)


def compact_hub_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Separate one legacy Hub row into evaluation metadata and provenance.

    Preserve both identifiers, existing messages, and all audio paths. Convert
    legacy instruction/output pairs without changing their text. Keep task
    annotations in JSON metadata and move construction details to provenance.

    Parameters
    ----------
    row
        A legacy row containing every column in `MAIN_COLUMNS`.

    Returns
    -------
    tuple[dict, dict]
        Compact row and a provenance row with both IDs and `original_fields`.
        Overlay the decoded `original_fields` on the compact row to reconstruct
        the exact original row.

    Raises
    ------
    ValueError
        If identifiers, messages, annotations, or required columns are invalid.
    """
    missing = set(MAIN_COLUMNS) - row.keys()
    if missing:
        raise ValueError(f"Missing legacy columns: {sorted(missing)}")
    if any(not isinstance(row[k], str) or not row[k] for k in ("id", "sample_id")):
        raise ValueError("Both id and sample_id must be nonempty strings")
    result = {k: row[k] for k in MAIN_COLUMNS}
    messages = row["messages"]
    if messages is None:
        instruction, output = row.get("instruction"), row.get("output")
        if not all(isinstance(v, str) and v.strip() for v in (instruction, output)):
            raise ValueError("Missing instruction/output pair")
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [m.get("role") for m in messages if isinstance(m, dict)]
        != ["user", "assistant"]
        or any(
            not isinstance(m.get("content"), str) or not m["content"].strip()
            for m in messages
        )
    ):
        raise ValueError("Expected one user prompt and one assistant target")
    result["messages"] = messages
    raw_metadata = row["metadata"]
    try:
        metadata = json.loads(raw_metadata) if raw_metadata else {}
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain a JSON object") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata must contain a JSON object")
    for key in ("source_dataset", "source_id"):
        if result[key] is None and isinstance(metadata.get(key), str):
            result[key] = metadata[key]
    annotations = {k: v for k, v in metadata.items() if k not in _CONSTRUCTION_KEYS}
    for key in _ANNOTATION_COLUMNS:
        value = row.get(key)
        if value is not None:
            if key in annotations and annotations[key] != value:
                raise ValueError(f"Conflicting annotation: {key}")
            annotations[key] = value
    result["metadata"] = json.dumps(annotations, ensure_ascii=False, sort_keys=True)
    original = {k: v for k, v in row.items() if k not in result or result[k] != v}
    provenance = {
        "id": row["id"],
        "sample_id": row["sample_id"],
        "original_fields": json.dumps(original, ensure_ascii=False, sort_keys=True),
    }
    return result, provenance
