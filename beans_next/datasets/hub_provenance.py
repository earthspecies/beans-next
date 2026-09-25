"""Recover ordered source references without changing evaluation audio."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

PROVENANCE_COLUMNS = (
    "source_datasets",
    "source_audio_ids",
    "source_file_paths",
    "source_urls",
    "audio_start_seconds",
    "audio_end_seconds",
    "source_id_types",
    "provenance_status",
)
_SOURCES = {
    "xeno-canto": "xeno-canto",
    "inaturalist": "inaturalist",
    "BirdeepCropped": "birdeep",
    "PowdermillCropped": "powdermill",
    "BirdVoxFullNightCropped": "birdvox-full-night",
    "NocturnalBirdMigration": "nocturnal-bird-migration",
    "doi:10.1080/09524622.2025.2500380": "f0-bioacoustic",
    "10.5061/dryad.v9s4mw73w": "plains-zebra",
    "10.64898/2026.04.02.715916": "carrion-crow",
    "Giant Otters (Mumm & Knörnschild 2014)": "giant-otters",
    "DCASE-2021-Task-5": "dcase-2021-task-5",
    "Hainan Gibbons": "hainan-gibbons",
}


def ordered_source_provenance(
    original: Mapping[str, Any],
    *,
    manifest_paths: Mapping[tuple[str, str], str],
    otter_ids: Mapping[str, str],
) -> dict[str, list[Any]]:
    """Recover source identity for each audio slot in its original order.

    Parameters
    ----------
    original
        Restored metadata row before compaction, with source paths preserved.
    manifest_paths
        Unique original filenames mapped to archive-relative paths, keyed by
        normalized dataset name and filename (BIRDeep and Powdermill).
    otter_ids
        Original otter filenames mapped to call IDs from the annotation table.

    Returns
    -------
    dict[str, list]
        Parallel provenance lists. Unknown IDs, URLs, and times remain null.
        Known derivative filenames are explicitly distinguished from raw IDs.

    Raises
    ------
    ValueError
        If paths, source names, IDs, manifest matches, or crop fields conflict.
    """
    metadata = json.loads(original["metadata"] or "{}")
    raw_source = original.get("source_dataset") or metadata.get("source_dataset")
    paths = (
        original.get("audio_paths")
        if original["tier"] in (3, 4)
        else [original.get("audio_path_original_sample_rate")]
    )
    if not paths or any(not isinstance(p, str) or not p for p in paths):
        raise ValueError("Missing source paths")
    out: dict[str, list[Any]] = {key: [] for key in PROVENANCE_COLUMNS}
    for path in paths:
        source = _SOURCES.get(raw_source)
        if raw_source == "xeno-canto+inaturalist new unseen holdouts":
            if path.startswith("xeno-canto/"):
                source = "xeno-canto"
            elif path.startswith("inaturalist/"):
                source = "inaturalist"
        if source is None:
            raise ValueError(f"Unknown source: {raw_source!r}")
        filename = PurePosixPath(path).name
        source_path = filename
        aid: str | None = filename
        url = None
        start = end = None
        kind = "source_filename"
        status = "filename_only"
        if source == "xeno-canto":
            match = re.match(r"XC(\d+)", filename)
            number = (
                match[1] if match else metadata.get("xc_id") or metadata.get("audio_id")
            )
            if not str(number).removeprefix("XC").isdigit():
                raise ValueError("Missing Xeno-canto recording ID")
            number = str(number).removeprefix("XC")
            if original["tier"] != 4:
                for key in ("xc_id", "audio_id"):
                    if (
                        metadata.get(key)
                        and str(metadata[key]).removeprefix("XC") != number
                    ):
                        raise ValueError("Conflicting Xeno-canto recording IDs")
            aid, url = "XC" + number, "https://xeno-canto.org/" + number
            kind, status = "recording_id", "metadata_id"
        elif source == "inaturalist":
            aid = None
            kind, status = "unresolved", "unresolved_recording_id"
        elif source in ("birdeep", "powdermill", "birdvox-full-night"):
            match = re.fullmatch(r"(.+)__crop_(\d+)_(\d+)\.wav", filename)
            if not match:
                raise ValueError("Missing crop boundaries")
            aid = match[1]
            start, end = int(match[2]) / 1000, int(match[3]) / 1000
            if end <= start:
                raise ValueError("Invalid crop interval")
            for key, parsed in (("crop_start", start), ("crop_end", end)):
                if original.get(key) is not None and abs(original[key] - parsed) > 1e-6:
                    raise ValueError("Conflicting crop boundaries")
            if source == "birdvox-full-night":
                unit = re.fullmatch(r"BirdVox-full-night_unit(\d+)", aid)
                if not unit:
                    raise ValueError("Unknown BirdVox unit")
                source_path = f"BirdVox-full-night_flac-audio_unit{unit[1]}.flac"
                aid = source_path
                status = "crop_filename"
            else:
                source_path = manifest_paths.get((source, aid))
                if not source_path:
                    raise ValueError(f"Missing manifest match: {source}/{aid}")
                status = "manifest_filename"
        elif source == "nocturnal-bird-migration":
            match = re.fullmatch(r"[^/]+#(\d+)\.wav", filename)
            if not match:
                raise ValueError("Unknown Nocturnal Bird Migration filename")
            aid = "XC" + match[1]
            source_path = "train_nbm_xc/" + filename
            url = "https://xeno-canto.org/" + match[1]
            kind, status = "upstream_recording_id", "upstream_id_from_filename"
        elif source == "giant-otters":
            aid = otter_ids.get(filename)
            if not aid:
                raise ValueError(f"Unknown original otter clip: {filename}")
            kind, status = "call_id", "manifest_id"
        elif source == "f0-bioacoustic":
            source_path = aid = path
        elif source in ("dcase-2021-task-5", "hainan-gibbons", "carrion-crow"):
            kind, status = "derived_clip_filename", "derived_clip_only"
        values = (source, aid, source_path, url, start, end, kind, status)
        for key, value in zip(PROVENANCE_COLUMNS, values, strict=True):
            out[key].append(value)
    return out


def restore_provenance_row(
    row: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore a legacy row, excluding fields added by a later schema.

    Parameters
    ----------
    row
        Current evaluation row.
    provenance
        Matching provenance row, optionally including `added_columns` JSON.

    Returns
    -------
    dict
        Exact row from before the compact-schema migration.

    Raises
    ------
    ValueError
        If the example identifiers do not match.
    """
    if any(row[k] != provenance[k] for k in ("id", "sample_id")):
        raise ValueError("Provenance identifiers do not match")
    added = set(json.loads(provenance.get("added_columns") or "[]"))
    return {
        **{k: v for k, v in row.items() if k not in added},
        **json.loads(provenance["original_fields"]),
    }
