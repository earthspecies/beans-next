"""Check original identities, crop precision, and ordered multi-audio provenance."""

import json

import pytest

from beans_next.datasets.hub_provenance import (
    ordered_source_provenance,
    restore_provenance_row,
)


def test_mixed_source_slots_keep_repeats_and_unknown_sound_ids() -> None:
    row = {
        "tier": 4,
        "source_dataset": "xeno-canto+inaturalist new unseen holdouts",
        "metadata": "{}",
        "audio_paths": [
            "xeno-canto/raw/XC123-test.wav",
            "inaturalist/raw/inat_456.wav",
            "xeno-canto/raw/XC123-test.wav",
        ],
    }
    result = ordered_source_provenance(row, manifest_paths={}, otter_ids={})
    assert result["source_audio_ids"] == ["XC123", None, "XC123"]
    assert result["source_datasets"] == ["xeno-canto", "inaturalist", "xeno-canto"]
    assert result["source_urls"][1] is None
    assert result["source_id_types"][1] is None
    assert all(len(values) == 3 for values in result.values())


def test_internal_row_id_is_not_used_as_recording_id() -> None:
    row = {
        "tier": 1,
        "source_dataset": "xeno-canto",
        "source_id": "249",
        "metadata": json.dumps({"audio_id": "191191"}),
        "audio_path_original_sample_rate": "audio/recording.flac",
    }
    result = ordered_source_provenance(row, manifest_paths={}, otter_ids={})
    assert result["source_audio_ids"] == ["XC191191"]


def test_crop_uses_original_manifest_path_and_milliseconds() -> None:
    row = {
        "tier": 3,
        "source_dataset": None,
        "metadata": json.dumps({"source_dataset": "BirdeepCropped"}),
        "audio_paths": ["audio/AM1.WAV__crop_0053090_0056088.wav"],
        "crop_start": 53.09,
        "crop_end": 56.088,
    }
    result = ordered_source_provenance(
        row,
        manifest_paths={("birdeep", "AM1.WAV"): "Audios/AM1/AM1.WAV"},
        otter_ids={},
    )
    assert result["source_audio_ids"] == ["AM1.WAV"]
    assert result["source_file_paths"] == ["Audios/AM1/AM1.WAV"]
    assert result["audio_start_seconds"] == [53.09]
    assert result["audio_end_seconds"] == [56.088]
    row["crop_start"] = 1.0
    with pytest.raises(ValueError, match="Conflicting crop"):
        ordered_source_provenance(
            row,
            manifest_paths={("birdeep", "AM1.WAV"): "Audios/AM1/AM1.WAV"},
            otter_ids={},
        )


@pytest.mark.parametrize("source", ["DCASE-2021-Task-5", "Hainan Gibbons"])
def test_derived_clip_suffix_is_not_invented_as_crop_time(source: str) -> None:
    row = {
        "tier": 4,
        "source_dataset": source,
        "metadata": "{}",
        "audio_paths": ["audio/recording.105_005.wav"],
    }
    result = ordered_source_provenance(row, manifest_paths={}, otter_ids={})
    assert result["source_audio_ids"] == ["recording.105_005.wav"]
    assert result["source_id_types"] == ["derived_clip_filename"]
    assert result["audio_start_seconds"] == [None]
    assert result["audio_end_seconds"] == [None]


def test_new_fields_do_not_break_exact_original_reconstruction() -> None:
    row = {
        "id": "a",
        "sample_id": "b",
        "source_dataset": "normalized",
        "source_audio_ids": ["XC123"],
    }
    provenance = {
        "id": "a",
        "sample_id": "b",
        "source_id": "row-17",
        "added_columns": '["source_audio_ids"]',
        "original_fields": '{"source_dataset": "original", "source_id": "row-17"}',
    }
    assert restore_provenance_row(row, provenance) == {
        "id": "a",
        "sample_id": "b",
        "source_dataset": "original",
        "source_id": "row-17",
    }
    provenance["sample_id"] = "wrong"
    with pytest.raises(ValueError, match="identifiers"):
        restore_provenance_row(row, provenance)
