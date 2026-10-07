"""Focused tests for the standalone Gaussian-noise manifest validator."""

# ruff: noqa: ANN401

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any


def _load_validator() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "validate_gaussian_noise_manifest.py"
    spec = importlib.util.spec_from_file_location("gaussian_noise_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import validator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _write_wav(
    path: Path, frames: list[tuple[int, ...]], sample_rate: int = 16_000
) -> None:
    channels = len(frames[0])
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(
            b"".join(struct.pack("<h", value) for frame in frames for value in frame)
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audio_metadata(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as stream:
        return {
            "path": path.name,
            "sha256": _sha256(path),
            "frames": stream.getnframes(),
            "sample_rate": stream.getframerate(),
            "channels": stream.getnchannels(),
        }


def _manifest(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "source.wav"
    _write_wav(source, [(0, 1000), (1000, 0), (-1000, 0), (0, -1000)])
    source_info = _audio_metadata(source)
    records: list[dict[str, Any]] = []
    for task_id, sample_id, slot_index in (
        ("caption", "s1", 0),
        ("caption", "s2", 0),
        ("qa", "s3", 0),
        ("qa", "s3", 1),
    ):
        noise = tmp_path / f"noise-{sample_id}-{slot_index}.wav"
        # +/-0.1 FS has exactly -20 dBFS RMS; opposite-polarity frames make
        # the finite fixture exactly zero mean in every channel.
        _write_wav(noise, [(3277, -3277), (-3277, 3277), (3277, -3277), (-3277, 3277)])
        noise_info = _audio_metadata(noise)
        protocol_version = "beans-next.gaussian-noise.v1"
        seed_digest = validator.derive_seed_sha256(
            "beans-rev-1",
            source_info["sha256"],
            slot_index,
            protocol_version,
            17,
        )
        seed = int.from_bytes(bytes.fromhex(seed_digest[:16]), "big", signed=False)
        noise_stats = validator._audio_properties(noise)
        metadata_path = tmp_path / f"noise-{sample_id}-{slot_index}.wav.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "beans_next.gaussian_noise.v1",
                    "source_identity": source_info["sha256"],
                    "slot_index": slot_index,
                    "seed": seed,
                    "seed_sha256": seed_digest,
                    "sha256": noise_info["sha256"],
                    "frames": noise_info["frames"],
                    "sample_rate": noise_info["sample_rate"],
                    "channels": noise_info["channels"],
                }
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "slot_index": slot_index,
                "source_identity": source_info["sha256"],
                "source_path": str(source),
                "source_sha256": source_info["sha256"],
                "noise_path": str(noise),
                "metadata_path": str(metadata_path),
                "seed": seed,
                "seed_sha256": seed_digest,
                "audio_sha256": noise_info["sha256"],
                "frames": noise_info["frames"],
                "sample_rate": noise_info["sample_rate"],
                "channels": noise_info["channels"],
                "duration_seconds": noise_stats["duration_seconds"],
                "mean": noise_stats["mean"],
                "rms": noise_stats["rms"],
                "peak": noise_stats["peak"],
                "rms_dbfs": noise_stats["rms_dbfs"],
            }
        )
    artifact_rows = [
        {"task_id": "caption", "sample_id": "s1", "predictions": ["a"]},
        {"task_id": "caption", "sample_id": "s2", "predictions": ["b"]},
        {"task_id": "qa", "sample_id": "s3", "predictions": ["c"]},
    ]
    for name in ("predictions.jsonl", "processed_predictions.jsonl"):
        (tmp_path / name).write_text(
            "".join(json.dumps(row) + "\n" for row in artifact_rows), encoding="utf-8"
        )
    return {
        "schema_version": "beans_next.gaussian_noise_manifest.v1",
        "protocol": "zero-mean-white-Gaussian-noise",
        "protocol_version": "beans-next.gaussian-noise.v1",
        "dataset_revision": "beans-rev-1",
        "global_seed": 17,
        "rms_dbfs": -20.0,
        "code_commit": "abc1234",
        "expected_counts": {"tasks": 2, "samples": 3, "slots": 4},
        "records": records,
        "artifacts": {
            "predictions": "predictions.jsonl",
            "processed_predictions": "processed_predictions.jsonl",
        },
    }


def test_seed_derivation_is_stable_and_slot_sensitive() -> None:
    first = validator.derive_noise_seed("rev", "source", 0, "v1", 9)
    second = validator.derive_noise_seed("rev", "source", 0, "v1", 9)
    assert first == second
    assert first != validator.derive_noise_seed("rev", "source", 1, "v1", 9)
    assert first != validator.derive_noise_seed("rev-2", "source", 0, "v1", 9)


def test_valid_manifest_checks_audio_properties_checksums_and_artifact_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")

    report = validator.validate_manifest(path)

    assert report.valid, report.errors
    assert report.stats["tasks"] == 2
    assert report.stats["samples"] == 3
    assert report.stats["slots"] == 4
    assert report.stats["artifact_counts"] == {
        "predictions": 3,
        "processed_predictions": 3,
    }


def test_validator_rejects_checksum_audio_shape_and_seed_tampering(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["records"][0]["source_sha256"] = "0" * 64
    manifest["records"][1]["frames"] = 99
    manifest["records"][2]["seed"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_manifest(path)

    assert not report.valid
    joined = "\n".join(report.errors)
    assert "checksum mismatch" in joined
    assert "frames mismatch" in joined
    assert "does not match deterministic derivation" in joined


def test_validator_rejects_duplicate_out_of_order_records_counts_and_alignment(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    # Repeat s1 after another sample and put an artifact row out of order.
    manifest["records"][1]["sample_id"] = "s1"
    manifest["records"][1]["slot_index"] = 0
    manifest["records"][1]["placeholder_index"] = 0
    manifest["expected_counts"]["samples"] = 99
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"task_id": "caption", "sample_id": "s2"},
        {"task_id": "caption", "sample_id": "s1"},
        {"task_id": "qa", "sample_id": "s3"},
    ]
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    path = tmp_path / "bad-order.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_manifest(path)

    assert not report.valid
    joined = "\n".join(report.errors)
    assert "duplicate task/sample/slot" in joined
    assert "expected samples count 99" in joined
    assert "not aligned to manifest sample order" in joined


def test_jsonl_header_and_nested_sample_slots_are_supported(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    records = manifest.pop("records")
    header = dict(manifest)
    lines = [json.dumps(header), *(json.dumps(record) for record in records)]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = validator.validate_manifest(path)

    assert report.valid, report.errors
    assert report.stats["slots"] == 4


def test_task_id_is_optional_and_artifact_alignment_can_use_sample_ids(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["expected_counts"]["tasks"] = 1
    for record in manifest["records"]:
        record.pop("task_id")
    rows = [
        {"sample_id": "s1"},
        {"sample_id": "s2"},
        {"sample_id": "s3"},
    ]
    for name in ("predictions.jsonl", "processed_predictions.jsonl"):
        (tmp_path / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    path = tmp_path / "optional-task.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_manifest(path)

    assert report.valid, report.errors
    assert report.stats["tasks"] == 1


def test_outcome_summary_and_audio_noise_delta_analysis() -> None:
    summary = validator.summarize_outcomes(
        [
            {"sample_id": "a", "finish_reason": "stop", "parsed": True},
            {"sample_id": "b", "finish_reason": "refusal", "refusal": True},
            {"sample_id": "c", "error": "parse failed", "finish_reason": "error"},
        ]
    )
    assert summary["total"] == 3
    assert summary["parse_failures"] == 1
    assert summary["refusals"] == 1
    assert summary["errors"] == 1
    assert summary["finish_reasons"] == {"error": 1, "refusal": 1, "stop": 1}

    deltas = validator.compute_audio_noise_deltas(
        [
            {"task_id": "t", "sample_id": "a", "score": 0.8},
            {"task_id": "t", "sample_id": "b", "scores": {"accuracy": 0.5}},
        ],
        [
            {"task_id": "t", "sample_id": "a", "score": 0.3},
            {"task_id": "t", "sample_id": "b", "scores": {"accuracy": 0.2}},
        ],
    )
    assert deltas["unmatched_real"] == []
    assert deltas["unmatched_noise"] == []
    assert deltas["matched"][0]["metrics"]["score"]["audio_minus_noise"] == 0.5
    assert deltas["matched"][1]["metrics"]["accuracy"]["audio_minus_noise"] == 0.3


def test_cli_has_nonzero_failure_and_json_report(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["rms_dbfs"] = -10.0
    path = tmp_path / "bad-cli.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    script = (
        Path(__file__).parents[1] / "scripts" / "validate_gaussian_noise_manifest.py"
    )

    failure = subprocess.run(
        ["python", str(script), str(path)], capture_output=True, text=True, check=False
    )
    report = subprocess.run(
        ["python", str(script), str(path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert failure.returncode != 0
    assert "protocol rms_dbfs" in failure.stderr
    assert report.returncode != 0
    assert json.loads(report.stdout)["valid"] is False
