#!/usr/bin/env python3
# ruff: noqa: ANN401, DOC201, DOC501, E501
"""Validate a deterministic BEANS-Next Gaussian-noise manifest.

The validator deliberately has no dependency on the BEANS-Next package.  This
makes it useful on a login node, on a copied result bundle, and in CI.  The
canonical document is a JSON object with ``schema_version``, ``protocol``, and
``records`` keys.  A record describes one ``(task_id, sample_id, slot_index)``
audio slot and has ``source`` and ``noise`` objects, each containing a path,
SHA-256 checksum, frame count, sample rate, and channel count.  The loader also
accepts JSONL records and the common aliases used by result writers.

Examples
--------
Validate a manifest and its local audio files::

    uv run python scripts/validate_gaussian_noise_manifest.py results/noise.json

Validate with expected counts::

    uv run python scripts/validate_gaussian_noise_manifest.py results/noise.json \
        --expected-tasks 12 --expected-samples 480

The module exposes :func:`validate_manifest`, :func:`summarize_outcomes`, and
:func:`compute_audio_noise_deltas` for small analysis jobs without importing
the benchmark runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import wave
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "beans_next.gaussian_noise_manifest.v1"
PROTOCOL_VERSION = "beans-next.gaussian-noise.v1"
MODALITY = "gaussian-noise"
TARGET_RMS_DBFS = -20.0
DEFAULT_RMS_TOLERANCE_DB = 0.25
DEFAULT_MEAN_TOLERANCE = 1.0e-3
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_MISSING = object()
_SUMMARY_ARTIFACT_NAMES = {
    "summary",
    "summary_json",
    "run_summary",
    "run_summary_json",
    "model_identity",
    "checkpoint",
    "run_config",
}


class ManifestError(ValueError):
    """Raised for an unreadable or malformed Gaussian-noise manifest."""


@dataclass
class ValidationReport:
    """Structured result returned by :func:`validate_manifest`.

    Parameters
    ----------
    valid
        Whether no validation errors were found.
    errors
        Human-readable failures.  An empty list means validation passed.
    warnings
        Non-fatal observations, such as a remote URI that could not be
        inspected locally.
    stats
        Counts and derived values useful to callers and the CLI.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return whether validation passed."""

        return self.valid

    def raise_if_invalid(self) -> "ValidationReport":
        """Raise :class:`ManifestError` if this report contains failures.

        Returns
        -------
        ValidationReport
            This report, allowing ``validate_manifest(...).raise_if_invalid()``.

        Raises
        ------
        ManifestError
            If one or more validation failures are present.
        """

        if not self.valid:
            raise ManifestError("\n".join(self.errors))
        return self


@dataclass(frozen=True)
class _Document:
    """Normalized manifest document."""

    metadata: Mapping[str, Any]
    records: list[Mapping[str, Any]]
    base_dir: Path


def _first(mapping: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    """Return the first present, non-null value from ``mapping``."""

    for name in names:
        value = mapping.get(name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return default


def _normalise_text(value: object) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def _load_json_or_jsonl(path: Path) -> object:
    """Read a JSON document or a JSON Lines document.

    Parameters
    ----------
    path
        Input path.

    Returns
    -------
    object
        Parsed JSON value, or a list of parsed JSON values for JSONL.

    Raises
    ------
    ManifestError
        If the path cannot be read or contains invalid JSON.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped:
        raise ManifestError(f"manifest is empty: {path}")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as whole_error:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise ManifestError(
                    f"invalid JSON at {path}:{line_number}: {line_error.msg}"
                ) from whole_error
        if not rows:
            raise ManifestError(f"manifest is empty: {path}") from whole_error
        return rows


def _records_from_value(value: object, *, source: str) -> list[Mapping[str, Any]]:
    """Extract record mappings from a list or a record envelope."""

    if isinstance(value, Mapping):
        for key in ("records", "entries", "items", "audio_slots", "slots", "samples"):
            nested = value.get(key, _MISSING)
            if nested is not _MISSING:
                return _records_from_value(nested, source=f"{source}.{key}")
        return [value]
    if isinstance(value, list):
        records: list[Mapping[str, Any]] = []
        for index, row in enumerate(value):
            if not isinstance(row, Mapping):
                raise ManifestError(f"{source}[{index}] must be a JSON object")
            records.append(row)
        return records
    raise ManifestError(f"{source} must be a JSON object or array")


def _flatten_sample_records(
    records: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Flatten per-sample ``audio_slots`` records without losing sample fields."""

    flattened: list[Mapping[str, Any]] = []
    for row in records:
        slots = _first(row, "audio_slots", "slots", "audio", default=_MISSING)
        if not isinstance(slots, list):
            flattened.append(row)
            continue
        # A record can itself be a slot record with an ``audio`` object.  Only
        # flatten lists, and inherit identity fields from the sample envelope.
        base = {
            key: value
            for key, value in row.items()
            if key not in {"audio_slots", "slots", "audio"}
        }
        for index, slot in enumerate(slots):
            if not isinstance(slot, Mapping):
                flattened.append({**base, "slot_index": index, "source": slot})
                continue
            merged = dict(base)
            merged.update(slot)
            merged.setdefault("slot_index", index)
            flattened.append(merged)
    return flattened


def _normalise_document(value: object, *, path: Path) -> _Document:
    """Normalize JSON or JSONL top-level shapes into metadata and records."""

    metadata: Mapping[str, Any]
    records_value: object
    if isinstance(value, Mapping):
        metadata = value
        records_value = _first(
            value,
            "records",
            "entries",
            "items",
            "audio_slots",
            "slots",
            "samples",
            default=_MISSING,
        )
        if records_value is _MISSING:
            records_value = []
    elif isinstance(value, list):
        # JSONL can optionally begin with a header object.  A header is any row
        # with protocol/schema keys and no slot identity.
        header: Mapping[str, Any] | None = None
        record_rows = list(value)
        if record_rows and isinstance(record_rows[0], Mapping):
            first_row = record_rows[0]
            has_header_key = any(
                key in first_row for key in ("protocol", "schema_version", "manifest")
            )
            has_slot_key = any(
                key in first_row
                for key in ("slot_index", "audio_slot", "source", "noise")
            )
            if has_header_key and not has_slot_key:
                header = first_row
                record_rows = record_rows[1:]
        metadata = header or {}
        records_value = record_rows
    else:
        raise ManifestError(f"{path} must contain a JSON object or JSONL objects")

    records = _flatten_sample_records(
        _records_from_value(records_value, source=str(path))
    )

    # Permit a JSON manifest to reference records in a separate JSONL file.
    if not records and isinstance(metadata, Mapping):
        record_path = _first(
            metadata,
            "records_path",
            "entries_path",
            "slots_path",
            "samples_path",
            default=_MISSING,
        )
        if record_path is not _MISSING:
            referenced = _resolve_path(record_path, base_dir=path.parent)
            records = _flatten_sample_records(
                _records_from_value(
                    _load_json_or_jsonl(referenced), source=str(referenced)
                )
            )
    return _Document(metadata=metadata, records=records, base_dir=path.parent)


def derive_noise_seed(
    dataset_revision: object,
    source_audio_identity: object,
    slot_index: int,
    protocol_version: object,
    global_seed: object,
) -> int:
    """Derive the reproducible seed for one Gaussian-noise audio slot.

    The byte payload is canonical JSON, so the same protocol inputs produce
    the same seed independently of Python's hash randomisation or worker order.

    Parameters
    ----------
    dataset_revision
        Immutable dataset revision or commit identifier.
    source_audio_identity
        Stable source identity, normally the source SHA-256 checksum.
    slot_index
        Zero-based audio slot index within the sample.
    protocol_version
        Noise protocol version.
    global_seed
        Recorded global seed for the evaluation.

    Returns
    -------
    int
        A non-negative 63-bit seed.

    Raises
    ------
    ValueError
        If ``slot_index`` is negative or ``global_seed`` is not an integer.
    """

    if (
        isinstance(slot_index, bool)
        or not isinstance(slot_index, int)
        or slot_index < 0
    ):
        raise ValueError("slot_index must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, (str, int, float)):
        raise ValueError("global_seed must be a JSON scalar")
    if isinstance(global_seed, float) and not math.isfinite(global_seed):
        raise ValueError("global_seed must be finite")
    value = int.from_bytes(
        bytes.fromhex(
            derive_seed_sha256(
                dataset_revision,
                source_audio_identity,
                slot_index,
                protocol_version,
                global_seed,
            )[:16]
        ),
        "big",
        signed=False,
    )
    return value


def derive_seed_sha256(
    dataset_revision: object,
    source_audio_identity: object,
    slot_index: int,
    protocol_version: object,
    global_seed: object,
) -> str:
    """Return the full SHA-256 digest used for deterministic seed derivation.

    Returns
    -------
    str
        Lower-case hexadecimal SHA-256 digest.  The first eight bytes interpreted
        as an unsigned big-endian integer are the corresponding noise seed.
    """

    if (
        isinstance(slot_index, bool)
        or not isinstance(slot_index, int)
        or slot_index < 0
    ):
        raise ValueError("slot_index must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, (str, int, float)):
        raise ValueError("global_seed must be a JSON scalar")
    if isinstance(global_seed, float) and not math.isfinite(global_seed):
        raise ValueError("global_seed must be finite")
    payload = json.dumps(
        {
            "dataset_revision": str(dataset_revision),
            "source_identity": str(source_audio_identity),
            "slot_index": slot_index,
            "protocol_version": str(protocol_version),
            "global_seed": global_seed,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# A short alias is convenient for generation scripts and backwards-compatible
# with early experiment notebooks that called this helper ``derive_seed``.
derive_seed = derive_noise_seed


def _resolve_path(value: object, *, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"audio path must be a non-empty string, got {value!r}")
    raw = value.strip()
    if raw.startswith("file://"):
        raw = raw[7:]
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _checksum(value: object) -> str | None:
    if isinstance(value, Mapping):
        value = _first(value, "sha256", "checksum", "hash", default=None)
    if not isinstance(value, str):
        return None
    result = value.strip().lower()
    if result.startswith("sha256:"):
        result = result[7:]
    return result if _SHA256_RE.fullmatch(result) else None


def _checksum_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestError(f"cannot read audio artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _seed_scalar(value: object) -> object | None:
    """Validate the integer/string seed scalar used by the production config."""

    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _audio_properties(path: Path) -> dict[str, float | int]:
    """Return frames, sample rate, channels, mean, RMS dBFS, and peak.

    ``soundfile`` is used when available for WAV/FLAC/OGG support.  The small
    ``wave`` fallback keeps validation useful in minimal CI environments.
    """

    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except ImportError:
        sf = None
    if sf is not None:
        try:
            info = sf.info(str(path))
            total = 0
            sum_value = 0.0
            sum_square = 0.0
            peak = 0.0
            for block in sf.blocks(
                str(path), blocksize=131072, dtype="float64", always_2d=True
            ):
                if block.size == 0:
                    continue
                total += int(block.size)
                sum_value += float(block.sum())
                sum_square += float((block * block).sum())
                peak = max(peak, float(abs(block).max()))
            if total == 0:
                raise ManifestError(f"audio artifact has zero samples: {path}")
            mean = sum_value / total
            rms = math.sqrt(sum_square / total)
            rms_dbfs = 20.0 * math.log10(rms) if rms > 0 else float("-inf")
            return {
                "frames": int(info.frames),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "duration_seconds": float(info.frames / info.samplerate),
                "mean": mean,
                "rms_dbfs": rms_dbfs,
                "rms": rms,
                "peak": peak,
            }
        except Exception as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(f"cannot inspect audio artifact {path}: {exc}") from exc

    try:
        with wave.open(str(path), "rb") as stream:
            frames = stream.getnframes()
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            width = stream.getsampwidth()
            raw = stream.readframes(frames)
    except (OSError, wave.Error) as exc:
        raise ManifestError(
            f"cannot inspect {path}; install soundfile for non-WAV audio ({exc})"
        ) from exc
    if frames <= 0 or channels <= 0 or sample_rate <= 0:
        raise ManifestError(f"audio artifact has invalid shape: {path}")
    if width not in (1, 2, 3, 4):
        raise ManifestError(f"unsupported PCM sample width {width} in {path}")
    values: list[float] = []
    scale = float(1 << (8 * width - 1))
    if width == 1:
        values = [(byte - 128) / 128.0 for byte in raw]
    else:
        for offset in range(0, len(raw), width):
            chunk = raw[offset : offset + width]
            integer = int.from_bytes(chunk, "little", signed=True)
            values.append(integer / scale)
    total = len(values)
    mean = sum(values) / total
    rms = math.sqrt(sum(value * value for value in values) / total)
    return {
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": float(frames / sample_rate),
        "mean": mean,
        "rms_dbfs": 20.0 * math.log10(rms) if rms > 0 else float("-inf"),
        "rms": rms,
        "peak": max(abs(value) for value in values),
    }


def _audio_object(record: Mapping[str, Any], kind: str) -> Mapping[str, Any] | None:
    value = _first(record, kind, f"{kind}_audio", f"{kind}_artifact", default=None)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        return {"path": value}
    # Production run manifests flatten provenance into one record.  Rebuild a
    # lightweight audio object so the same checksum/property checks apply.
    if kind == "source":
        path = _first(record, "source_path", default=_MISSING)
        checksum = _first(record, "source_sha256", default=_MISSING)
    else:
        path = _first(record, "noise_path", "path", default=_MISSING)
        checksum = _first(record, "audio_sha256", "noise_sha256", default=_MISSING)
    if path is _MISSING and checksum is _MISSING:
        return None
    result: dict[str, Any] = {
        "path": path,
        "sha256": checksum,
        "frames": record.get("frames", _MISSING),
        "sample_rate": record.get("sample_rate", _MISSING),
        "channels": record.get("channels", _MISSING),
        "duration_seconds": record.get("duration_seconds", _MISSING),
    }
    if kind == "noise":
        for name in ("mean", "rms", "rms_dbfs", "peak"):
            if name in record:
                result[name] = record[name]
        if "seed" in record:
            result["seed"] = record["seed"]
    return result
    return None


def _property(value: Mapping[str, Any], *names: str) -> Any:
    return _first(value, *names, default=_MISSING)


def _validate_audio_object(
    obj: Mapping[str, Any] | None,
    *,
    kind: str,
    record_label: str,
    base_dir: Path,
    errors: list[str],
    warnings: list[str],
    check_audio: bool,
    rms_target: float,
    mean_tolerance: float,
    rms_tolerance_db: float,
) -> dict[str, Any] | None:
    if obj is None:
        errors.append(f"{record_label}: missing {kind} audio object")
        return None
    path_value = _property(obj, "path", "uri", "file", "audio_path")
    if path_value is _MISSING:
        errors.append(f"{record_label}.{kind}: missing path")
        path = None
    else:
        try:
            path = _resolve_path(path_value, base_dir=base_dir)
        except ManifestError as exc:
            errors.append(f"{record_label}.{kind}: {exc}")
            path = None

    checksum_value = _property(obj, "sha256", "checksum", "hash")
    checksum = _checksum(checksum_value)
    if checksum is None:
        errors.append(
            f"{record_label}.{kind}: sha256 checksum must be 64 hexadecimal characters"
        )
    declared: dict[str, Any] = {}
    required_properties = (
        ("frames", ("frames", "num_frames", "n_frames")),
        ("sample_rate", ("sample_rate", "samplerate", "sampling_rate")),
        ("channels", ("channels", "n_channels")),
    )
    for canonical, aliases in required_properties:
        value = _property(obj, *aliases)
        integer = _integer(value)
        if integer is None or integer <= 0:
            errors.append(
                f"{record_label}.{kind}: {canonical} must be a positive integer"
            )
        else:
            declared[canonical] = integer
    for canonical, aliases in (
        ("mean", ("mean", "waveform_mean")),
        ("rms", ("rms", "linear_rms")),
        ("rms_dbfs", ("rms_dbfs", "rms_db")),
        ("peak", ("peak", "max_abs", "peak_abs")),
    ):
        value = _property(obj, *aliases)
        number = _number(value)
        if value is not _MISSING and number is None:
            errors.append(f"{record_label}.{kind}: {canonical} must be finite numeric")
        elif number is not None:
            declared[canonical] = number

    if path is None:
        return declared
    if not path.is_file():
        # URI-like paths are not inspectable on this host.  They still require
        # all declared metadata and checksum, but are reported as a failure for
        # local paths to prevent silently validating an incomplete cache.
        if isinstance(path_value, str) and "://" in path_value:
            warnings.append(
                f"{record_label}.{kind}: remote URI not inspected: {path_value}"
            )
        else:
            errors.append(f"{record_label}.{kind}: audio file does not exist: {path}")
        return declared
    if not check_audio:
        return declared
    try:
        actual_checksum = _checksum_for_path(path)
        if checksum is not None and actual_checksum != checksum:
            errors.append(
                f"{record_label}.{kind}: checksum mismatch (manifest {checksum}, file {actual_checksum})"
            )
        actual = _audio_properties(path)
    except ManifestError as exc:
        errors.append(f"{record_label}.{kind}: {exc}")
        return declared
    for property_name in ("frames", "sample_rate", "channels"):
        if (
            property_name in declared
            and actual[property_name] != declared[property_name]
        ):
            errors.append(
                f"{record_label}.{kind}: {property_name} mismatch "
                f"(manifest {declared[property_name]}, file {actual[property_name]})"
            )
    if kind == "noise":
        actual_rms = float(actual["rms_dbfs"])
        if abs(actual_rms - rms_target) > rms_tolerance_db:
            errors.append(
                f"{record_label}.noise: RMS {actual_rms:.4f} dBFS is outside "
                f"target {rms_target:.4f} ± {rms_tolerance_db:.4f} dB"
            )
        if abs(float(actual["mean"])) > mean_tolerance:
            errors.append(
                f"{record_label}.noise: waveform mean {float(actual['mean']):.6g} "
                f"exceeds tolerance {mean_tolerance:.6g}"
            )
        if float(actual["peak"]) > 1.0 + 1.0e-9:
            errors.append(
                f"{record_label}.noise: waveform clips (peak {float(actual['peak']):.6g} > 1)"
            )
    duration = _number(_property(obj, "duration_seconds", "duration"))
    if duration is not None:
        declared["duration_seconds"] = duration
        if abs(duration - float(actual["duration_seconds"])) > 1.0e-9:
            errors.append(
                f"{record_label}.{kind}: duration_seconds mismatch "
                f"(manifest {duration}, file {actual['duration_seconds']})"
            )
    for property_name in ("mean", "rms", "rms_dbfs", "peak"):
        if property_name in declared:
            tolerance = (
                mean_tolerance
                if property_name == "mean"
                else 1.0e-3
                if property_name == "rms"
                else rms_tolerance_db
                if property_name == "rms_dbfs"
                else 1.0e-6
            )
            if abs(float(actual[property_name]) - declared[property_name]) > tolerance:
                errors.append(
                    f"{record_label}.{kind}: {property_name} mismatch "
                    f"(manifest {declared[property_name]}, file {actual[property_name]})"
                )
    if "rms" not in actual:
        rms_dbfs = float(actual["rms_dbfs"])
        actual["rms"] = 10.0 ** (rms_dbfs / 20.0) if math.isfinite(rms_dbfs) else 0.0
    return {**declared, **actual, "path": str(path), "sha256": actual_checksum}


def _protocol(document: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _first(document, "protocol", "noise_protocol", "metadata", default=_MISSING)
    if value is _MISSING and isinstance(document.get("manifest"), Mapping):
        value = _first(
            document["manifest"],
            "protocol",
            "noise_protocol",
            "metadata",
            default=_MISSING,
        )
    return value if isinstance(value, Mapping) else {}


def _schema_version(document: Mapping[str, Any]) -> object:
    value = _first(
        document, "schema_version", "manifest_schema", "format", default=_MISSING
    )
    if value is _MISSING and isinstance(document.get("manifest"), Mapping):
        value = _first(
            document["manifest"], "schema_version", "manifest_schema", default=_MISSING
        )
    return value


def _validate_protocol(
    document: Mapping[str, Any], errors: list[str]
) -> tuple[Mapping[str, Any], float, float, float, object | None]:
    schema = _schema_version(document)
    schema_norm = _normalise_text(schema) if schema is not _MISSING else ""
    accepted_schemas = {
        "beans-next.gaussian-noise-manifest.v1",
        "gaussian-noise-manifest-v1",
        "gaussian-noise-manifest-1",
        "gaussian-noise-v1",
    }
    if schema_norm not in accepted_schemas:
        errors.append(
            "schema_version must identify a Gaussian-noise manifest "
            f"(expected {SCHEMA_VERSION!r}, got {schema!r})"
        )
    raw_protocol = _first(document, "protocol", "noise_protocol", default=_MISSING)
    protocol = _protocol(document)
    if raw_protocol is _MISSING and not protocol:
        errors.append("missing protocol object")
    protocol_text = (
        _normalise_text(raw_protocol) if isinstance(raw_protocol, str) else ""
    )
    mode = _first(
        protocol,
        "mode",
        "modality",
        "modality_mode",
        default=_first(document, "mode", "modality", "modality_mode", default=_MISSING),
    )
    if _normalise_text(mode) != MODALITY and "gaussian-noise" not in protocol_text:
        errors.append(f"protocol mode must be {MODALITY!r}, got {mode!r}")
    distribution = _first(
        protocol,
        "distribution",
        "noise_distribution",
        default=raw_protocol if isinstance(raw_protocol, str) else _MISSING,
    )
    if distribution is _MISSING:
        errors.append("protocol distribution is required")
    elif _normalise_text(distribution) not in {
        "normal",
        "gaussian",
        "white-gaussian",
        "white-noise-gaussian",
        "zero-mean-white-gaussian-noise",
    }:
        errors.append(
            f"protocol distribution must be Gaussian/normal, got {distribution!r}"
        )
    protocol_version = _first(
        protocol,
        "protocol_version",
        "version",
        default=_first(document, "protocol_version", default=_MISSING),
    )
    if (
        protocol_version is _MISSING
        or not isinstance(protocol_version, str)
        or not protocol_version.strip()
    ):
        errors.append("protocol_version is required")
        protocol_version = PROTOCOL_VERSION
    rms_target_value = _first(
        protocol,
        "rms_dbfs",
        "target_rms_dbfs",
        "rms_db",
        default=_first(
            document,
            "rms_dbfs",
            "target_rms_dbfs",
            "rms_db",
            default=TARGET_RMS_DBFS,
        ),
    )
    rms_target = _number(rms_target_value)
    if rms_target is None:
        errors.append("protocol rms_dbfs must be finite numeric")
        rms_target = TARGET_RMS_DBFS
    elif abs(rms_target - TARGET_RMS_DBFS) > 1.0e-6:
        errors.append(
            f"protocol rms_dbfs must be exactly {TARGET_RMS_DBFS:g}, got {rms_target:g}"
        )
    mean_target_value = _first(
        protocol,
        "mean",
        "target_mean",
        "waveform_mean",
        default=_first(document, "mean", "target_mean", default=0.0),
    )
    mean_target = _number(mean_target_value)
    if mean_target is None:
        errors.append("protocol mean must be finite numeric")
        mean_target = 0.0
    elif abs(mean_target) > DEFAULT_MEAN_TOLERANCE:
        errors.append(f"protocol mean must be zero, got {mean_target:g}")
    global_seed_value = _first(
        protocol,
        "global_seed",
        "seed",
        default=_first(document, "global_seed", default=_MISSING),
    )
    global_seed = _seed_scalar(global_seed_value)
    if global_seed is None:
        errors.append("global_seed is required and must be an integer")
    dataset_revision = _first(
        protocol,
        "dataset_revision",
        "dataset_commit",
        "dataset_version",
        default=_first(document, "dataset_revision", default=_MISSING),
    )
    if (
        dataset_revision is _MISSING
        or not isinstance(dataset_revision, str)
        or not dataset_revision.strip()
    ):
        errors.append("dataset_revision is required")
    code_commit = _first(
        protocol,
        "code_commit",
        "git_commit",
        "commit",
        default=_first(document, "code_commit", "git_commit", default=_MISSING),
    )
    if (
        code_commit is _MISSING
        or not isinstance(code_commit, str)
        or not code_commit.strip()
    ):
        errors.append("code_commit is required")
    rms_tolerance_value = _first(
        protocol, "rms_tolerance_db", default=DEFAULT_RMS_TOLERANCE_DB
    )
    rms_tolerance = _number(rms_tolerance_value)
    if rms_tolerance is None:
        errors.append("protocol rms_tolerance_db must be a non-negative number")
        rms_tolerance = DEFAULT_RMS_TOLERANCE_DB
    return protocol, rms_target, mean_target, rms_tolerance, global_seed


def _record_identity(
    record: Mapping[str, Any], index: int
) -> tuple[str, str, int] | None:
    sample_id = _first(record, "sample_id", "sample", "id", default=_MISSING)
    task_id = _first(record, "task_id", "task", "task_name", default="__default_task__")
    slot_value = _first(
        record,
        "slot_index",
        "audio_slot_index",
        "audio_index",
        "slot",
        "audio_slot",
        default=_MISSING,
    )
    if (
        sample_id is _MISSING
        or not isinstance(sample_id, (str, int))
        or not str(sample_id).strip()
    ):
        return None
    slot = _integer(slot_value)
    if slot is None:
        return None
    return str(task_id), str(sample_id), slot


def _artifact_rows(value: object, *, path: Path) -> list[Mapping[str, Any]]:
    """Load rows from an artifact path or already parsed artifact value."""

    if isinstance(value, (str, Path)):
        artifact_path = _resolve_path(value, base_dir=path.parent)
        return _artifact_rows(_load_json_or_jsonl(artifact_path), path=artifact_path)
    if isinstance(value, Mapping):
        for key in ("rows", "items", "records", "responses", "predictions"):
            if key in value and isinstance(value[key], list):
                return _artifact_rows(value[key], path=path)
        return [value]
    if isinstance(value, list):
        rows: list[Mapping[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ManifestError(f"artifact {path}[{index}] must be a JSON object")
            rows.append(item)
        return rows
    raise ManifestError(f"artifact {path} must be a path, object, or array")


def _artifact_key(row: Mapping[str, Any]) -> tuple[str | None, str, int | None] | None:
    sample = _first(row, "sample_id", "sample", "id", default=_MISSING)
    if sample is _MISSING or sample is None or not str(sample).strip():
        return None
    task_value = _first(row, "task_id", "task", "task_name", default=_MISSING)
    task = None if task_value is _MISSING else str(task_value)
    slot_value = _first(
        row, "slot_index", "audio_slot_index", "audio_index", "slot", default=_MISSING
    )
    slot = _integer(slot_value) if slot_value is not _MISSING else None
    return task, str(sample), slot


def _expected_counts(document: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _first(document, "expected_counts", "expected", "counts", default=_MISSING)
    if value is _MISSING and isinstance(document.get("manifest"), Mapping):
        value = _first(
            document["manifest"], "expected_counts", "expected", "counts", default={}
        )
    return value if isinstance(value, Mapping) else {}


def _count_value(mapping: Mapping[str, Any], *names: str) -> int | None:
    value = _first(mapping, *names, default=_MISSING)
    return _integer(value)


def _check_production_sidecar(
    record: Mapping[str, Any],
    *,
    label: str,
    base_dir: Path,
    errors: list[str],
) -> None:
    """Check the immutable sidecar referenced by a production manifest row."""

    is_flat = any(
        key in record
        for key in ("source_path", "noise_path", "audio_sha256", "seed_sha256")
    )
    if not is_flat:
        return
    metadata_value = _first(record, "metadata_path", "sidecar_path", default=_MISSING)
    if metadata_value is _MISSING:
        errors.append(f"{label}: metadata_path is required in production manifest rows")
        return
    try:
        metadata_path = _resolve_path(metadata_value, base_dir=base_dir)
    except ManifestError as exc:
        errors.append(f"{label}: metadata_path: {exc}")
        return
    if not metadata_path.is_file():
        errors.append(f"{label}: metadata sidecar does not exist: {metadata_path}")
        return
    try:
        metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot parse metadata sidecar {metadata_path}: {exc}")
        return
    if not isinstance(metadata_value, Mapping):
        errors.append(f"{label}: metadata sidecar must contain a JSON object")
        return
    if metadata_value.get("schema_version") not in {
        "beans_next.gaussian_noise.v1",
        "beans-next.gaussian-noise.v1",
    }:
        errors.append(f"{label}: metadata sidecar has unexpected schema_version")
    comparisons = {
        "source_identity": "source_identity",
        "slot_index": "slot_index",
        "seed": "seed",
        "seed_sha256": "seed_sha256",
        "sha256": "audio_sha256",
        "frames": "frames",
        "sample_rate": "sample_rate",
        "channels": "channels",
    }
    for sidecar_name, record_name in comparisons.items():
        expected = record.get(record_name, _MISSING)
        observed = metadata_value.get(sidecar_name, _MISSING)
        if expected is not _MISSING and observed != expected:
            errors.append(
                f"{label}: metadata sidecar {sidecar_name} does not match manifest"
            )


def _validate_artifacts(
    document: Mapping[str, Any],
    *,
    path: Path,
    manifest_keys: list[tuple[str, str, int]],
    errors: list[str],
) -> dict[str, int]:
    artifacts_value = _first(document, "artifacts", "artifact_paths", default={})
    if not isinstance(artifacts_value, Mapping):
        errors.append("artifacts must be a JSON object when provided")
        return {}
    counts: dict[str, int] = {}
    artifact_key_lists: dict[str, list[tuple[str | None, str, int | None]]] = {}
    for name, value in artifacts_value.items():
        normalized_name = _normalise_text(name).replace("-", "_")
        if normalized_name in _SUMMARY_ARTIFACT_NAMES | {
            "manifest",
            "noise",
            "source",
            "metadata",
        } or normalized_name.startswith(
            ("summary", "run_summary", "model_identity", "checkpoint", "run_config")
        ):
            continue
        try:
            rows = _artifact_rows(value, path=path)
        except ManifestError as exc:
            errors.append(f"artifact {name!r}: {exc}")
            continue
        keys: list[tuple[str, str, int | None]] = []
        for row_index, row in enumerate(rows):
            key = _artifact_key(row)
            if key is None:
                errors.append(f"artifact {name!r}[{row_index}] is missing sample_id")
            else:
                keys.append(key)
        counts[str(name)] = len(rows)
        artifact_key_lists[str(name)] = keys
    expected_sample_keys: list[tuple[str, str, int | None]] = [
        (task, sample, None) for task, sample in _ordered_sample_keys(manifest_keys)
    ]
    expected_slot_keys = [(task, sample, slot) for task, sample, slot in manifest_keys]
    for name, keys in artifact_key_lists.items():
        if not keys:
            continue
        if all(key[2] is None for key in keys):
            keys_match = (
                keys == expected_sample_keys
                or all(key[0] is None for key in keys)
                and [key[1] for key in keys] == [key[1] for key in expected_sample_keys]
            )
            if not keys_match:
                errors.append(
                    f"artifact {name!r} is not aligned to manifest sample order "
                    f"(got {len(keys)} rows, expected {len(expected_sample_keys)})"
                )
        elif keys != expected_slot_keys and not (
            all(key[0] is None for key in keys)
            and [(key[1], key[2]) for key in keys]
            == [(key[1], key[2]) for key in expected_slot_keys]
        ):
            errors.append(
                f"artifact {name!r} is not aligned to manifest slot order "
                f"(got {len(keys)} rows, expected {len(expected_slot_keys)})"
            )
    names = list(artifact_key_lists)
    if names:
        first = artifact_key_lists[names[0]]
        for name in names[1:]:
            other = artifact_key_lists[name]
            same_alignment = other == first or (
                len(other) == len(first)
                and all(left[1:] == right[1:] for left, right in zip(first, other, strict=False))
            )
            if not same_alignment:
                errors.append(
                    f"artifacts {names[0]!r} and {name!r} have different ID alignment"
                )
    return counts


def _ordered_sample_keys(keys: Sequence[tuple[str, str, int]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for task, sample, _slot in keys:
        value = (task, sample)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def validate_manifest(
    manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    expected_tasks: int | None = None,
    expected_samples: int | None = None,
    expected_slots: int | None = None,
    check_audio: bool = True,
    check_artifacts: bool = True,
    base_dir: str | Path | None = None,
) -> ValidationReport:
    """Strictly validate a Gaussian-noise manifest.

    Parameters
    ----------
    manifest
        Path to JSON/JSONL, or an already parsed mapping/list.
    expected_tasks, expected_samples, expected_slots
        Optional command-line expectations.  Manifest-provided expected counts
        are checked whenever present.
    check_audio
        If true, inspect local audio files and verify their checksum and audio
        properties.  Metadata is still checked when false.
    check_artifacts
        If true, validate any listed prediction/processed/scored artifact
        paths and their sample/slot order.
    base_dir
        Base directory for relative paths when ``manifest`` is in-memory.

    Returns
    -------
    ValidationReport
        Report with ``valid=False`` and explanatory ``errors`` on failure.
    """

    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).expanduser().resolve()
        try:
            document = _normalise_document(_load_json_or_jsonl(path), path=path)
        except ManifestError as exc:
            return ValidationReport(False, [str(exc)], [], {})
    else:
        path = Path(base_dir or ".").expanduser().resolve() / "<in-memory-manifest>"
        try:
            document = _normalise_document(manifest, path=path)
        except ManifestError as exc:
            return ValidationReport(False, [str(exc)], [], {})

    protocol, rms_target, _mean_target, rms_tolerance, global_seed = _validate_protocol(
        document.metadata, errors
    )
    mean_tolerance_value = _first(
        protocol, "mean_tolerance", default=DEFAULT_MEAN_TOLERANCE
    )
    mean_tolerance = _number(mean_tolerance_value)
    if mean_tolerance is None or mean_tolerance < 0:
        errors.append("protocol mean_tolerance must be a non-negative number")
        mean_tolerance = DEFAULT_MEAN_TOLERANCE
    if rms_tolerance < 0 or not math.isfinite(rms_tolerance):
        errors.append("protocol rms_tolerance_db must be a non-negative number")
        rms_tolerance = DEFAULT_RMS_TOLERANCE_DB
    dataset_revision = _first(
        protocol,
        "dataset_revision",
        "dataset_commit",
        "dataset_version",
        default=_first(document.metadata, "dataset_revision", default=""),
    )
    protocol_version = _first(
        protocol, "protocol_version", "version", default=PROTOCOL_VERSION
    )

    record_keys: list[tuple[str, str, int]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    closed_groups: set[tuple[str, str]] = set()
    current_group: tuple[str, str] | None = None
    actual_seeds: set[int] = set()
    for index, record in enumerate(document.records):
        label = f"record[{index}]"
        identity = _record_identity(record, index)
        if identity is None:
            errors.append(
                f"{label}: requires non-empty sample_id, task_id (or default), and non-negative integer slot_index"
            )
            continue
        task_id, sample_id, slot_index = identity
        group = (task_id, sample_id)
        key = identity
        if key in seen_keys:
            errors.append(f"{label}: duplicate task/sample/slot record {key!r}")
        seen_keys.add(key)
        record_keys.append(key)
        if current_group != group:
            if group in closed_groups:
                errors.append(
                    f"{label}: sample {group!r} is out of order (its slots are not contiguous)"
                )
            if current_group is not None:
                closed_groups.add(current_group)
            current_group = group
        expected_slot = sum(1 for item in record_keys[:-1] if item[:2] == group)
        if slot_index != expected_slot:
            errors.append(
                f"{label}: slot_index {slot_index} breaks contiguous slot order for {group!r}; expected {expected_slot}"
            )
        for field_name, aliases in (
            ("record_index", ("record_index", "ordinal", "index")),
            (
                "placeholder_index",
                ("placeholder_index", "audio_placeholder_index", "prompt_slot_index"),
            ),
        ):
            declared = _first(record, *aliases, default=_MISSING)
            if declared is not _MISSING:
                number = _integer(declared)
                expected = slot_index if field_name == "placeholder_index" else index
                if number is None or number != expected:
                    errors.append(
                        f"{label}: {field_name} {declared!r} is not aligned (expected {expected})"
                    )
        source = _audio_object(record, "source")
        noise = _audio_object(record, "noise")
        source_result = _validate_audio_object(
            source,
            kind="source",
            record_label=label,
            base_dir=document.base_dir,
            errors=errors,
            warnings=warnings,
            check_audio=check_audio,
            rms_target=rms_target,
            mean_tolerance=mean_tolerance,
            rms_tolerance_db=rms_tolerance,
        )
        noise_result = _validate_audio_object(
            noise,
            kind="noise",
            record_label=label,
            base_dir=document.base_dir,
            errors=errors,
            warnings=warnings,
            check_audio=check_audio,
            rms_target=rms_target,
            mean_tolerance=mean_tolerance,
            rms_tolerance_db=rms_tolerance,
        )
        if source_result is not None and noise_result is not None:
            for property_name in ("frames", "sample_rate", "channels"):
                if (
                    property_name in source_result
                    and property_name in noise_result
                    and source_result[property_name] != noise_result[property_name]
                ):
                    errors.append(
                        f"{label}: source/noise {property_name} mismatch "
                        f"({source_result[property_name]} vs {noise_result[property_name]})"
                    )
            source_checksum = source_result.get("sha256")
            noise_checksum = noise_result.get("sha256")
            if source_checksum and noise_checksum and source_checksum == noise_checksum:
                errors.append(f"{label}: source and noise checksums are identical")
        for kind, audio_object in (("source", source), ("noise", noise)):
            if audio_object is None:
                continue
            nested_slot = _first(
                audio_object,
                "slot_index",
                "audio_slot_index",
                "audio_index",
                "slot",
                default=_MISSING,
            )
            if nested_slot is not _MISSING and _integer(nested_slot) != slot_index:
                errors.append(
                    f"{label}.{kind}: nested slot index {nested_slot!r} "
                    f"does not match record slot_index {slot_index}"
                )
        source_identity = _first(
            record,
            "source_audio_identity",
            "source_identity",
            "source_id",
            default=(source_result or {}).get("sha256", _MISSING),
        )
        if source_identity is _MISSING or not str(source_identity).strip():
            errors.append(
                f"{label}: source_audio_identity is required for seed derivation"
            )
        seed_value = _first(
            record,
            "seed",
            "noise_seed",
            default=(noise or {}).get("seed", _MISSING) if noise else _MISSING,
        )
        seed = _integer(seed_value)
        if seed is None:
            errors.append(
                f"{label}: deterministic seed is required and must be an integer"
            )
        elif global_seed is not None and source_identity is not _MISSING:
            try:
                expected_seed_digest = derive_seed_sha256(
                    dataset_revision,
                    source_identity,
                    slot_index,
                    protocol_version,
                    global_seed,
                )
                expected_seed = int.from_bytes(
                    bytes.fromhex(expected_seed_digest[:16]),
                    byteorder="big",
                    signed=False,
                )
                if seed != expected_seed:
                    errors.append(
                        f"{label}: seed {seed} does not match deterministic derivation {expected_seed}"
                    )
                seed_digest = _checksum(_first(record, "seed_sha256", default=None))
                is_flat = any(
                    key in record
                    for key in (
                        "source_path",
                        "noise_path",
                        "audio_sha256",
                        "seed_sha256",
                    )
                )
                if is_flat and seed_digest is None:
                    errors.append(
                        f"{label}: seed_sha256 must be a 64-character hexadecimal digest"
                    )
                elif seed_digest is not None and seed_digest != expected_seed_digest:
                    errors.append(
                        f"{label}: seed_sha256 does not match deterministic derivation"
                    )
            except ValueError as exc:
                errors.append(f"{label}: cannot derive deterministic seed: {exc}")
        if seed is not None:
            if seed in actual_seeds:
                warnings.append(
                    f"{label}: deterministic seed {seed} is reused; this is allowed for repeated source identities"
                )
            actual_seeds.add(seed)
        _check_production_sidecar(
            record, label=label, base_dir=document.base_dir, errors=errors
        )
        placeholder_count = _first(
            record,
            "placeholder_count",
            "audio_placeholder_count",
            "num_audio_placeholders",
            default=_MISSING,
        )
        if placeholder_count is not _MISSING:
            count = _integer(placeholder_count)
            if count is None or count <= slot_index:
                errors.append(
                    f"{label}: placeholder_count must be greater than slot_index ({slot_index})"
                )

    expected = _expected_counts(document.metadata)
    expected_task_count = expected_tasks
    expected_sample_count = expected_samples
    expected_slot_count = expected_slots
    if expected_task_count is None:
        expected_task_count = _count_value(
            expected, "tasks", "task_count", "expected_tasks"
        )
        if expected_task_count is None:
            expected_task_count = _count_value(
                document.metadata, "expected_task_count", "task_count"
            )
    if expected_sample_count is None:
        expected_sample_count = _count_value(
            expected, "samples", "sample_count", "expected_samples"
        )
        if expected_sample_count is None:
            expected_sample_count = _count_value(
                document.metadata, "expected_sample_count", "sample_count"
            )
    if expected_slot_count is None:
        expected_slot_count = _count_value(
            expected, "slots", "slot_count", "audio_slots", "expected_slots"
        )
        if expected_slot_count is None:
            expected_slot_count = _count_value(
                document.metadata, "expected_slot_count", "slot_count"
            )
    task_count = len({task for task, _sample, _slot in record_keys})
    sample_count = len({(task, sample) for task, sample, _slot in record_keys})
    slot_count = len(record_keys)
    if not record_keys:
        errors.append("manifest records must contain at least one audio slot")
    observed_counts = {
        "tasks": task_count,
        "samples": sample_count,
        "slots": slot_count,
    }
    for label, expected_value, actual_value in (
        ("tasks", expected_task_count, task_count),
        ("samples", expected_sample_count, sample_count),
        ("slots", expected_slot_count, slot_count),
    ):
        if expected_value is not None:
            if expected_value < 0:
                errors.append(f"expected {label} count must be non-negative")
            elif expected_value != actual_value:
                errors.append(
                    f"expected {label} count {expected_value}, found {actual_value}"
                )
    artifact_counts: dict[str, int] = {}
    if (
        check_artifacts
        and _first(document.metadata, "artifacts", "artifact_paths", default=_MISSING)
        is not _MISSING
    ):
        artifact_counts = _validate_artifacts(
            document.metadata, path=path, manifest_keys=record_keys, errors=errors
        )

    stats: dict[str, Any] = {
        **observed_counts,
        "artifact_counts": artifact_counts,
        "schema_version": _schema_version(document.metadata),
        "protocol_version": protocol_version,
        "rms_dbfs": rms_target,
        "global_seed": global_seed,
    }
    return ValidationReport(not errors, errors, warnings, stats)


def validate_manifest_file(path: str | Path, **kwargs: Any) -> ValidationReport:
    """Alias for :func:`validate_manifest` accepting a manifest file path."""

    return validate_manifest(path, **kwargs)


def _rows_for_analysis(value: object) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        return _artifact_rows(_load_json_or_jsonl(path), path=path)
    if isinstance(value, Mapping):
        for key in ("rows", "items", "records", "responses", "predictions"):
            if key in value and isinstance(value[key], list):
                return _artifact_rows(value[key], path=Path("<in-memory-artifact>"))
        # Summary JSONs can provide aggregate metrics directly.
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [row for row in value if isinstance(row, Mapping)]
    raise ManifestError(
        "analysis input must be a path, mapping, or sequence of mappings"
    )


def summarize_outcomes(value: object) -> dict[str, Any]:
    """Count parse failures, refusals, errors, and finish reasons.

    Parameters
    ----------
    value
        Predictions JSON/JSONL path, parsed rows, or a summary mapping.

    Returns
    -------
    dict
        ``total``, counts for ``parse_failures``, ``refusals``, ``errors``, a
        ``finish_reasons`` mapping, and corresponding rates in ``[0, 1]``.
    """

    rows = _rows_for_analysis(value)
    parse_failures = 0
    refusals = 0
    errors = 0
    reasons: Counter[str] = Counter()
    for row in rows:
        reason_value = _first(row, "finish_reason", "finishReason", default=None)
        if reason_value is not None:
            reasons[str(reason_value)] += 1
        error_value = _first(row, "error", "exception", "request_error", default=None)
        status = _normalise_text(_first(row, "status", default=""))
        if error_value not in (None, "", [], {}):
            errors += 1
        elif status in {"error", "failed", "failure", "exception"}:
            errors += 1
        parse_flag = _first(
            row,
            "parse_failure",
            "parse_failed",
            "parse_error",
            "parser_error",
            default=None,
        )
        parsed = _first(row, "parsed", "parse_ok", default=_MISSING)
        if parse_flag not in (None, False, "", [], {}):
            parse_failures += 1
        elif parsed is False:
            parse_failures += 1
        elif isinstance(error_value, str) and "parse" in error_value.lower():
            parse_failures += 1
        refusal = _first(row, "refusal", "is_refusal", "refused", default=None)
        text = _first(row, "text", "prediction", "response", default="")
        reason_norm = _normalise_text(reason_value) if reason_value is not None else ""
        if (
            refusal is True
            or (isinstance(refusal, str) and refusal.strip())
            or "refus" in reason_norm
        ):
            refusals += 1
        elif isinstance(text, str) and text.strip().lower() in {
            "refusal",
            "refused",
            "i cannot answer",
        }:
            refusals += 1
    total = len(rows)
    denominator = float(total) if total else 1.0
    return {
        "total": total,
        "parse_failures": parse_failures,
        "refusals": refusals,
        "errors": errors,
        "finish_reasons": dict(sorted(reasons.items())),
        "parse_failure_rate": parse_failures / denominator if total else 0.0,
        "refusal_rate": refusals / denominator if total else 0.0,
        "error_rate": errors / denominator if total else 0.0,
    }


def _metric_values(row: Mapping[str, Any]) -> dict[str, float]:
    """Extract numeric metrics from a result row or summary."""

    values: dict[str, float] = {}
    for key in ("score", "value", "metric", "accuracy", "cider", "spider"):
        number = _number(row.get(key, _MISSING))
        if number is not None:
            values[key] = number
    for container_name in ("scores", "metrics"):
        container = row.get(container_name)
        if isinstance(container, Mapping):
            nested_mean = container.get("mean")
            if isinstance(nested_mean, Mapping):
                container = nested_mean
            for key, value in container.items():
                number = _number(value)
                if number is not None:
                    values[str(key)] = number
    return values


def compute_audio_noise_deltas(real: object, noise: object) -> dict[str, Any]:
    """Compute metric deltas for matched real-audio and noise artifacts.

    The returned ``audio_minus_noise`` values follow the paper convention
    ``real_audio - gaussian_noise``.  Inputs can be JSON/JSONL files, rows, or
    summary mappings.  Rows are matched by ``(task_id, sample_id, slot_index)``
    when IDs are present; aggregate summaries are matched by metric name.

    Returns
    -------
    dict
        ``matched`` rows, ``unmatched_real``, ``unmatched_noise``, and aggregate
        ``metrics`` with ``real``, ``noise``, and ``audio_minus_noise`` values.
    """

    real_rows = _rows_for_analysis(real)
    noise_rows = _rows_for_analysis(noise)
    real_by_key: dict[tuple[str, str, int | None], Mapping[str, Any]] = {}
    noise_by_key: dict[tuple[str, str, int | None], Mapping[str, Any]] = {}
    real_aggregate: dict[str, float] = {}
    noise_aggregate: dict[str, float] = {}
    for row in real_rows:
        key = _artifact_key(row)
        metrics = _metric_values(row)
        if key is None:
            real_aggregate.update(metrics)
        else:
            real_by_key[key] = row
    for row in noise_rows:
        key = _artifact_key(row)
        metrics = _metric_values(row)
        if key is None:
            noise_aggregate.update(metrics)
        else:
            noise_by_key[key] = row
    matched: list[dict[str, Any]] = []
    for key in real_by_key.keys() & noise_by_key.keys():
        real_metrics = _metric_values(real_by_key[key])
        noise_metrics = _metric_values(noise_by_key[key])
        metric_deltas = {
            metric: {
                "real": real_metrics[metric],
                "noise": noise_metrics[metric],
                "audio_minus_noise": real_metrics[metric] - noise_metrics[metric],
            }
            for metric in real_metrics.keys() & noise_metrics.keys()
        }
        matched.append({"key": key, "metrics": dict(sorted(metric_deltas.items()))})
    metrics: dict[str, dict[str, float]] = {}
    for metric in real_aggregate.keys() & noise_aggregate.keys():
        metrics[metric] = {
            "real": real_aggregate[metric],
            "noise": noise_aggregate[metric],
            "audio_minus_noise": real_aggregate[metric] - noise_aggregate[metric],
        }

    def _key_sort(key: tuple[str | None, str, int | None]) -> tuple[str, str, str]:
        return tuple("" if value is None else str(value) for value in key)

    return {
        "matched": sorted(matched, key=lambda row: _key_sort(row["key"])),
        "unmatched_real": sorted(set(real_by_key) - set(noise_by_key), key=_key_sort),
        "unmatched_noise": sorted(set(noise_by_key) - set(real_by_key), key=_key_sort),
        "metrics": dict(sorted(metrics.items())),
    }


# Descriptive aliases keep the helper convenient to discover from notebooks and
# shell wrappers without introducing a package import dependency.
validate_gaussian_noise_manifest = validate_manifest
summarize_prediction_outcomes = summarize_outcomes
compute_real_noise_deltas = compute_audio_noise_deltas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="Gaussian-noise JSON or JSONL manifest"
    )
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-slots", type=int)
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="validate declared metadata without opening local audio files",
    )
    parser.add_argument(
        "--skip-artifact-check",
        action="store_true",
        help="ignore artifact paths listed in the manifest",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit the complete report as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest validator CLI.

    Returns
    -------
    int
        ``0`` for a valid manifest, ``1`` for validation failures, and ``2``
        for invalid command-line usage or unreadable input.
    """

    args = _build_parser().parse_args(argv)
    report = validate_manifest(
        args.manifest,
        expected_tasks=args.expected_tasks,
        expected_samples=args.expected_samples,
        expected_slots=args.expected_slots,
        check_audio=not args.skip_audio_check,
        check_artifacts=not args.skip_artifact_check,
    )
    if args.json_output:
        print(
            json.dumps(
                {
                    "valid": report.valid,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "stats": report.stats,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for error in report.errors:
            print(f"FAIL: {error}", file=sys.stderr)
        for warning in report.warnings:
            print(f"WARN: {warning}", file=sys.stderr)
        if report.valid:
            print(
                "PASS: Gaussian-noise manifest is valid "
                f"(tasks={report.stats.get('tasks', 0)}, "
                f"samples={report.stats.get('samples', 0)}, "
                f"slots={report.stats.get('slots', 0)})"
            )
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
