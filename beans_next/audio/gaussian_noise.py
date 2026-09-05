"""Deterministic, cacheable Gaussian-noise audio materialization.

The primitive in this module deliberately has no dependency on a model, runner,
or dataset implementation.  A caller supplies the source recording identity
and slot number; the source file supplies only the audio shape (frame count,
sample rate, and channel count).  Consequently all models and worker orders
receive the same generated file for a given protocol key.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

DEFAULT_PROTOCOL_VERSION = "beans-next.gaussian-noise.v1"
_DEFAULT_CACHE_ENV = "BEANS_NEXT_GAUSSIAN_NOISE_CACHE"
_DEFAULT_GPFS_CACHE = Path("/gpfs/scratch/acw777/beans-next/gaussian-noise")
_DEFAULT_LOCAL_CACHE = Path.home() / ".cache" / "beans-next" / "gaussian-noise"


def _default_cache_dir() -> Path:
    """Return the cluster cache when available, otherwise a local cache path.

    Returns
    -------
    pathlib.Path
        Default shared or per-user cache directory.
    """

    configured = os.environ.get(_DEFAULT_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    if _DEFAULT_GPFS_CACHE.parent.parent.exists():
        return _DEFAULT_GPFS_CACHE
    return _DEFAULT_LOCAL_CACHE


@dataclass(frozen=True, slots=True)
class GaussianNoiseConfig:
    """Configuration for deterministic Gaussian-noise generation.

    ``rms_dbfs`` is expressed relative to full-scale amplitude 1.0.  The
    default is the BEANS Gaussian-noise protocol value of -20 dBFS.  Set
    ``cache_dir`` explicitly for a shared scratch cache on a cluster.
    """

    dataset_revision: str = "unknown"
    global_seed: int | str = 0
    protocol_version: str = DEFAULT_PROTOCOL_VERSION
    rms_dbfs: float = -20.0
    cache_dir: Path = field(default_factory=_default_cache_dir)

    def __post_init__(self) -> None:
        if not str(self.dataset_revision):
            raise ValueError("dataset_revision must not be empty")
        if not str(self.protocol_version):
            raise ValueError("protocol_version must not be empty")
        if not np.isfinite(float(self.rms_dbfs)) or float(self.rms_dbfs) >= 0:
            raise ValueError("rms_dbfs must be finite and below 0 dBFS")
        try:
            hash(self.global_seed)
        except TypeError as exc:
            raise TypeError("global_seed must be hashable/canonicalizable") from exc
        object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())

    def materialize(
        self,
        source_path: str | os.PathLike[str],
        source_identity: str,
        slot_index: int,
    ) -> GaussianNoiseRecord:
        """Generate or retrieve one deterministic noise recording.

        Returns
        -------
        GaussianNoiseRecord
            Validated cache record for the generated audio.
        """

        return materialize(source_path, source_identity, slot_index, config=self)


@dataclass(frozen=True, slots=True)
class GaussianNoiseRecord:
    """Immutable description of a generated noise recording and its sidecar."""

    path: Path
    metadata_path: Path
    seed: int
    seed_sha256: str
    cache_key: str
    sha256: str
    source_sha256: str
    source_identity: str
    slot_index: int
    dataset_revision: str
    protocol_version: str
    rms_dbfs: float
    mean: float
    rms: float
    peak: float
    frames: int
    sample_rate: int
    channels: int

    @property
    def audio_path(self) -> Path:
        """Alias useful to request/transport integrations."""

        return self.path

    @property
    def sidecar_path(self) -> Path:
        """Alias for the immutable JSON metadata sidecar."""

        return self.metadata_path

    @property
    def checksum(self) -> str:
        """SHA-256 checksum of the generated audio file."""

        return self.sha256

    @property
    def sample_rate_hz(self) -> int:
        return self.sample_rate

    @property
    def channel_count(self) -> int:
        return self.channels


def _canonical_seed_material(
    config: GaussianNoiseConfig, source_identity: str, slot_index: int
) -> bytes:
    """Serialize seed inputs canonically so equivalent workers agree exactly.

    Returns
    -------
    bytes
        Canonical UTF-8 JSON used as the hash input.

    Raises
    ------
    TypeError
        If the global seed cannot be represented as canonical JSON.
    """

    material: dict[str, Any] = {
        "dataset_revision": str(config.dataset_revision),
        "source_identity": str(source_identity),
        "slot_index": int(slot_index),
        "protocol_version": str(config.protocol_version),
        "global_seed": config.global_seed,
    }
    try:
        encoded = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("global_seed must be JSON-canonicalizable") from exc
    return encoded


def derive_seed(
    dataset_revision: str,
    source_identity: str,
    slot_index: int,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    global_seed: int | str = 0,
) -> int:
    """Derive the stable NumPy seed from the complete protocol key.

    Returns
    -------
    int
        Unsigned 64-bit seed derived from SHA-256.
    """

    config = GaussianNoiseConfig(
        dataset_revision=dataset_revision,
        global_seed=global_seed,
        protocol_version=protocol_version,
    )
    return int.from_bytes(
        hashlib.sha256(
            _canonical_seed_material(config, source_identity, slot_index)
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _seed_digest(
    config: GaussianNoiseConfig, source_identity: str, slot_index: int
) -> str:
    return hashlib.sha256(
        _canonical_seed_material(config, source_identity, slot_index)
    ).hexdigest()


def _cache_key(seed_digest: str, rms_dbfs: float) -> str:
    """Identify one rendered amplitude without changing the protocol seed."""

    material = json.dumps(
        {"seed_sha256": seed_digest, "rms_dbfs": float(rms_dbfs)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_publish(temp_path: Path, destination: Path) -> bool:
    """Publish a temporary file without replacing an existing cache entry.

    Returns
    -------
    bool
        ``True`` when this worker published the entry, otherwise ``False``.
    """

    try:
        os.link(temp_path, destination)
    except FileExistsError:
        temp_path.unlink(missing_ok=True)
        return False
    finally:
        # A successful link leaves the temporary directory entry behind.
        temp_path.unlink(missing_ok=True)
    return True


def _write_temp_bytes(directory: Path, suffix: str, payload: bytes) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=".gaussian-noise-", suffix=suffix, dir=directory
    )
    temp_path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _write_temp_audio(directory: Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write a byte-stable IEEE-float WAV without mutable encoder metadata.

    Returns
    -------
    pathlib.Path
        Temporary file containing the deterministic WAV payload.

    Raises
    ------
    ValueError
        If the recording is too large for the RIFF 32-bit size field.
    """

    frames, channels = audio.shape
    sample_width = 4
    samples = np.asarray(audio, dtype="<f4", order="C").tobytes(order="C")
    if len(samples) > 0xFFFFFFFF - 48:
        raise ValueError("Gaussian-noise WAV exceeds the RIFF 32-bit size limit")
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        3,  # WAVE_FORMAT_IEEE_FLOAT
        channels,
        sample_rate,
        sample_rate * channels * sample_width,
        channels * sample_width,
        sample_width * 8,
    )
    fact_chunk = struct.pack("<4sII", b"fact", 4, frames)
    data_chunk = struct.pack("<4sI", b"data", len(samples)) + samples
    payload = (
        struct.pack("<4sI4s", b"RIFF", 48 + len(samples), b"WAVE")
        + fmt_chunk
        + fact_chunk
        + data_chunk
    )
    return _write_temp_bytes(directory, ".wav", payload)


def _noise_audio(
    frames: int, channels: int, config: GaussianNoiseConfig, seed: int
) -> np.ndarray:
    """Create centered Gaussian samples at the requested RMS without clipping.

    Returns
    -------
    numpy.ndarray
        Two-dimensional floating-point waveform shaped frames by channels.

    Raises
    ------
    RuntimeError
        If the random draw has zero variance.
    """

    rng = np.random.default_rng(seed)
    audio = rng.standard_normal((frames, channels), dtype=np.float64)
    audio -= np.mean(audio, axis=0, keepdims=True, dtype=np.float64)
    target_rms = float(10.0 ** (float(config.rms_dbfs) / 20.0))
    current_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if current_rms == 0.0:
        raise RuntimeError("Gaussian generator produced zero variance")
    audio *= target_rms / current_rms

    # Gaussian distributions have unbounded tails.  The primary -20 dBFS
    # protocol must stay below full scale; fail rather than silently changing
    # its requested RMS.  The deliberately louder -10 dBFS sensitivity arm is
    # stored as float WAV and may contain samples outside [-1, 1].
    peak = float(np.max(np.abs(audio), initial=0.0))
    if float(config.rms_dbfs) <= -20.0 and peak >= 1.0:
        raise RuntimeError(
            "Gaussian waveform exceeds full scale at the required non-clipping level"
        )
    return audio


def _record_from_metadata(metadata: dict[str, Any]) -> GaussianNoiseRecord:
    return GaussianNoiseRecord(
        path=Path(metadata["path"]),
        metadata_path=Path(metadata["metadata_path"]),
        seed=int(metadata["seed"]),
        seed_sha256=str(metadata["seed_sha256"]),
        cache_key=str(metadata["cache_key"]),
        sha256=str(metadata["sha256"]),
        source_sha256=str(metadata["source_sha256"]),
        source_identity=str(metadata["source_identity"]),
        slot_index=int(metadata["slot_index"]),
        dataset_revision=str(metadata["dataset_revision"]),
        protocol_version=str(metadata["protocol_version"]),
        rms_dbfs=float(metadata["rms_dbfs"]),
        mean=float(metadata["mean"]),
        rms=float(metadata["rms"]),
        peak=float(metadata["peak"]),
        frames=int(metadata["frames"]),
        sample_rate=int(metadata["sample_rate"]),
        channels=int(metadata["channels"]),
    )


def _cached_record(
    audio_path: Path,
    metadata_path: Path,
    config: GaussianNoiseConfig,
    source_identity: str,
    slot_index: int,
    seed_digest: str,
    cache_key: str,
) -> GaussianNoiseRecord | None:
    if not audio_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid Gaussian-noise cache sidecar: {metadata_path}"
        ) from exc
    expected = {
        "schema_version": "beans_next.gaussian_noise.v1",
        "cache_key": cache_key,
        "dataset_revision": str(config.dataset_revision),
        "source_identity": str(source_identity),
        "slot_index": int(slot_index),
        "protocol_version": str(config.protocol_version),
        "rms_dbfs_requested": float(config.rms_dbfs),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Gaussian-noise cache metadata mismatch for {audio_path}"
            )
    if metadata.get("path") != str(audio_path) or metadata.get("metadata_path") != str(
        metadata_path
    ):
        raise RuntimeError(
            f"Gaussian-noise cache path metadata mismatch for {audio_path}"
        )
    actual_sha = _sha256(audio_path)
    if metadata.get("sha256") != actual_sha:
        raise RuntimeError(f"Gaussian-noise cache checksum mismatch for {audio_path}")
    source_path = Path(metadata["source_path"])
    if metadata.get("source_sha256") != _sha256(source_path):
        raise RuntimeError(f"Gaussian-noise source checksum mismatch for {audio_path}")
    info = sf.info(audio_path)
    if (
        int(info.frames) != int(metadata["frames"])
        or int(info.samplerate) != int(metadata["sample_rate"])
        or int(info.channels) != int(metadata["channels"])
    ):
        raise RuntimeError(
            f"Gaussian-noise cache audio properties mismatch for {audio_path}"
        )
    return _record_from_metadata(metadata)


def materialize(
    source_path: str | os.PathLike[str],
    source_identity: str,
    slot_index: int,
    config: GaussianNoiseConfig | None = None,
) -> GaussianNoiseRecord:
    """Generate or retrieve deterministic white Gaussian noise for one source.

    The source recording is never modified.  Its frame count, sample rate, and
    channel count are copied to the generated WAV.  The WAV and immutable JSON
    sidecar are both published atomically into ``config.cache_dir``.

    Returns
    -------
    GaussianNoiseRecord
        Validated immutable cache record.

    Raises
    ------
    FileNotFoundError
        If the source audio does not exist.
    ValueError
        If the slot or source audio properties are invalid.
    RuntimeError
        If an existing cache entry is invalid or publication fails.
    """

    config = config or GaussianNoiseConfig()
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if int(slot_index) < 0:
        raise ValueError("slot_index must be non-negative")
    slot_index = int(slot_index)
    source_identity = str(source_identity)
    info = sf.info(source)
    frames, sample_rate, channels = (
        int(info.frames),
        int(info.samplerate),
        int(info.channels),
    )
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise ValueError(
            "source audio must contain frames, a sample rate, and channels"
        )

    seed_digest = _seed_digest(config, source_identity, slot_index)
    cache_key = _cache_key(seed_digest, config.rms_dbfs)
    seed = int.from_bytes(
        bytes.fromhex(seed_digest[:16]), byteorder="big", signed=False
    )
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / f"{cache_key}.wav"
    metadata_path = cache_dir / f"{cache_key}.wav.json"
    cached = _cached_record(
        audio_path,
        metadata_path,
        config,
        source_identity,
        slot_index,
        seed_digest,
        cache_key,
    )
    if cached is not None:
        return cached

    audio = _noise_audio(frames, channels, config, seed)
    actual_mean = float(np.mean(audio, dtype=np.float64))
    actual_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    actual_peak = float(np.max(np.abs(audio), initial=0.0))
    temp_audio = _write_temp_audio(cache_dir, audio, sample_rate)
    _atomic_publish(temp_audio, audio_path)
    checksum = _sha256(audio_path)
    source_checksum = _sha256(source)

    metadata: dict[str, Any] = {
        "schema_version": "beans_next.gaussian_noise.v1",
        "protocol": "zero-mean-white-Gaussian-noise",
        "protocol_version": str(config.protocol_version),
        "dataset_revision": str(config.dataset_revision),
        "global_seed": config.global_seed,
        "source_identity": source_identity,
        "source_path": str(source),
        "source_sha256": source_checksum,
        "slot_index": slot_index,
        "cache_key": cache_key,
        "seed": seed,
        "seed_sha256": seed_digest,
        "rms_dbfs_requested": float(config.rms_dbfs),
        "rms_dbfs": float(20.0 * np.log10(actual_rms)) if actual_rms else -np.inf,
        "mean": actual_mean,
        "rms": actual_rms,
        "peak": actual_peak,
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": float(frames / sample_rate),
        "format": "WAV",
        "subtype": "FLOAT",
        "sha256": checksum,
        "audio_sha256": checksum,
        "path": str(audio_path),
        "metadata_path": str(metadata_path),
    }
    metadata_bytes = (
        json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    temp_metadata = _write_temp_bytes(cache_dir, ".json", metadata_bytes)
    _atomic_publish(temp_metadata, metadata_path)

    # A concurrent worker may have completed the sidecar between our initial
    # cache check and publication.  Always validate the immutable final pair.
    cached = _cached_record(
        audio_path,
        metadata_path,
        config,
        source_identity,
        slot_index,
        seed_digest,
        cache_key,
    )
    if cached is None:  # pragma: no cover - defensive against a broken filesystem
        raise RuntimeError(
            f"Unable to publish Gaussian-noise cache entry: {audio_path}"
        )
    return cached
