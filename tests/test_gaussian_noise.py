"""Focused tests for deterministic Gaussian-noise materialization."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from beans_next.audio.gaussian_noise import GaussianNoiseConfig, materialize


def test_materialize_is_deterministic_and_preserves_shape(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    source = np.zeros((321, 2), dtype=np.float32)
    sf.write(source_path, source, 22050, subtype="PCM_16")
    config = GaussianNoiseConfig(
        dataset_revision="beans-v2",
        global_seed=17,
        cache_dir=tmp_path / "cache",
    )

    first = materialize(source_path, "recording-abc", 1, config)
    second = materialize(source_path, "recording-abc", 1, config)
    assert first == second
    assert first.path.read_bytes() == second.path.read_bytes()
    generated, rate = sf.read(first.path, always_2d=True, dtype="float64")
    assert generated.shape == source.shape
    assert rate == 22050
    assert abs(float(np.mean(generated))) < 1e-7
    assert np.isclose(np.sqrt(np.mean(generated**2)), 0.1, rtol=2e-5)
    assert np.max(np.abs(generated)) < 1.0

    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["frames"] == 321
    assert metadata["sample_rate"] == 22050
    assert metadata["channels"] == 2
    assert metadata["sha256"] == first.sha256


def test_seed_inputs_and_slots_are_distinct(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    sf.write(source_path, np.zeros(20, dtype=np.float32), 8000)
    config = GaussianNoiseConfig(dataset_revision="r1", cache_dir=tmp_path / "cache")
    slot_zero = materialize(source_path, "same", 0, config)
    slot_one = materialize(source_path, "same", 1, config)
    other_revision = materialize(
        source_path,
        "same",
        0,
        GaussianNoiseConfig(dataset_revision="r2", cache_dir=tmp_path / "cache"),
    )
    assert slot_zero.cache_key != slot_one.cache_key
    assert slot_zero.cache_key != other_revision.cache_key
    assert slot_zero.path.read_bytes() != slot_one.path.read_bytes()


def test_sensitivity_levels_share_seed_but_not_cache_entry(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    sf.write(source_path, np.zeros(2_000, dtype=np.float32), 8_000)
    common = {"dataset_revision": "r1", "cache_dir": tmp_path / "cache"}
    quiet = materialize(
        source_path, "same", 0, GaussianNoiseConfig(**common, rms_dbfs=-30)
    )
    loud = materialize(
        source_path, "same", 0, GaussianNoiseConfig(**common, rms_dbfs=-10)
    )

    assert quiet.seed == loud.seed
    assert quiet.seed_sha256 == loud.seed_sha256
    assert quiet.cache_key != loud.cache_key
    assert quiet.path != loud.path
    quiet_audio, _ = sf.read(quiet.path, dtype="float64")
    loud_audio, _ = sf.read(loud.path, dtype="float64")
    assert np.isclose(np.sqrt(np.mean(quiet_audio**2)), 10 ** (-30 / 20))
    assert np.isclose(np.sqrt(np.mean(loud_audio**2)), 10 ** (-10 / 20))
