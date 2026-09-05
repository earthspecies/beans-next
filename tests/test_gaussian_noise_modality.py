"""Integration expectations for the deterministic Gaussian-noise modality.

These tests use temporary WAVs instead of mocking the source-audio loader.  The
noise ablation must therefore keep the normal prompt/audio pathway intact while
substituting generated audio at the last possible point.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from beans_next.api.types import DatasetExample
from beans_next.audio.gaussian_noise import GaussianNoiseConfig
from beans_next.cli import _build_parser
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER
from beans_next.prompts.renderer import (
    AudioPlaceholderAlignmentError,
    AudioSlotSpec,
    PromptRenderer,
    PromptSpec,
    load_builtin_prompt_yaml,
)
from beans_next.runner.runner import _exclude_requested_sample_ids


def _noise_config(cache_dir: Path, *, global_seed: int = 1729) -> GaussianNoiseConfig:
    """Build the explicit reproducibility protocol used by these tests.

    Returns
    -------
    GaussianNoiseConfig
        Fixed −20 dBFS protocol configuration rooted at ``cache_dir``.
    """
    return GaussianNoiseConfig(
        cache_dir=cache_dir,
        dataset_revision="beans-next-test-revision",
        global_seed=global_seed,
        protocol_version="gaussian-noise-v1",
        rms_dbfs=-20.0,
    )


def _write_source(
    path: Path,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 0.75,
    channels: int = 1,
    value: float = 0.1,
) -> None:
    """Write a small, deliberately non-noise source recording."""
    n_samples = int(sample_rate * duration_seconds)
    waveform = np.full((n_samples, channels), value, dtype=np.float32)
    if channels == 1:
        waveform = waveform[:, 0]
    sf.write(path, waveform, sample_rate, subtype="PCM_16")


def _single_spec(content: str = f"{AUDIO_PLACEHOLDER}\nAnswer.") -> PromptSpec:
    return PromptSpec(
        prompt_id="test.gaussian.single.v1",
        message_templates=(("user", content),),
        audio_slots=(
            AudioSlotSpec(metadata_key="audio_path", payload_type="file_path"),
        ),
    )


def _read_audio(audio_path: str) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    return waveform, sample_rate


def test_single_audio_replaced_with_zero_mean_fixed_rms_noise_without_mutating_source(
    tmp_path: Path,
) -> None:
    """Single-audio replacement preserves source properties and the prompt."""
    source = tmp_path / "source.wav"
    _write_source(source, sample_rate=8_000, duration_seconds=1.25, channels=2)
    source_bytes = source.read_bytes()
    spec = _single_spec(f"system context\n{AUDIO_PLACEHOLDER}\nAnswer.")
    example = DatasetExample(
        sample_id="single-audio",
        task_id="beans_next_species",
        metadata={"audio_path": str(source)},
    )

    request = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "noise-cache"),
    ).render(example)

    assert request.messages[0].content == spec.message_templates[0][1]
    assert len(request.audio_inputs) == 1
    generated_path = Path(request.audio_inputs[0].data)
    assert generated_path != source
    assert generated_path.is_file()
    assert source.read_bytes() == source_bytes

    noise, sample_rate = _read_audio(str(generated_path))
    source_audio, source_sample_rate = _read_audio(str(source))
    assert sample_rate == source_sample_rate == 8_000
    assert noise.shape == source_audio.shape == (10_000, 2)
    assert abs(float(noise.mean())) < 0.01
    rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    assert 10 ** ((-20.0 - 0.35) / 20) <= rms <= 10 ** ((-20.0 + 0.35) / 20)
    assert float(np.max(np.abs(noise))) <= 1.0


def test_gaussian_noise_is_repeatable_and_seeded_by_protocol_identity(
    tmp_path: Path,
) -> None:
    """Repeated renders match byte-for-byte; changing the global seed changes noise."""
    source = tmp_path / "source.wav"
    _write_source(source, sample_rate=16_000, duration_seconds=0.5)
    example = DatasetExample(
        sample_id="repeatable",
        metadata={"audio_path": str(source)},
    )
    spec = _single_spec()

    first = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "cache", global_seed=11),
    ).render(example)
    second = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "cache", global_seed=11),
    ).render(example)
    changed = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "other-cache", global_seed=12),
    ).render(example)

    first_bytes = Path(first.audio_inputs[0].data).read_bytes()
    second_bytes = Path(second.audio_inputs[0].data).read_bytes()
    changed_bytes = Path(changed.audio_inputs[0].data).read_bytes()
    assert first.audio_inputs[0].data == second.audio_inputs[0].data
    assert first_bytes == second_bytes
    assert first_bytes != changed_bytes


def test_multi_audio_replaces_every_slot_left_to_right_and_preserves_properties(
    tmp_path: Path,
) -> None:
    """Every list item receives independent noise in original placeholder order."""
    paths = [
        tmp_path / "support-a.wav",
        tmp_path / "support-b.wav",
        tmp_path / "query.wav",
    ]
    properties = [(8_000, 0.4, 1), (16_000, 0.6, 2), (11_025, 0.8, 1)]
    for path, (sample_rate, duration, channels) in zip(paths, properties, strict=True):
        _write_source(
            path,
            sample_rate=sample_rate,
            duration_seconds=duration,
            channels=channels,
        )
    conversation = (
        f"Support one: {AUDIO_PLACEHOLDER}\n"
        f"Support two: {AUDIO_PLACEHOLDER}\n"
        f"Query: {AUDIO_PLACEHOLDER}\nChoose A or B."
    )
    spec = PromptSpec(
        prompt_id="test.gaussian.multi.v1",
        message_templates=(("user", conversation),),
        audio_slots=(
            AudioSlotSpec(
                list_metadata_key="audio_paths",
                payload_type="file_path",
            ),
        ),
    )
    example = DatasetExample(
        sample_id="multi-audio",
        metadata={"audio_paths": [str(path) for path in paths]},
    )

    request = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "noise-cache"),
    ).render(example)

    assert request.messages[0].content == conversation
    assert [audio.payload_type for audio in request.audio_inputs] == [
        "file_path",
        "file_path",
        "file_path",
    ]
    assert [Path(audio.data) for audio in request.audio_inputs] != paths
    for generated, source, (expected_rate, _duration, channels) in zip(
        request.audio_inputs, paths, properties, strict=True
    ):
        noise, sample_rate = _read_audio(generated.data)
        source_audio, source_rate = _read_audio(str(source))
        assert sample_rate == source_rate == expected_rate
        assert noise.shape == source_audio.shape
        assert noise.shape[1] == channels
        assert abs(float(noise.mean())) < 0.02


def test_captioning_keeps_prompt_and_decoding_configuration_unchanged(
    tmp_path: Path,
) -> None:
    """Captioning changes only the audio payload, never its normal prompt contract."""
    source = tmp_path / "caption.wav"
    _write_source(source, sample_rate=16_000, duration_seconds=0.7)
    example = DatasetExample(sample_id="caption", metadata={"audio_path": str(source)})
    spec = load_builtin_prompt_yaml("captioning_bioacoustic_v1.yaml")
    normal = PromptRenderer(spec).render(example)
    noisy = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "noise-cache"),
    ).render(example)

    assert noisy.messages == normal.messages
    assert noisy.generation_config == normal.generation_config
    assert len(noisy.audio_inputs) == len(normal.audio_inputs) == 1
    assert noisy.audio_inputs[0].data != normal.audio_inputs[0].data


def test_tier4_support_and_query_audio_keep_placeholder_order(tmp_path: Path) -> None:
    """Tier-4 support recordings precede the query recording on the wire."""
    support_a = tmp_path / "support-a.wav"
    support_b = tmp_path / "support-b.wav"
    query = tmp_path / "query.wav"
    _write_source(support_a, value=0.11)
    _write_source(support_b, value=0.22)
    _write_source(query, value=0.33)
    conversation = (
        f"Classify the query using the support examples.\n"
        f"Support A {AUDIO_PLACEHOLDER}\n"
        f"Support B {AUDIO_PLACEHOLDER}\n"
        f"Query {AUDIO_PLACEHOLDER}\nAnswer A or B."
    )
    spec = PromptSpec(
        prompt_id="test.gaussian.tier4.v1",
        message_templates=(("user", "{{ metadata.conversation }}"),),
        audio_slots=(
            AudioSlotSpec(
                list_metadata_key="support_audio_paths",
                payload_type="file_path",
            ),
            AudioSlotSpec(metadata_key="query_audio_path", payload_type="file_path"),
        ),
    )
    example = DatasetExample(
        sample_id="tier4",
        task_id="beans_next_tier4",
        metadata={
            "conversation": conversation,
            "support_audio_paths": [str(support_a), str(support_b)],
            "query_audio_path": str(query),
        },
    )

    request = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "noise-cache"),
    ).render(example)

    assert request.messages[0].content == conversation
    assert len(request.audio_inputs) == 3
    # The first two slots are support audio and the final slot is the query;
    # each generated file must be a valid replacement in that exact order.
    for generated, source in zip(
        request.audio_inputs,
        (support_a, support_b, query),
        strict=True,
    ):
        assert Path(generated.data).is_file()
        assert Path(generated.data) != source
        assert _read_audio(generated.data)[0].shape == _read_audio(str(source))[0].shape


@pytest.mark.parametrize(
    "content",
    [f"{AUDIO_PLACEHOLDER}\nOnly one", "No audio placeholder"],
)
def test_gaussian_noise_does_not_bypass_placeholder_alignment(
    tmp_path: Path,
    content: str,
) -> None:
    """Noise mode retains the same safety check as normal audio mode."""
    source = tmp_path / "alignment.wav"
    _write_source(source)
    example = DatasetExample(
        sample_id="misaligned",
        metadata={"audio_path": str(source)},
    )
    spec = PromptSpec(
        prompt_id="test.gaussian.alignment.v1",
        message_templates=(("user", content),),
        audio_slots=(
            AudioSlotSpec(metadata_key="audio_path", payload_type="file_path"),
            AudioSlotSpec(metadata_key="audio_path", payload_type="file_path"),
        ),
    )

    with pytest.raises(AudioPlaceholderAlignmentError) as exc_info:
        PromptRenderer(
            spec,
            modality_mode="gaussian-noise",
            gaussian_noise_config=_noise_config(tmp_path / "noise-cache"),
        ).render(example)
    assert exc_info.value.sample_id == "misaligned"


def test_noise_outputs_are_independent_of_example_worker_order(tmp_path: Path) -> None:
    """Per-sample identities make serial and concurrent rendering equivalent."""
    sources = []
    examples = []
    for index in range(6):
        source = tmp_path / f"source-{index}.wav"
        _write_source(
            source,
            duration_seconds=0.25 + index / 100,
            value=0.05 * (index + 1),
        )
        sources.append(source)
        examples.append(
            DatasetExample(
                sample_id=f"worker-{index}",
                metadata={"audio_path": str(source)},
            )
        )
    spec = _single_spec()

    serial_renderer = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "serial-cache"),
    )
    serial = {
        example.sample_id: Path(
            serial_renderer.render(example).audio_inputs[0].data
        ).read_bytes()
        for example in examples
    }
    worker_renderer = PromptRenderer(
        spec,
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "worker-cache"),
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        rendered = list(executor.map(worker_renderer.render, reversed(examples)))
    concurrent = {
        request.sample_id: Path(request.audio_inputs[0].data).read_bytes()
        for request in rendered
    }
    assert concurrent == serial


def test_cli_accepts_gaussian_noise_modality_choice() -> None:
    """The command-line mode must be selectable without changing audio loading."""
    args = _build_parser().parse_args(
        [
            "run",
            "--predict-url",
            "http://localhost:1/predict",
            "--modality-mode",
            "gaussian-noise",
        ]
    )
    assert args.modality_mode == "gaussian-noise"


def test_cli_excludes_only_exact_requested_sample_ids() -> None:
    """Documented model-incompatible rows can be removed without prefix matches."""
    args = _build_parser().parse_args(
        [
            "run",
            "--predict-url",
            "http://localhost:1/predict",
            "--exclude-sample-id",
            "sample:2",
            "--exclude-sample-id",
            "sample:4",
        ]
    )
    examples = [DatasetExample(sample_id=f"sample:{index}") for index in range(5)]
    filtered = _exclude_requested_sample_ids(examples, args=args)
    assert [example.sample_id for example in filtered] == [
        "sample:0",
        "sample:1",
        "sample:3",
    ]


def test_manifest_records_protocol_commit_checksums_and_sorted_slots(
    tmp_path: Path,
) -> None:
    """Run manifests contain deterministic, resume-safe protocol provenance."""
    sources = [tmp_path / "b.wav", tmp_path / "a.wav"]
    for source in sources:
        _write_source(source)
    renderer = PromptRenderer(
        _single_spec(),
        modality_mode="gaussian-noise",
        gaussian_noise_config=_noise_config(tmp_path / "cache"),
    )
    for sample_id, source in (("b", sources[0]), ("a", sources[1])):
        renderer.render(
            DatasetExample(sample_id=sample_id, metadata={"audio_path": str(source)})
        )
    manifest_path = tmp_path / "manifest.json"
    renderer.write_gaussian_noise_manifest(manifest_path, code_commit="deadbeef")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "beans_next.gaussian_noise_manifest.v1"
    assert payload["dataset_revision"] == "beans-next-test-revision"
    assert payload["global_seed"] == 1729
    assert payload["rms_dbfs"] == -20.0
    assert payload["code_commit"] == "deadbeef"
    assert [row["sample_id"] for row in payload["records"]] == ["a", "b"]
    assert all(len(row["source_sha256"]) == 64 for row in payload["records"])
    assert all(len(row["audio_sha256"]) == 64 for row in payload["records"])
