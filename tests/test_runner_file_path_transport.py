"""Tests for optional shared-filesystem audio transport."""

from __future__ import annotations

from pathlib import Path

from beans_next.api.types import AudioInput, ChatMessage, ModelRequest
from beans_next.runner.runner import model_request_to_wire_item


def _request(path: Path) -> ModelRequest:
    return ModelRequest(
        sample_id="sample",
        messages=[ChatMessage(role="user", content="<Audio><AudioHere></Audio>")],
        audio_inputs=[
            AudioInput(payload_type="file_path", data=str(path), sample_rate=16_000)
        ],
    )


def test_file_path_is_preserved_for_shared_filesystem(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")

    item = model_request_to_wire_item(_request(wav), preserve_file_paths=True)

    assert item.audio_inputs[0].payload_type == "file_path"
    assert item.audio_inputs[0].data == str(wav)


def test_file_path_defaults_to_base64_for_remote_server(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")

    item = model_request_to_wire_item(_request(wav))

    assert item.audio_inputs[0].payload_type == "base64_wav"
    assert item.audio_inputs[0].data == "UklGRg=="
