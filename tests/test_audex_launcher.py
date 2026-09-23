"""Contract + unit tests for the Audex adapter launcher (``predictions_v1``).

Mirrors the stub-mode conformance pattern in ``tests/test_launcher_stubs.py``
and adds Audex-specific unit coverage: ``file_path`` -> ``file://`` mapping,
``<think>``/codec-token stripping, and ``AUDEX_EXTRA_BODY_JSON`` passthrough.
"""

from __future__ import annotations

import base64
import importlib
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_launcher_conformance(*, app: str, env: Mapping[str, str]) -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    cmd = [
        sys.executable,
        "scripts/with_uvicorn.py",
        "--app",
        app,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--cwd",
        ".",
        "--ready-timeout-s",
        "15",
        "--poll-interval-s",
        "0.05",
        *[x for kv in env.items() for x in ("--env", f"{kv[0]}={kv[1]}")],
        "--",
        sys.executable,
        "scripts/check_launcher.py",
        base_url,
    ]

    merged_env = dict(os.environ)
    merged_env.update(dict(env))

    res = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        env=merged_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        "Launcher conformance failed.\n"
        f"stdout:\n{res.stdout}\n"
        f"stderr:\n{res.stderr}\n"
        f"app={app!r} port={port}"
    )


def _load_module() -> ModuleType:
    return importlib.import_module("examples.servers.audex.serve")


def test_strip_think_and_codec_tokens_removes_think_span() -> None:
    module = _load_module()

    out = module._strip_think_and_codec_tokens(
        "<think>internal reasoning that should vanish</think>final answer"
    )

    assert out == "final answer"


def test_strip_think_and_codec_tokens_removes_codec_placeholders() -> None:
    module = _load_module()

    out = module._strip_think_and_codec_tokens(
        "dog barking<|audio_codec_12|> <audio_pad> at night"
    )

    assert "audio_codec" not in out
    assert "audio_pad" not in out
    assert "dog barking" in out
    assert "at night" in out


def test_strip_think_and_codec_tokens_leaves_clean_text_untouched() -> None:
    module = _load_module()

    out = module._strip_think_and_codec_tokens("a dog is barking")

    assert out == "a dog is barking"


def test_audio_url_content_item_maps_file_path_to_file_uri(tmp_path: Path) -> None:
    module = _load_module()

    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFF....WAVEfmt ")

    audio_input = module.HttpAudioInput(
        payload_type="file_path",
        data=str(wav_path),
        sample_rate=16000,
    )

    item = module._audio_url_content_item(audio_input, None)

    assert item["type"] == "audio_url"
    assert item["audio_url"]["url"] == wav_path.resolve().as_uri()
    assert item["audio_url"]["url"].startswith("file://")


def test_audio_url_content_item_falls_back_to_data_uri_for_base64() -> None:
    module = _load_module()

    audio_input = module.HttpAudioInput(
        payload_type="base64_wav",
        data="AAAA",
        sample_rate=16000,
    )

    item = module._audio_url_content_item(audio_input, None)

    assert item["audio_url"]["url"] == "data:audio/wav;base64,AAAA"


def _write_wav(path: Path, *, seconds: float, sample_rate: int = 16000) -> None:
    import numpy as np
    import soundfile as sf

    n = int(seconds * sample_rate)
    samples = np.linspace(-1.0, 1.0, n, dtype="float32")
    sf.write(str(path), samples, sample_rate, format="WAV")


def test_trim_wav_file_truncates_over_cap_clip(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    monkeypatch.setenv("AUDEX_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)

    out = module._trim_wav_file(src, 2)

    import soundfile as sf

    info = sf.info(str(out))
    assert out != src.resolve()
    assert info.frames / info.samplerate == 2.0


def test_trim_wav_file_passes_through_under_cap_clip(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    monkeypatch.setenv("AUDEX_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))

    src = tmp_path / "short.wav"
    _write_wav(src, seconds=2.0)

    out = module._trim_wav_file(src, 10)

    assert out == src.resolve()


def test_trim_wav_file_reuses_cache(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    monkeypatch.setenv("AUDEX_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)

    first = module._trim_wav_file(src, 2)
    mtime = first.stat().st_mtime_ns
    second = module._trim_wav_file(src, 2)

    assert second == first
    assert second.stat().st_mtime_ns == mtime


def test_audio_url_content_item_no_cap_is_noop(tmp_path: Path) -> None:
    module = _load_module()

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)
    audio_input = module.HttpAudioInput(
        payload_type="file_path", data=str(src), sample_rate=16000
    )

    item = module._audio_url_content_item(audio_input, None)

    assert item["audio_url"]["url"] == src.resolve().as_uri()


def test_trim_base64_wav_truncates_over_cap_clip(tmp_path: Path) -> None:
    module = _load_module()

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)
    b64 = base64.b64encode(src.read_bytes()).decode("ascii")

    trimmed = module._trim_base64_wav(b64, 2)

    import io

    import soundfile as sf

    info = sf.info(io.BytesIO(base64.b64decode(trimmed)))
    assert trimmed != b64
    assert info.frames / info.samplerate == 2.0


def test_trim_base64_wav_passes_through_under_cap_clip(tmp_path: Path) -> None:
    module = _load_module()

    src = tmp_path / "short.wav"
    _write_wav(src, seconds=2.0)
    b64 = base64.b64encode(src.read_bytes()).decode("ascii")

    trimmed = module._trim_base64_wav(b64, 10)

    assert trimmed == b64


def test_audex_processes_proxy_batches_concurrently(
    monkeypatch: "__import__('pytest').MonkeyPatch",
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "CFG",
        replace(module.CFG, stub=False, max_concurrency=2),
    )

    seen: list[str] = []

    def fake_item_response(sample_id: str, raw: dict[str, Any]) -> object:
        seen.append(sample_id)
        return module.PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[raw["value"]],
            error=None,
        )

    monkeypatch.setattr(module, "_item_response_or_error", fake_item_response)

    responses = module._process_request_batch(
        [
            {"sample_id": "b", "value": "two"},
            {"sample_id": "a", "value": "one"},
        ]
    )

    assert seen == ["b", "a"]
    assert [r.sample_id for r in responses] == ["b", "a"]
    assert [r.predictions[0] for r in responses] == ["two", "one"]


def test_audex_extra_body_json_merged_into_upstream_request(
    monkeypatch: "__import__('pytest').MonkeyPatch",
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "CFG",
        replace(
            module.CFG,
            stub=False,
            upstream_base_url="http://upstream.invalid",
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0.7,
                "top_p": 0.9,
                "logit_bias": {"29": -100, "30": -100, "31": -100},
            },
        ),
    )

    captured: dict[str, Any] = {}

    class _FakeResponse:
        """Stand-in for an `httpx.Response` returned by the upstream chat call."""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "a dog is barking"}}]}

    class _FakeHttp:
        """Stand-in for the module-level HTTP client used to reach vLLM."""

        def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
            captured["url"] = url
            captured["body"] = json
            return _FakeResponse()

    monkeypatch.setattr(module, "HTTP", _FakeHttp())

    text = module._call_upstream_chat_completion(
        messages=[{"role": "user", "content": "<AudioHere> what is this?"}],
        audio_inputs=[],
        generation_config=module.HttpGenerationConfig(max_tokens=32, temperature=0.0),
    )

    assert text == "a dog is barking"
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["top_p"] == 0.9
    # Extra body's temperature (vendor sampling arm) overrides the
    # per-request generation_config value applied earlier in the same call.
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["logit_bias"] == {"29": -100, "30": -100, "31": -100}
