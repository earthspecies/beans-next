"""Contract + unit tests for the Gemma 4 adapter launcher (``predictions_v1``).

Mirrors ``tests/test_audex_launcher.py``'s stub-mode conformance pattern and
adds Gemma4-specific unit coverage: ``file_path`` -> ``file://`` mapping,
``<think>`` stripping (no codec-token stripping — Gemma 4 has no
audio-generation path), ``GEMMA4_EXTRA_BODY_JSON`` passthrough, the
``min(protocol_cap, GEMMA4_MAX_AUDIO_SECONDS)`` composition (including the
30s clamp applying with **no** protocol cap present, which has no Audex
analogue), and audio-after-text prompt assembly.
"""

from __future__ import annotations

import base64
import importlib
import math
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


def test_gemma4_stub_mode_conforms() -> None:
    _run_launcher_conformance(
        app="examples.servers.gemma4.serve:app",
        env={"GEMMA4_ADAPTER_STUB": "1"},
    )


def _load_module() -> ModuleType:
    return importlib.import_module("examples.servers.gemma4.serve")


def test_strip_think_removes_think_span() -> None:
    module = _load_module()

    out = module._strip_think(
        "<think>internal reasoning that should vanish</think>final answer"
    )

    assert out == "final answer"


def test_strip_think_leaves_clean_text_untouched() -> None:
    module = _load_module()

    out = module._strip_think("a dog is barking")

    assert out == "a dog is barking"


def test_strip_think_does_not_touch_codec_like_tokens() -> None:
    """Gemma 4 has no audio-generation path (Decision 4): unlike Audex, this
    launcher must NOT strip codec/placeholder-shaped tokens."""

    module = _load_module()

    out = module._strip_think("dog barking<|audio_codec_12|> <audio_pad> at night")

    assert "<|audio_codec_12|>" in out
    assert "<audio_pad>" in out


def test_audio_url_content_item_maps_file_path_to_file_uri(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    # No effective cap in play here (mirrors Audex's max_length_seconds=None
    # case) -- this test is only about the file_path -> file:// mapping.
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=math.inf))

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


def test_audio_url_content_item_falls_back_to_data_uri_for_base64(
    monkeypatch: "__import__('pytest').MonkeyPatch",
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=math.inf))

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


def test_effective_cap_seconds_uses_protocol_cap_when_smaller(
    monkeypatch: "__import__('pytest').MonkeyPatch",
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=30.0))

    assert module._effective_cap_seconds(5) == 5.0


def test_effective_cap_seconds_uses_gemma_ceiling_when_smaller(
    monkeypatch: "__import__('pytest').MonkeyPatch",
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=30.0))

    # Protocol cap larger than Gemma's ceiling -> ceiling wins: min(50, 30).
    assert module._effective_cap_seconds(50) == 30.0


def test_effective_cap_seconds_clamps_to_30s_with_no_protocol_cap(
    monkeypatch: "__import__('pytest').MonkeyPatch",
) -> None:
    """The tiers case: no Audex analogue. BEANS-Next tier tasks send no
    max_length_seconds at all, so the 30s Gemma ceiling is the only limit in
    play and must still apply."""

    module = _load_module()
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=30.0))

    assert module._effective_cap_seconds(None) == 30.0


def test_trim_wav_file_truncates_over_cap_clip(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    monkeypatch.setenv("GEMMA4_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)

    out = module._trim_wav_file(src, 2.0)

    import soundfile as sf

    info = sf.info(str(out))
    assert out != src.resolve()
    assert info.frames / info.samplerate == 2.0


def test_trim_wav_file_passes_through_under_cap_clip(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    monkeypatch.setenv("GEMMA4_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))

    src = tmp_path / "short.wav"
    _write_wav(src, seconds=2.0)

    out = module._trim_wav_file(src, 10.0)

    assert out == src.resolve()


def test_trim_wav_file_reuses_cache(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    monkeypatch.setenv("GEMMA4_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)

    first = module._trim_wav_file(src, 2.0)
    mtime = first.stat().st_mtime_ns
    second = module._trim_wav_file(src, 2.0)

    assert second == first
    assert second.stat().st_mtime_ns == mtime


def test_trim_wav_file_noop_when_cap_is_infinite(tmp_path: Path) -> None:
    module = _load_module()

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)

    out = module._trim_wav_file(src, math.inf)

    assert out == src.resolve()


def test_audio_url_content_item_applies_30s_clamp_with_no_protocol_cap(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    """The tiers case: max_length_seconds=None must still trim at
    GEMMA4_MAX_AUDIO_SECONDS."""

    module = _load_module()
    monkeypatch.setenv("GEMMA4_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=1.0))

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)
    audio_input = module.HttpAudioInput(
        payload_type="file_path", data=str(src), sample_rate=16000
    )

    item = module._audio_url_content_item(audio_input, None)

    import soundfile as sf

    served_path = Path(item["audio_url"]["url"].removeprefix("file://"))
    info = sf.info(str(served_path))
    assert info.frames / info.samplerate == 1.0


def test_audio_url_content_item_min_cap_composition(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    """min(protocol_cap, GEMMA4_MAX_AUDIO_SECONDS): protocol cap (3s) is
    tighter than Gemma's ceiling (30s) -> 3s wins."""

    module = _load_module()
    monkeypatch.setenv("GEMMA4_TRIMMED_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=30.0))

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)
    audio_input = module.HttpAudioInput(
        payload_type="file_path", data=str(src), sample_rate=16000
    )

    item = module._audio_url_content_item(audio_input, 3)

    import soundfile as sf

    served_path = Path(item["audio_url"]["url"].removeprefix("file://"))
    info = sf.info(str(served_path))
    assert info.frames / info.samplerate == 3.0


def test_audio_url_content_item_passes_through_when_under_both_caps(
    tmp_path: Path, monkeypatch: "__import__('pytest').MonkeyPatch"
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "CFG", replace(module.CFG, max_audio_seconds=30.0))

    src = tmp_path / "short.wav"
    _write_wav(src, seconds=2.0)
    audio_input = module.HttpAudioInput(
        payload_type="file_path", data=str(src), sample_rate=16000
    )

    item = module._audio_url_content_item(audio_input, 10)

    assert item["audio_url"]["url"] == src.resolve().as_uri()


def test_trim_base64_wav_truncates_over_cap_clip(tmp_path: Path) -> None:
    module = _load_module()

    src = tmp_path / "long.wav"
    _write_wav(src, seconds=5.0)
    b64 = base64.b64encode(src.read_bytes()).decode("ascii")

    trimmed = module._trim_base64_wav(b64, 2.0)

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

    trimmed = module._trim_base64_wav(b64, 10.0)

    assert trimmed == b64


def test_call_upstream_places_audio_after_text_by_default(
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
            audio_position="after_text",
            max_audio_seconds=math.inf,
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
            captured["body"] = json
            return _FakeResponse()

    monkeypatch.setattr(module, "HTTP", _FakeHttp())

    module._call_upstream_chat_completion(
        messages=[{"role": "user", "content": "what is this?"}],
        audio_inputs=[
            module.HttpAudioInput(
                payload_type="base64_wav", data="AAAA", sample_rate=16000
            )
        ],
        generation_config=module.HttpGenerationConfig(max_tokens=32),
    )

    content = captured["body"]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[-1]["type"] == "audio_url"


def test_call_upstream_places_audio_before_text_when_configured(
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
            audio_position="before_text",
            max_audio_seconds=math.inf,
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
            captured["body"] = json
            return _FakeResponse()

    monkeypatch.setattr(module, "HTTP", _FakeHttp())

    module._call_upstream_chat_completion(
        messages=[{"role": "user", "content": "what is this?"}],
        audio_inputs=[
            module.HttpAudioInput(
                payload_type="base64_wav", data="AAAA", sample_rate=16000
            )
        ],
        generation_config=module.HttpGenerationConfig(max_tokens=32),
    )

    content = captured["body"]["messages"][-1]["content"]
    assert content[0]["type"] == "audio_url"
    assert content[-1] == {"type": "text", "text": "what is this?"}


def test_gemma4_processes_proxy_batches_concurrently(
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


def test_gemma4_extra_body_json_merged_into_upstream_request(
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
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
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
        messages=[{"role": "user", "content": "what is this?"}],
        audio_inputs=[],
        generation_config=module.HttpGenerationConfig(max_tokens=32, temperature=0.0),
    )

    assert text == "a dog is barking"
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["top_p"] == 0.95
    assert captured["body"]["top_k"] == 64
    # Extra body's temperature (vendor sampling arm) overrides the
    # per-request generation_config value applied earlier in the same call.
    assert captured["body"]["temperature"] == 1.0
