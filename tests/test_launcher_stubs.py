"""Stub-mode launcher contract conformance tests.

These tests ensure Tier-1 launchers implement the `predictions_v1` HTTP contract
in stub mode, without requiring GPUs or external API keys.

The core conformance logic is shared with the CLI tool `scripts/check_launcher.py`.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import wave
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_launcher_conformance(
    *,
    app: str,
    env: Mapping[str, str],
    app_dir: str | None = None,
) -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    app_dir_args = ["--app-dir", app_dir] if app_dir is not None else []

    cmd = [
        sys.executable,
        "scripts/with_uvicorn.py",
        "--app",
        app,
        *app_dir_args,
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


def test_openai_compatible_proxy_stub_mode_conforms() -> None:
    _run_launcher_conformance(
        app="examples.servers.openai_compatible_proxy.serve:app",
        env={"OPENAI_PROXY_STUB": "1"},
    )


def test_openai_compatible_proxy_loads_gemini_cfg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "gemini.cfg"
    cfg_path.write_text("google-test-key\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gemini-3.1-pro")
    monkeypatch.setenv("GEMINI_CFG_PATH", str(cfg_path))
    monkeypatch.setenv("OPENAI_CFG_PATH", str(tmp_path / "missing-openai.cfg"))

    module = importlib.import_module("examples.servers.openai_compatible_proxy.serve")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module._ensure_openai_api_key_loaded()

    assert os.environ["OPENAI_API_KEY"] == "google-test-key"


def test_openai_compatible_proxy_replaces_audio_placeholder_in_place() -> None:
    module = importlib.import_module("examples.servers.openai_compatible_proxy.serve")
    audio_item = {
        "type": "input_audio",
        "input_audio": {"data": "abc", "format": "wav"},
    }

    messages = module._inject_audio_items(
        [{"role": "user", "content": "before <Audio><AudioHere></Audio> after"}],
        [audio_item],
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before "},
                audio_item,
                {"type": "text", "text": " after"},
            ],
        }
    ]


def test_openai_compatible_proxy_processes_proxy_batches_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("examples.servers.openai_compatible_proxy.serve")
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


def test_dummy_launcher_conforms() -> None:
    _run_launcher_conformance(
        app="examples.servers.dummy.serve:app",
        env={},
    )


def test_vllm_adapter_stub_mode_conforms() -> None:
    _run_launcher_conformance(
        app="examples.servers.vllm.adapter:app",
        env={"VLLM_ADAPTER_STUB": "1"},
    )


def test_af3_stub_mode_conforms() -> None:
    _run_launcher_conformance(
        app="examples.servers.af3.serve:app",
        env={"AF3_STUB": "1"},
    )


def test_naturelm_v1_0_stub_mode_conforms() -> None:
    _run_launcher_conformance(
        app="serve:app",
        app_dir="examples/servers/naturelm-v1.0",
        env={"NATURELM_V1_0_STUB": "1"},
    )


def test_naturelm_v1_1_stub_mode_conforms() -> None:
    _run_launcher_conformance(
        app="serve:app",
        app_dir="examples/servers/naturelm-v1.1",
        env={"NATURELM_STUB_MODE": "1"},
    )


def _minimal_wav_b64(*, sample_rate: int = 16_000, n_frames: int = 16) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _run_openai_proxy_live_smoke(
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> None:
    if httpx is None:  # pragma: no cover
        raise AssertionError("httpx is required for live proxy tests")

    port = _free_port()
    local_base = f"http://127.0.0.1:{port}"
    wav_b64 = _minimal_wav_b64()

    env = {
        "OPENAI_PROXY_STUB": "0",
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": api_key,
        "OPENAI_MODEL": model,
        # Keep upstream request bounded even if defaults drift.
        "OPENAI_PROXY_TIMEOUT_SEC": "30",
        "OPENAI_PROXY_RETRIES": "0",
    }

    body: dict[str, Any] = {
        "schema_version": "predictions_v1",
        "requests": [
            {
                "sample_id": "live_smoke_0",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<Audio><AudioHere></Audio>\n"
                            "In one word, what is this audio?"
                        ),
                    }
                ],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": wav_b64,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 128, "temperature": 0.0},
            }
        ],
    }

    cmd = [
        sys.executable,
        "scripts/with_uvicorn.py",
        "--app",
        "examples.servers.openai_compatible_proxy.serve:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--cwd",
        ".",
        "--ready-timeout-s",
        "20",
        "--poll-interval-s",
        "0.05",
        *[x for kv in env.items() for x in ("--env", f"{kv[0]}={kv[1]}")],
        "--",
        sys.executable,
        "-c",
        (
            "import json,sys;"
            "import httpx;"
            f"base={local_base!r};"
            f"body={json.dumps(body, separators=(',', ':'))!r};"
            "c=httpx.Client(timeout=30.0);"
            "r=c.get(base+'/health'); r.raise_for_status();"
            "r=c.get(base+'/info'); r.raise_for_status();"
            "r=c.post(base+'/predict', json=json.loads(body));"
            "r.raise_for_status();"
            "doc=r.json();"
            "assert doc.get('schema_version')=='predictions_v1';"
            "resp=doc.get('responses');"
            "assert isinstance(resp,list) and len(resp)==1;"
            "item=resp[0];"
            "assert item.get('sample_id')=='live_smoke_0';"
            "assert isinstance(item.get('predictions'),list);"
            "assert item.get('error') in (None,'');"
            "pred=item.get('predictions') or []; "
            "assert pred and isinstance(pred[0],str) and pred[0].strip();"
        ),
    ]

    merged_env = dict(os.environ)
    merged_env.update(env)

    res = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        env=merged_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        "Live proxy smoke test failed.\n"
        f"stdout:\n{res.stdout}\n"
        f"stderr:\n{res.stderr}\n"
        f"base_url={base_url!r} model={model!r}"
    )


@pytest.mark.skipif(
    os.environ.get("BEANS_PRO_LIVE_OPENAI", "0").strip() != "1",
    reason="Set BEANS_PRO_LIVE_OPENAI=1 to enable live OpenAI proxy smoke test.",
)
def test_openai_compatible_proxy_live_openai_smoke() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AssertionError("OPENAI_API_KEY must be set when BEANS_PRO_LIVE_OPENAI=1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-audio-preview").strip()
    _run_openai_proxy_live_smoke(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").strip(),
        api_key=api_key,
        model=model,
    )


@pytest.mark.skipif(
    os.environ.get("BEANS_PRO_LIVE_GEMINI", "0").strip() != "1",
    reason="Set BEANS_PRO_LIVE_GEMINI=1 to enable live Gemini proxy smoke test.",
)
def test_openai_compatible_proxy_live_gemini_smoke() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AssertionError("OPENAI_API_KEY must be set when BEANS_PRO_LIVE_GEMINI=1")
    model = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash").strip()
    _run_openai_proxy_live_smoke(
        base_url=os.environ.get(
            "OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ).strip(),
        api_key=api_key,
        model=model,
    )
