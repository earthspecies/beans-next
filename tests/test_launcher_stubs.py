"""Stub-mode launcher contract conformance tests.

These tests ensure Tier-1 launchers implement the `predictions_v1` HTTP contract
in stub mode, without requiring GPUs or external API keys.

The core conformance logic is shared with the CLI tool `scripts/check_launcher.py`.
"""

from __future__ import annotations

import base64
import io
import os
import socket
import subprocess
import sys
import wave
from collections.abc import Mapping

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


def test_af3_drops_only_zero_token_trailing_window() -> None:
    """AF-Next drops a sub-three-frame tail but keeps valid trailing audio."""
    import numpy as np

    from examples.servers.af3.serve import _drop_zero_token_audio_tail

    window_size = 480_000
    unsafe = np.zeros(window_size * 2 + 319, dtype=np.float32)
    safe = np.zeros(window_size * 2 + 480, dtype=np.float32)
    short = np.zeros(319, dtype=np.float32)

    trimmed = _drop_zero_token_audio_tail(
        unsafe, window_size=window_size, min_tail_samples=480
    )
    assert trimmed.shape == (window_size * 2,)
    assert (
        _drop_zero_token_audio_tail(safe, window_size=window_size, min_tail_samples=480)
        is safe
    )
    assert (
        _drop_zero_token_audio_tail(
            short, window_size=window_size, min_tail_samples=480
        )
        is short
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
