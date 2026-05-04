"""Tests for scripts/with_uvicorn.py server-only mode (--ready-file / --stop-file)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def _wait_for_file(path: Path, *, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for file: {path}")


def test_with_uvicorn_server_only_mode_writes_ready_and_stops(tmp_path: Path) -> None:
    ready_file = tmp_path / "dummy.ready"
    heartbeat_file = tmp_path / "dummy.heartbeat"
    stop_file = tmp_path / "dummy.stop"
    port = 19081

    cmd = [
        sys.executable,
        "scripts/with_uvicorn.py",
        "--app",
        "examples.servers.dummy.serve:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--cwd",
        ".",
        "--ready-timeout-s",
        "10",
        "--poll-interval-s",
        "0.05",
        "--ready-file",
        str(ready_file),
        "--heartbeat-file",
        str(heartbeat_file),
        "--heartbeat-interval-s",
        "0.05",
        "--stop-file",
        str(stop_file),
        "--stop-timeout-s",
        "10",
    ]

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_file(ready_file, timeout_s=10.0)
        payload = json.loads(ready_file.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "with_uvicorn.ready.v1"
        assert payload["ready_url"] == f"http://127.0.0.1:{port}/health"
        assert isinstance(payload["pid"], int) and payload["pid"] > 0
        assert isinstance(payload["ts_unix"], (int, float)) and payload["ts_unix"] > 0

        _wait_for_file(heartbeat_file, timeout_s=10.0)
        mtime0 = heartbeat_file.stat().st_mtime
        time.sleep(0.15)
        mtime1 = heartbeat_file.stat().st_mtime
        assert mtime1 > mtime0

        stop_file.write_text("stop\n", encoding="utf-8")
        rc = proc.wait(timeout=10.0)
        assert rc == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
