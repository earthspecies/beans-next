"""Tests for `scripts/with_uvicorn.py` server lifecycle wrapper."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ok(url: str, *, timeout_s: float = 10.0) -> None:
    deadline = __import__("time").time() + float(timeout_s)
    last_err: str | None = None
    while __import__("time").time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                if 200 <= int(resp.status) < 300:
                    return
                last_err = f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        __import__("time").sleep(0.05)
    raise AssertionError(f"Timed out waiting for {url}: {last_err}")


def _assert_http_down(url: str, *, timeout_s: float = 5.0) -> None:
    deadline = __import__("time").time() + float(timeout_s)
    while __import__("time").time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1):
                pass
        except Exception:
            return
        __import__("time").sleep(0.05)
    raise AssertionError(f"URL still reachable (expected down): {url}")


def _write_asgi_app(tmp_path: Path) -> None:
    (tmp_path / "serve.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            async def app(scope, receive, send):
                if scope["type"] != "http":
                    return
                path = scope.get("path") or "/"
                if path == "/health":
                    body = b"ok"
                    status = 200
                else:
                    body = b"not found"
                    status = 404
                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": body})
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_with_uvicorn_import_mode_starts_and_tears_down(tmp_path: Path) -> None:
    _write_asgi_app(tmp_path)
    port = _free_port()
    url = f"http://127.0.0.1:{port}/health"

    cmd = [
        sys.executable,
        "scripts/with_uvicorn.py",
        "--app",
        "serve:app",
        "--app-dir",
        str(tmp_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ready-url",
        url,
        "--",
        sys.executable,
        "-c",
        f"import urllib.request; urllib.request.urlopen({url!r}).read()",
    ]
    res = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    _assert_http_down(url)


def test_with_uvicorn_server_cmd_mode_starts_and_tears_down(tmp_path: Path) -> None:
    _write_asgi_app(tmp_path)
    port = _free_port()
    url = f"http://127.0.0.1:{port}/health"

    server_cmd = " ".join(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "serve:app",
            "--app-dir",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]
    )

    cmd = [
        sys.executable,
        "scripts/with_uvicorn.py",
        "--server-cwd",
        str(tmp_path),
        "--server-cmd",
        server_cmd,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ready-url",
        url,
        "--",
        sys.executable,
        "-c",
        f"import urllib.request; urllib.request.urlopen({url!r}).read()",
    ]
    res = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    _assert_http_down(url)
