"""Run a command while a uvicorn app is alive.

This script exists to make launcher conformance checks deterministic under
restricted execution policies:

- No dependence on sibling/background processes.
- No shell backgrounding via `&`.
- No `sleep`-based readiness guessing.

It starts a uvicorn app in a subprocess, polls a health URL until ready, then
either:

- runs a child command, or
- enters a server-only mode that blocks until a sentinel file exists.

In all cases it terminates the server (best-effort) before exiting.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_DEFAULT_READY_TIMEOUT_S: Final[float] = 15.0
_DEFAULT_POLL_INTERVAL_S: Final[float] = 0.2
_DEFAULT_STOP_TIMEOUT_S: Final[float] = 60.0 * 60.0
_DEFAULT_HEARTBEAT_INTERVAL_S: Final[float] = 5.0
_DEFAULT_OUTPUT_TAIL_BYTES: Final[int] = 200_000
_DEFAULT_READY_EARLY_FAIL_REPEATS: Final[int] = 4
_DEFAULT_READY_ERROR_BODY_MAX_BYTES: Final[int] = 32_000
_DEFAULT_READY_ERROR_BODY_EXCERPT_CHARS: Final[int] = 2_000


@dataclass(frozen=True)
class _RunConfig:
    """Resolved runtime configuration for one `with_uvicorn` execution.

    Parameters
    ----------
    app
        Uvicorn app import path (e.g. ``"serve:app"``).
    app_dir
        Optional directory added to uvicorn's import path via `--app-dir`. This
        supports apps that live in a directory that is not importable from the
        current `PYTHONPATH`.
    host
        Bind host for the uvicorn server.
    port
        Bind port for the uvicorn server.
    server_cmd
        Optional command used to start the server process in "external env/cwd"
        mode. When set, the server is started by executing this command instead
        of `python -m uvicorn <app> ...`. The command must bind to the selected
        `host` + `port`, and should not background itself.
    server_cwd
        Working directory for the uvicorn server process.
    cmd_cwd
        Working directory for the child command (defaults to invocation cwd).
    ready_url
        URL polled until the server is ready (typically ``/health``).
    ready_timeout_s
        Deadline (seconds) for readiness polling.
    poll_interval_s
        Poll interval (seconds) for readiness checks.
    env
        Environment variables passed to the uvicorn process.
    cmd
        Optional child command executed after readiness is confirmed.
    ready_file
        Optional file to create after readiness is confirmed.
    stop_file
        Optional file whose existence ends server-only mode.
    stop_timeout_s
        Deadline (seconds) for waiting on ``stop_file`` in server-only mode.
    heartbeat_file
        Optional file to touch periodically while in server-only mode.
    heartbeat_interval_s
        Seconds between heartbeat touches.
    """

    app: str | None
    app_dir: Path | None
    host: str
    port: int
    server_cmd: Sequence[str] | None
    server_cwd: Path
    cmd_cwd: Path
    ready_url: str
    ready_timeout_s: float
    poll_interval_s: float
    env: Mapping[str, str]
    cmd: Sequence[str] | None
    ready_file: Path | None
    stop_file: Path | None
    stop_timeout_s: float
    heartbeat_file: Path | None
    heartbeat_interval_s: float


def _parse_env_kv(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--env expects KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env expects non-empty KEY in {raw!r}")
        out[key] = value
    return out


class _OutputTail:
    """A bounded tail buffer for capturing subprocess output.

    Parameters
    ----------
    max_bytes
        Maximum number of bytes retained in the buffer. When new output is
        appended beyond this cap, the oldest bytes are dropped.
    """

    def __init__(self, *, max_bytes: int) -> None:
        self._max_bytes = int(max_bytes)
        self._buf = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            self._buf.extend(chunk)
            if len(self._buf) > self._max_bytes:
                self._buf = self._buf[-self._max_bytes :]

    def text(self) -> str:
        with self._lock:
            data = bytes(self._buf)
        return data.decode("utf-8", errors="replace")


def _capture_output(
    proc: subprocess.Popen[bytes], *, tail: _OutputTail
) -> threading.Thread:
    def _reader() -> None:
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    return
                tail.append(chunk)
        except Exception:  # noqa: BLE001
            return

    t = threading.Thread(target=_reader, name="with_uvicorn.stdout", daemon=True)
    t.start()
    return t


def _format_proc_state(proc: subprocess.Popen[bytes]) -> str:
    rc = proc.poll()
    if rc is None:
        return "running"
    return f"exited (returncode={rc})"


def _format_server_context(
    *, cfg: _RunConfig, server_cmd: Sequence[str], proc: subprocess.Popen[bytes]
) -> str:
    return "\n".join(
        [
            "with_uvicorn server context:",
            f"- cmd: {json.dumps(list(server_cmd))}",
            f"- cwd: {str(cfg.server_cwd)}",
            f"- app_dir: {str(cfg.app_dir) if cfg.app_dir is not None else '(none)'}",
            f"- ready_url: {cfg.ready_url}",
            f"- ready_timeout_s: {cfg.ready_timeout_s}",
            f"- poll_interval_s: {cfg.poll_interval_s}",
            f"- state: {_format_proc_state(proc)}",
        ]
    )


def _format_child_context(
    *, cfg: _RunConfig, cmd: Sequence[str], returncode: int
) -> str:
    return "\n".join(
        [
            "with_uvicorn child context:",
            f"- cmd: {json.dumps(list(cmd))}",
            f"- cwd: {str(cfg.cmd_cwd)}",
            f"- returncode: {returncode}",
        ]
    )


def _wait_ready(
    url: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    proc: subprocess.Popen[bytes],
) -> None:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    last_http_sig: tuple[int, str] | None = None
    last_http_sig_repeats = 0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(  # noqa: TRY003
                "Server process exited while waiting for readiness at "
                f"{url} ({_format_proc_state(proc)}). See diagnostics below."
            )
        try:
            req = urllib.request.Request(
                url,
                method="GET",
                headers={"User-Agent": "beans-next/with_uvicorn"},
            )
            with urllib.request.urlopen(req, timeout=1) as resp:
                if 200 <= int(resp.status) < 300:
                    return
                last_err = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            body_bytes: bytes = b""
            try:
                # `HTTPError` is also a file-like object; read once, cap size.
                body_bytes = exc.read(_DEFAULT_READY_ERROR_BODY_MAX_BYTES) or b""
            except Exception:  # noqa: BLE001
                body_bytes = b""
            body_text = body_bytes.decode("utf-8", errors="replace").strip()
            excerpt = body_text[:_DEFAULT_READY_ERROR_BODY_EXCERPT_CHARS]
            if body_text and len(body_text) > _DEFAULT_READY_ERROR_BODY_EXCERPT_CHARS:
                excerpt = excerpt.rstrip() + "\n… (truncated)"

            sig = (int(getattr(exc, "code", 0) or 0), body_text)
            if last_http_sig == sig:
                last_http_sig_repeats += 1
            else:
                last_http_sig = sig
                last_http_sig_repeats = 1

            if excerpt:
                last_err = f"HTTP {exc.code}: {excerpt}"
            else:
                last_err = f"HTTP {exc.code}"

            # Early-fail only when the server is consistently returning the same
            # non-2xx status with an identical, non-empty body. This helps avoid
            # waiting a full timeout on stable misconfiguration errors (e.g. a
            # 503 with a useful error payload) while remaining conservative for
            # typical warm-up paths whose body changes over time (or is empty).
            if (
                last_http_sig_repeats >= _DEFAULT_READY_EARLY_FAIL_REPEATS
                and sig[0] >= 400
                and bool(sig[1])
            ):
                raise RuntimeError(  # noqa: TRY003
                    "Server readiness repeatedly returned the same non-2xx "
                    f"response ({last_http_sig_repeats}x): HTTP {sig[0]}.\n"
                    "Last response body excerpt:\n"
                    f"{excerpt}"
                ) from exc
        except urllib.error.URLError as exc:
            last_err = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        select.select([], [], [], poll_interval_s)
    raise TimeoutError(
        f"Timed out waiting for readiness at {url} ({last_err}); "
        f"server {_format_proc_state(proc)}"
    )


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def _write_ready_json(path: Path, *, url: str, pid: int) -> None:
    payload = {
        "schema_version": "with_uvicorn.ready.v1",
        "ready_url": url,
        "pid": pid,
        "ts_unix": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _wait_for_file(
    path: Path,
    *,
    timeout_s: float,
    poll_interval_s: float,
    heartbeat_file: Path | None,
    heartbeat_interval_s: float,
    proc: subprocess.Popen[bytes] | None,
) -> None:
    deadline = time.time() + timeout_s
    next_heartbeat = 0.0
    while time.time() < deadline:
        now = time.time()
        if path.exists():
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(  # noqa: TRY003
                f"Server process exited while waiting for file {path} "
                f"({_format_proc_state(proc)})"
            )
        if heartbeat_file is not None and now >= next_heartbeat:
            _touch_file(heartbeat_file)
            next_heartbeat = now + heartbeat_interval_s
        select.select([], [], [], poll_interval_s)
    raise TimeoutError(f"Timed out waiting for file to exist: {path}")


def _terminate_process(
    proc: subprocess.Popen[bytes], *, timeout_s: float = 5.0
) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.pid is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                proc.send_signal(signal.SIGTERM)
        else:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=timeout_s)
        return
    except Exception:  # noqa: BLE001
        pass
    if proc.poll() is not None:
        return
    try:
        if proc.pid is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                proc.kill()
        else:
            proc.kill()
        proc.wait(timeout=timeout_s)
    except Exception:  # noqa: BLE001
        return


def _parse_server_cmd(raw: str) -> Sequence[str]:
    cmd = tuple(shlex.split(str(raw)))
    if not cmd:
        raise SystemExit("--server-cmd must be non-empty")
    return cmd


def _policy_check_no_backgrounding(tokens: Sequence[str], *, flag_name: str) -> None:
    joined = " ".join(tokens)
    if "&" in tokens or joined.strip().endswith("&"):
        raise SystemExit(f"{flag_name} must not background the server with '&'")


def _policy_check_no_sleep_prefix(tokens: Sequence[str], *, flag_name: str) -> None:
    # Don't allow "sleep 5 && ..." readiness guessing.
    if not tokens:
        return
    if tokens[0] == "sleep" or tokens[:2] == ["bash", "-lc"] and "sleep" in tokens[2:3]:
        raise SystemExit(f"{flag_name} must not start with `sleep ...`")
    if tokens[0] in {"bash", "sh", "zsh"}:
        try:
            idx = list(tokens).index("-lc")
        except ValueError:
            return
        if idx + 1 >= len(tokens):
            return
        cmd_str = str(tokens[idx + 1]).lstrip()
        if cmd_str.startswith("sleep "):
            raise SystemExit(f"{flag_name} must not start with `sleep ...`")


def _policy_check_no_shuf(tokens: Sequence[str], *, flag_name: str) -> None:
    # Disallow `shuf` anywhere in the command; ports must be fixed.
    if any(tok == "shuf" or tok.endswith("/shuf") for tok in tokens):
        raise SystemExit(f"{flag_name} must not use `shuf` (ports must be fixed)")


def _extract_flag_value(tokens: Sequence[str], *, flag: str) -> str | None:
    # Supports `--flag value` and `--flag=value`.
    for i, tok in enumerate(tokens):
        if tok == flag:
            if i + 1 >= len(tokens):
                return None
            return tokens[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _policy_check_server_cmd_binds_expected_host_port(
    tokens: Sequence[str], *, host: str, port: int
) -> None:
    port_raw = _extract_flag_value(tokens, flag="--port")
    if port_raw is None:
        raise SystemExit("--server-cmd must include an explicit `--port <PORT>`")
    try:
        cmd_port = int(port_raw)
    except ValueError as exc:
        raise SystemExit(
            f"--server-cmd has non-integer --port value: {port_raw!r}"
        ) from exc
    if cmd_port != int(port):
        raise SystemExit(
            f"--server-cmd binds to --port {cmd_port}, expected {int(port)} "
            "(must match --port)."
        )

    host_raw = _extract_flag_value(tokens, flag="--host")
    if host_raw is not None and str(host_raw) != str(host):
        raise SystemExit(
            f"--server-cmd binds to --host {host_raw!r}, expected {str(host)!r} "
            "(must match --host when provided)."
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="with_uvicorn.py",
        description=(
            "Start a uvicorn app, poll readiness, then run a command or wait for "
            "a sentinel file; stop server on exit."
        ),
    )
    p.add_argument(
        "--app",
        required=False,
        help="Uvicorn app import path, e.g. examples.servers.dummy.serve:app",
    )
    p.add_argument(
        "--app-dir",
        default=None,
        help=(
            "Directory containing the uvicorn app module when it is not importable "
            "from the current environment (passed through to `uvicorn --app-dir`)."
        ),
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1).",
    )
    p.add_argument("--port", type=int, required=True, help="Bind port.")
    p.add_argument(
        "--cwd",
        default=".",
        help="Working directory for uvicorn server process (default: current dir).",
    )
    p.add_argument(
        "--server-cwd",
        default=None,
        help=(
            "Working directory for the server process when using --server-cmd. "
            "Defaults to --cwd when unset."
        ),
    )
    p.add_argument(
        "--server-cmd",
        default=None,
        help=(
            "If set, start the server by executing this command in --server-cwd "
            "(or --cwd). Readiness is still polled via --ready-url."
        ),
    )
    p.add_argument(
        "--cmd-cwd",
        default=None,
        help="Working directory for the child command (default: current dir).",
    )
    p.add_argument(
        "--ready-url",
        default=None,
        help="Readiness URL to poll; defaults to http://<host>:<port>/health",
    )
    p.add_argument(
        "--ready-timeout-s",
        type=float,
        default=_DEFAULT_READY_TIMEOUT_S,
        help=f"Seconds to wait for readiness (default: {_DEFAULT_READY_TIMEOUT_S}).",
    )
    p.add_argument(
        "--poll-interval-s",
        type=float,
        default=_DEFAULT_POLL_INTERVAL_S,
        help=f"Seconds between polls (default: {_DEFAULT_POLL_INTERVAL_S}).",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env vars for uvicorn process; can be repeated.",
    )
    p.add_argument(
        "--ready-file",
        default=None,
        help=(
            "If set, write a JSON 'ready' document after readiness is confirmed. "
            "Intended for cross-process coordination (e.g. server-owner agent "
            "signaling clients)."
        ),
    )
    p.add_argument(
        "--heartbeat-file",
        default=None,
        help=(
            "If set (in server-only mode), touch this file periodically while "
            "waiting for --stop-file. Clients may treat a stale heartbeat as dead."
        ),
    )
    p.add_argument(
        "--heartbeat-interval-s",
        type=float,
        default=_DEFAULT_HEARTBEAT_INTERVAL_S,
        help=(
            "Seconds between heartbeat touches in server-only mode "
            f"(default: {_DEFAULT_HEARTBEAT_INTERVAL_S})."
        ),
    )
    p.add_argument(
        "--stop-file",
        default=None,
        help=(
            "If set (and no child command is provided), keep the server alive "
            "until this file exists, then exit 0."
        ),
    )
    p.add_argument(
        "--stop-timeout-s",
        type=float,
        default=_DEFAULT_STOP_TIMEOUT_S,
        help=(
            "Seconds to wait for --stop-file in server-only mode "
            f"(default: {_DEFAULT_STOP_TIMEOUT_S})."
        ),
    )
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help=(
            "Command to run after -- (e.g. -- uv run bash "
            "scripts/check_launcher.sh ...)."
        ),
    )
    return p


def _resolve_config(ns: argparse.Namespace) -> _RunConfig:
    base_server_cwd = Path(str(ns.cwd)).expanduser().resolve()
    server_cwd_raw = ns.server_cwd if ns.server_cwd is not None else base_server_cwd
    server_cwd = Path(str(server_cwd_raw)).expanduser().resolve()
    if not server_cwd.is_dir():
        raise SystemExit(f"--cwd must be a directory, got {server_cwd}")
    if ns.server_cwd is not None and not server_cwd.is_dir():
        raise SystemExit(f"--server-cwd must be a directory, got {server_cwd}")
    app_dir = (
        Path(str(ns.app_dir)).expanduser().resolve() if ns.app_dir is not None else None
    )
    if app_dir is not None and not app_dir.is_dir():
        raise SystemExit(f"--app-dir must be a directory, got {app_dir}")
    cmd_cwd_raw = ns.cmd_cwd if ns.cmd_cwd is not None else "."
    cmd_cwd = Path(str(cmd_cwd_raw)).expanduser().resolve()
    if not cmd_cwd.is_dir():
        raise SystemExit(f"--cmd-cwd must be a directory, got {cmd_cwd}")
    if not isinstance(ns.port, int) or ns.port < 1 or ns.port > 65535:
        raise SystemExit("--port must be an integer in [1, 65535]")
    ready_url = ns.ready_url
    if ready_url is None:
        ready_url = f"http://{ns.host}:{ns.port}/health"
    ready_timeout_s = float(ns.ready_timeout_s)
    if ready_timeout_s <= 0:
        raise SystemExit("--ready-timeout-s must be > 0")
    poll_interval_s = float(ns.poll_interval_s)
    if poll_interval_s <= 0:
        raise SystemExit("--poll-interval-s must be > 0")
    cmd_raw = list(ns.cmd)
    ready_file = (
        Path(str(ns.ready_file)).expanduser().resolve()
        if ns.ready_file is not None
        else None
    )
    stop_file = (
        Path(str(ns.stop_file)).expanduser().resolve()
        if ns.stop_file is not None
        else None
    )
    heartbeat_file = (
        Path(str(ns.heartbeat_file)).expanduser().resolve()
        if ns.heartbeat_file is not None
        else None
    )
    heartbeat_interval_s = float(ns.heartbeat_interval_s)
    if heartbeat_interval_s <= 0:
        raise SystemExit("--heartbeat-interval-s must be > 0")
    stop_timeout_s = float(ns.stop_timeout_s)
    if stop_timeout_s <= 0:
        raise SystemExit("--stop-timeout-s must be > 0")

    # `argparse.REMAINDER` includes the `--` separator; strip all leading
    # sentinel `--` tokens (users sometimes include an extra one), but do not
    # drop legitimate `--foo` arguments.
    cmd = cmd_raw
    while cmd[:1] == ["--"]:
        cmd = cmd[1:]
    cmd_seq: Sequence[str] | None = tuple(cmd) if cmd else None
    if cmd_seq is None and stop_file is None:
        raise SystemExit(
            "Missing child command and no --stop-file provided. Use either "
            "`... -- <command...>` or server-only mode with --stop-file."
        )
    env = dict(os.environ)
    env.update(_parse_env_kv(list(ns.env)))

    server_cmd: Sequence[str] | None = None
    if ns.server_cmd is not None:
        server_cmd = _parse_server_cmd(str(ns.server_cmd))
        _policy_check_no_backgrounding(server_cmd, flag_name="--server-cmd")
        _policy_check_no_sleep_prefix(server_cmd, flag_name="--server-cmd")
        _policy_check_no_shuf(server_cmd, flag_name="--server-cmd")
        _policy_check_server_cmd_binds_expected_host_port(
            server_cmd, host=str(ns.host), port=int(ns.port)
        )

    app = str(ns.app) if ns.app is not None else None
    if server_cmd is None:
        if app is None:
            raise SystemExit("Missing --app (required when --server-cmd is not set)")
    else:
        if app is not None:
            raise SystemExit(
                "Use either --app (import mode) or --server-cmd (command mode), "
                "not both"
            )

    return _RunConfig(
        app=app,
        app_dir=app_dir,
        host=str(ns.host),
        port=int(ns.port),
        server_cmd=server_cmd,
        server_cwd=server_cwd,
        cmd_cwd=cmd_cwd,
        ready_url=str(ready_url),
        ready_timeout_s=ready_timeout_s,
        poll_interval_s=poll_interval_s,
        env=env,
        cmd=cmd_seq,
        ready_file=ready_file,
        stop_file=stop_file,
        stop_timeout_s=stop_timeout_s,
        heartbeat_file=heartbeat_file,
        heartbeat_interval_s=heartbeat_interval_s,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    cfg = _resolve_config(ns)

    server_cmd: list[str]
    if cfg.server_cmd is not None:
        server_cmd = list(cfg.server_cmd)
    else:
        assert cfg.app is not None
        server_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            cfg.app,
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "--log-level",
            "info",
        ]
        if cfg.app_dir is not None:
            server_cmd.extend(["--app-dir", str(cfg.app_dir)])

    proc = subprocess.Popen(
        server_cmd,
        cwd=str(cfg.server_cwd),
        env=dict(cfg.env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    output_tail = _OutputTail(max_bytes=_DEFAULT_OUTPUT_TAIL_BYTES)
    _capture_output(proc, tail=output_tail)
    timed_out: TimeoutError | None = None
    failed: Exception | None = None
    diagnostics: str | None = None
    child_context: str | None = None
    try:
        _wait_ready(
            cfg.ready_url,
            timeout_s=cfg.ready_timeout_s,
            poll_interval_s=cfg.poll_interval_s,
            proc=proc,
        )
        if cfg.ready_file is not None:
            _write_ready_json(cfg.ready_file, url=cfg.ready_url, pid=int(proc.pid))

        if cfg.cmd is not None:
            child = subprocess.run(
                list(cfg.cmd),
                cwd=str(cfg.cmd_cwd),
                env=dict(cfg.env),
                check=False,
            )
            child_rc = int(child.returncode)
            if child_rc != 0:
                failed = RuntimeError(
                    f"Child command failed (returncode={child_rc}). "
                    "See diagnostics below."
                )
                diagnostics = _format_server_context(
                    cfg=cfg, server_cmd=server_cmd, proc=proc
                )
                child_context = _format_child_context(
                    cfg=cfg, cmd=cfg.cmd, returncode=child_rc
                )
            return child_rc

        assert cfg.stop_file is not None
        _wait_for_file(
            cfg.stop_file,
            timeout_s=cfg.stop_timeout_s,
            poll_interval_s=cfg.poll_interval_s,
            heartbeat_file=cfg.heartbeat_file,
            heartbeat_interval_s=cfg.heartbeat_interval_s,
            proc=proc,
        )
        return 0
    except TimeoutError as exc:
        timed_out = exc
        diagnostics = _format_server_context(cfg=cfg, server_cmd=server_cmd, proc=proc)
        return 1
    except Exception as exc:
        failed = exc
        diagnostics = _format_server_context(cfg=cfg, server_cmd=server_cmd, proc=proc)
        return 1
    finally:
        _terminate_process(proc)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:  # noqa: BLE001
                pass
        tail = output_tail.text().strip()
        if timed_out is not None:
            sys.stderr.write(f"{timed_out}\n")
            if diagnostics:
                sys.stderr.write(f"{diagnostics}\n")
            if tail:
                sys.stderr.write("--- uvicorn output (tail) ---\n")
                sys.stderr.write(tail)
                if not tail.endswith("\n"):
                    sys.stderr.write("\n")
        elif failed is not None:
            sys.stderr.write(f"{type(failed).__name__}: {failed}\n")
            if diagnostics:
                sys.stderr.write(f"{diagnostics}\n")
            if child_context:
                sys.stderr.write(f"{child_context}\n")
            if tail:
                sys.stderr.write("--- uvicorn output (tail) ---\n")
                sys.stderr.write(tail)
                if not tail.endswith("\n"):
                    sys.stderr.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
