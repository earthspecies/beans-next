"""Retry-policy behavior tests for :class:`beans_next.models.http.HttpClient`."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

import pytest

from beans_next.api.http_schemas import (
    PREDICTIONS_V1,
    HttpAudioInput,
    HttpChatMessage,
    HttpGenerationConfig,
    PredictionsV1Request,
    PredictionsV1RequestItem,
)
from beans_next.models.http import HttpClient, HttpClientFatalError, RetryPolicy


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for deterministic urllib-based tests."""

    daemon_threads = True


def _start_test_server(
    *, failures_before_success: int
) -> tuple[str, dict[str, Any], _ThreadingHTTPServer]:
    state: dict[str, Any] = {
        "failures_left": int(failures_before_success),
        "predict_calls": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        """Minimal `/health`, `/info`, `/predict` server with synthetic failures."""

        def log_message(self, _fmt: str, *_args: object) -> None:
            # Avoid noisy test output.
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = (json.dumps(payload) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if self.path == "/info":
                self._send_json(
                    200,
                    {
                        "name": "retry-policy-test-server",
                        "model": "unit-test",
                        "model_revision": "local",
                        "audio_payload_types": ["base64_wav"],
                        "max_batch_size": 16,
                        "supports_batching": True,
                        "schema_versions": [PREDICTIONS_V1],
                    },
                )
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self.send_response(404)
                self.end_headers()
                return

            state["predict_calls"] += 1
            if state["failures_left"] > 0:
                state["failures_left"] -= 1
                body = b"server error (synthetic)\n"
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            payload = json.loads(raw.decode("utf-8"))
            sample_ids = [x["sample_id"] for x in payload["requests"]]

            self._send_json(
                200,
                {
                    "schema_version": PREDICTIONS_V1,
                    "responses": [
                        {
                            "sample_id": sid,
                            "predictions": ["ok"],
                            "finish_reason": "stop",
                            "error": None,
                        }
                        for sid in sample_ids
                    ],
                },
            )

    server = _ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    return base_url, state, server


@pytest.fixture()
def request_one() -> PredictionsV1Request:
    return PredictionsV1Request(
        requests=[
            PredictionsV1RequestItem(
                sample_id="s0",
                messages=[HttpChatMessage(role="user", content="hi")],
                audio_inputs=[
                    HttpAudioInput(
                        payload_type="base64_wav",
                        data="AA==",
                        sample_rate=16000,
                    )
                ],
                generation_config=HttpGenerationConfig(max_tokens=1, temperature=0.0),
            )
        ]
    )


def test_retry_policy_max_attempts_override(request_one: PredictionsV1Request) -> None:
    base_url_1, state_1, server_1 = _start_test_server(failures_before_success=2)
    try:
        with HttpClient(f"{base_url_1}/predict", timeout=2.0) as client:
            client.generate(request_one)
        assert state_1["predict_calls"] == 3
    finally:
        server_1.shutdown()
        server_1.server_close()

    base_url_2, state_2, server_2 = _start_test_server(failures_before_success=2)
    try:
        policy = RetryPolicy(
            max_attempts=2,
            backoff_initial=0.0,
            backoff_max=0.0,
            jitter_fraction=0.0,
        )
        with pytest.raises(HttpClientFatalError):
            with HttpClient(
                f"{base_url_2}/predict",
                timeout=2.0,
                retry_policy=policy,
            ) as client:
                client.generate(request_one)
        assert state_2["predict_calls"] == 2
    finally:
        server_2.shutdown()
        server_2.server_close()


def test_retry_policy_retryable_status_override(
    request_one: PredictionsV1Request,
) -> None:
    base_url, state, server = _start_test_server(failures_before_success=1)
    try:
        policy = RetryPolicy(
            max_attempts=3,
            backoff_initial=0.0,
            backoff_max=0.0,
            jitter_fraction=0.0,
            retry_on_5xx=False,
            retry_http_statuses=frozenset(),
        )
        with pytest.raises(HttpClientFatalError):
            with HttpClient(
                f"{base_url}/predict",
                timeout=2.0,
                retry_policy=policy,
            ) as client:
                client.generate(request_one)
        assert state["predict_calls"] == 1
    finally:
        server.shutdown()
        server.server_close()
