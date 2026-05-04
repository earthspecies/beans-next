"""Tests for Increment I11 — JudgeScorer runner integration."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from beans_next.api.http_schemas import (
    PredictionsV1Request,
    PredictionsV1Response,
    PredictionsV1ResponseItem,
)
from beans_next.api.types import ChatMessage, DatasetExample, ModelRequest
from beans_next.judges.scorer import JudgeScorer
from beans_next.runner.runner import BenchmarkRunner, RunnerConfig


class _StubRenderer:
    """Minimal renderer stub returning a fixed single-message request."""

    @property
    def prompt_id(self) -> str:
        return "test-prompt"

    def render(self, example: DatasetExample) -> ModelRequest:
        return ModelRequest(
            sample_id=example.sample_id,
            messages=[ChatMessage(role="user", content=f"q={example.sample_id}")],
            audio_inputs=[],
            generation_config=None,
        )


class _StubHttpClient:
    """HttpClient-shaped stub returning 'dog' for every sample."""

    def __init__(self) -> None:
        self._predict_url = "http://stub.local/predict"
        self._server_info: dict[str, Any] = {
            "name": "stub",
            "schema_versions": ["predictions_v1"],
            "supports_batching": True,
            "max_batch_size": 8,
        }

    @property
    def predict_url(self) -> str:
        return self._predict_url

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
        out = [
            PredictionsV1ResponseItem(sample_id=item.sample_id, predictions=["dog"])
            for item in request.requests
        ]
        return PredictionsV1Response(responses=out)


class _JudgeHandler(BaseHTTPRequestHandler):
    """Stub judge service: returns score=0.8 for every item."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        out_items = [
            {"sample_id": it["sample_id"], "score": 0.8, "rationale": "ok", "error": None}  # noqa: E501
            for it in data["items"]
        ]
        resp = {"schema_version": "judge_scores_v1", "items": out_items}
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_judge_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _JudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/judge"


def test_judge_outputs_written_when_judge_configured(tmp_path: Path) -> None:
    server, judge_url = _start_judge_server()
    try:
        examples = [
            DatasetExample(sample_id="x-1", labels="dog"),
            DatasetExample(sample_id="x-2", labels="cat"),
        ]
        judge = JudgeScorer(judge_url, timeout=5.0, max_attempts=2)
        cfg = RunnerConfig(output_dir=tmp_path / "out", run_id="j-test")
        runner = BenchmarkRunner(
            _StubHttpClient(),
            _StubRenderer(),
            cfg,
            scorer=lambda *_: {"acc": 1.0},
            judge=judge,
        )
        runner.run(examples)

        judge_path = tmp_path / "out" / "judge_outputs.jsonl"
        assert judge_path.is_file(), "judge_outputs.jsonl must be created"
        raw_lines = judge_path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in raw_lines if ln.strip()]
        assert len(lines) == 2, f"expected 2 judge rows, got {len(lines)}"
        ids = {json.loads(ln)["sample_id"] for ln in lines}
        assert ids == {"x-1", "x-2"}
        for line in lines:
            row = json.loads(line)
            assert row["score"] == pytest.approx(0.8)
    finally:
        server.shutdown()
        server.server_close()


def test_judge_outputs_absent_when_no_judge(tmp_path: Path) -> None:
    examples = [DatasetExample(sample_id="y-1", labels="dog")]
    cfg = RunnerConfig(output_dir=tmp_path / "out", run_id="no-judge")
    runner = BenchmarkRunner(
        _StubHttpClient(),
        _StubRenderer(),
        cfg,
        scorer=lambda *_: {},
    )
    runner.run(examples)
    assert not (tmp_path / "out" / "judge_outputs.jsonl").exists()


def test_errored_samples_excluded_from_judge(tmp_path: Path) -> None:
    """Samples that fail inference should not appear in judge inputs."""
    server, judge_url = _start_judge_server()
    try:

        class _ErrorRenderer:
            """Renderer that raises ValueError for sample_id 'bad'."""

            @property
            def prompt_id(self) -> str:
                return "err-prompt"

            def render(self, example: DatasetExample) -> ModelRequest:
                if example.sample_id == "bad":
                    msg = "render error"
                    raise ValueError(msg)
                return ModelRequest(
                    sample_id=example.sample_id,
                    messages=[ChatMessage(role="user", content="q")],
                    audio_inputs=[],
                    generation_config=None,
                )

        examples = [
            DatasetExample(sample_id="bad", labels="x"),
            DatasetExample(sample_id="good", labels="dog"),
        ]
        judge = JudgeScorer(judge_url, timeout=5.0, max_attempts=2)
        cfg = RunnerConfig(output_dir=tmp_path / "out", run_id="err-test")
        runner = BenchmarkRunner(
            _StubHttpClient(),
            _ErrorRenderer(),
            cfg,
            scorer=lambda *_: {},
            judge=judge,
        )
        runner.run(examples)

        judge_path = tmp_path / "out" / "judge_outputs.jsonl"
        assert judge_path.is_file()
        raw_lines = judge_path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in raw_lines if ln.strip()]
        ids = {json.loads(ln)["sample_id"] for ln in lines}
        assert "bad" not in ids
        assert "good" in ids
    finally:
        server.shutdown()
        server.server_close()
