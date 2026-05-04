"""Tests for ``beans_next.judges`` (HTTP ``judge_scores_v1`` + templates)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from beans_next.api.types import DATASET_EXAMPLE_V1, DatasetExample
from beans_next.judges import (
    JudgeError,
    JudgeHttpError,
    JudgeScorer,
    list_judge_templates,
    post_judge_scores,
    register_judge_template,
)
from beans_next.judges.http_schemas import JudgeScoresV1Request


class _OkHandler(BaseHTTPRequestHandler):
    """HTTP stub: echo ``judge_scores_v1`` with deterministic per-id scores."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        out_items = []
        for it in data["items"]:
            sid = it["sample_id"]
            score = 0.25 if "low" in sid else 0.9
            out_items.append(
                {"sample_id": sid, "score": score, "rationale": "test", "error": None},
            )
        out = {"schema_version": "judge_scores_v1", "items": out_items}
        body = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _MismatchHandler(BaseHTTPRequestHandler):
    """Return a deliberately wrong ``sample_id`` to exercise client validation."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:
        out = {
            "schema_version": "judge_scores_v1",
            "items": [
                {"sample_id": "wrong", "score": 1.0, "rationale": None, "error": None},
            ],
        }
        body = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/"


def test_list_judge_templates_includes_builtin() -> None:
    names = list_judge_templates()
    assert "bioacoustic_open_qa_v1" in names
    assert "bioacoustic_counting_v1" in names


def test_register_judge_template_rejects_duplicate() -> None:
    with pytest.raises(JudgeError, match="already registered"):
        register_judge_template("bioacoustic_open_qa_v1", "x")


def test_build_request_renders_bioacoustic_rubric() -> None:
    ex = DatasetExample(
        schema_version=DATASET_EXAMPLE_V1,
        sample_id="s1",
        task_id="caption_task",
        labels="humpback whale song",
        metadata={},
    )
    scorer = JudgeScorer(
        "http://127.0.0.1:9/judge",
        template_id="bioacoustic_open_qa_v1",
    )
    req = scorer.build_request([ex], ["a whale sound"])
    assert len(req.items) == 1
    assert "bioacoustic" in req.items[0].rubric.lower()
    assert "humpback" in req.items[0].rubric
    assert req.items[0].reference_text == "humpback whale song"


def test_build_request_joins_list_labels() -> None:
    ex = DatasetExample(
        schema_version=DATASET_EXAMPLE_V1,
        sample_id="s2",
        task_id="det",
        labels=["A", "B"],
        metadata={},
    )
    scorer = JudgeScorer("http://127.0.0.1:9/judge")
    ref = scorer.build_request([ex], ["x"]).items[0].reference_text
    assert ref == "A, B"


def test_score_batch_end_to_end() -> None:
    server, url = _serve(_OkHandler)
    try:
        ex = DatasetExample(
            schema_version=DATASET_EXAMPLE_V1,
            sample_id="low-x",
            task_id="cap",
            labels="ref",
            metadata={},
        )
        scorer = JudgeScorer(url, timeout=5.0, max_attempts=2)
        out = scorer.score_batch([ex], ["cand"])
        assert len(out) == 1
        assert out[0].sample_id == "low-x"
        assert out[0].score == pytest.approx(0.25)
    finally:
        server.shutdown()
        server.server_close()


def test_post_judge_scores_id_mismatch() -> None:
    server, url = _serve(_MismatchHandler)
    try:
        req = JudgeScoresV1Request(
            items=[
                {
                    "sample_id": "expected",
                    "rubric": "r",
                    "reference_text": "a",
                    "candidate_text": "b",
                },
            ],
        )
        with pytest.raises(JudgeHttpError, match="sample_id"):
            post_judge_scores(url, req, timeout=5.0, max_attempts=1)
    finally:
        server.shutdown()
        server.server_close()


def test_build_request_length_mismatch() -> None:
    scorer = JudgeScorer("http://127.0.0.1:9/judge")
    ex = DatasetExample(
        schema_version=DATASET_EXAMPLE_V1,
        sample_id="a",
        task_id="t",
        labels="x",
        metadata={},
    )
    with pytest.raises(JudgeError, match="same length"):
        scorer.build_request([ex], ["a", "b"])
