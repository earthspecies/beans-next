"""Tests for `beans-next pairs` prompt/answer generation.

These tests validate reservoir sampling determinism and that the command writes a JSONL
file with the expected schema when pointed at a minimal predictions_v1 HTTP server.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from beans_next.api.types import DatasetExample


def _write_min_wav(path: Path) -> None:
    # Minimal valid PCM16 mono WAV header + silence payload.
    # 1 sample @ 16kHz is fine for our purposes.
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00")


class _Handler(BaseHTTPRequestHandler):
    """Minimal predictions_v1 HTTP handler for pair-generation tests."""

    server_version = "BeansProTestHTTP/1.0"

    def _send_json(self, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if self.path == "/info":
            self._send_json(
                {
                    "name": "test-launcher",
                    "model": "test-model",
                    "model_revision": "test",
                    "schema_versions": ["predictions_v1"],
                    "supports_batching": True,
                    "max_batch_size": 16,
                    "audio_payload_types": ["file_path"],
                }
            )
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.loads(raw.decode("utf-8"))
        reqs = body["requests"]
        responses = []
        for item in reqs:
            sid = item["sample_id"]
            responses.append(
                {
                    "sample_id": sid,
                    "predictions": ["A"],
                    "finish_reason": "stop",
                    "usage": None,
                    "latency_sec": 0.01,
                    "error": None,
                }
            )
        self._send_json({"schema_version": "predictions_v1", "responses": responses})


@pytest.fixture()
def http_server() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    host, port = srv.server_address
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_reservoir_sampling_is_deterministic(tmp_path: Path) -> None:
    from beans_next.scripts import generate_pairs as gp

    wav = tmp_path / "x.wav"
    _write_min_wav(wav)
    examples = [
        DatasetExample(
            sample_id=f"id{i}",
            task_id="beans_next_crow_description",
            split="crow-description",
            labels="A",
            metadata={"instruction": "<Audio><AudioHere></Audio>\nPick A.", "audio_path": str(wav)},
        )
        for i in range(50)
    ]
    a = gp._sample_reservoir_k(iter(examples), k=10, seed=123)
    b = gp._sample_reservoir_k(iter(examples), k=10, seed=123)
    assert [x.sample_id for x in a] == [x.sample_id for x in b]


def test_pairs_command_writes_jsonl(tmp_path: Path, http_server: str, monkeypatch: Any) -> None:
    from beans_next.scripts import generate_pairs as gp

    wav = tmp_path / "x.wav"
    _write_min_wav(wav)

    fake_examples = [
        DatasetExample(
            sample_id="s1",
            task_id="beans_next_crow_description",
            split="crow-description",
            labels="A",
            metadata={"instruction": "<Audio><AudioHere></Audio>\nPick A.", "audio_path": str(wav)},
        ),
        DatasetExample(
            sample_id="s2",
            task_id="beans_next_crow_description",
            split="crow-description",
            labels="A",
            metadata={"instruction": "<Audio><AudioHere></Audio>\nPick A.", "audio_path": str(wav)},
        ),
    ]

    def _fake_load(*_args: Any, **_kwargs: Any) -> list[DatasetExample]:
        return list(fake_examples)

    monkeypatch.setattr(gp, "_load_beans_next_examples", _fake_load)

    ns = gp.argparse.Namespace(
        predict_url=f"{http_server}/predict",
        model_tag="test",
        run_id="run1",
        output_dir=str(tmp_path / "out"),
        subsets="crow-description",
        k=2,
        seed=0,
        sample_strategy="first",
        workers=1,
        prompt="classification_beans_zero_official_v1",
        batch_size=2,
        http_timeout_sec=5.0,
    )
    code = gp.generate_pairs_main(ns)
    assert code == 0

    out_root = Path(ns.output_dir)
    assert (out_root / "manifest.json").is_file()
    pairs = out_root / "pairs.jsonl"
    assert pairs.is_file()
    lines = pairs.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["subset"] == "crow-description"
    assert row["raw_predictions"] == ["A"]
    assert row["processed_prediction"] == "A"
    assert row["ground_truth"] == "A"
