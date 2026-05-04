"""Runner regression: prompt render errors must be sample-level."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from beans_next.api.http_schemas import PredictionsV1Request, PredictionsV1Response
from beans_next.api.types import ChatMessage, DatasetExample, ModelRequest
from beans_next.runner.runner import BenchmarkRunner, RunnerConfig


class _SometimesFailingRenderer:
    """Renderer stub that raises for one sample id."""

    def __init__(self, *, fail_sample_id: str) -> None:
        self._fail_sample_id = fail_sample_id

    def render(self, example: DatasetExample) -> ModelRequest:
        if example.sample_id == self._fail_sample_id:
            raise ValueError(
                "metadata['audio_path'] must be a non-empty str "
                f"for sample_id={example.sample_id!r}"
            )
        return ModelRequest(
            sample_id=example.sample_id,
            messages=[ChatMessage(role="user", content="ok")],
            audio_inputs=[],
            generation_config=None,
        )


class _StubHttpClient:
    """Local `HttpClient`-shaped stub capturing requested sample ids."""

    def __init__(self) -> None:
        self.requested_sample_ids: list[str] = []
        self._server_info: dict[str, Any] = {
            "name": "stub",
            "schema_versions": ["predictions_v1"],
            "supports_batching": True,
            "max_batch_size": 8,
        }

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
        self.requested_sample_ids.extend([r.sample_id for r in request.requests])
        # Should never be called for the failing sample id.
        responses = [
            {"sample_id": r.sample_id, "predictions": [f"pred:{r.sample_id}"]}
            for r in request.requests
        ]
        return PredictionsV1Response.model_validate(
            {"schema_version": "predictions_v1", "responses": responses}
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def test_prompt_render_error_is_sample_level(tmp_path: Path) -> None:
    bad_id = "bad"
    good_id = "good"
    examples = [
        DatasetExample(sample_id=bad_id, task_id="t0", labels="x", metadata={}),
        DatasetExample(sample_id=good_id, task_id="t0", labels="x", metadata={}),
    ]
    client = _StubHttpClient()
    renderer = _SometimesFailingRenderer(fail_sample_id=bad_id)
    cfg = RunnerConfig(output_dir=tmp_path, run_id="test-run", prompt_version="pv1")
    runner = BenchmarkRunner(client=client, renderer=renderer, config=cfg)

    summary = runner.run(examples)

    assert client.requested_sample_ids == [good_id]
    assert summary.n_samples == 2
    assert summary.n_errors == 1

    pred_rows = _read_jsonl(tmp_path / "predictions.jsonl")
    assert [r["sample_id"] for r in pred_rows] == [bad_id, good_id]
    bad_row = pred_rows[0]
    assert bad_row["predictions"] == []
    assert isinstance(bad_row["error"], str) and bad_row["error"]


class _PartialResponseHttpClient:
    """Stub that returns only the first item from each batch request."""

    def __init__(self) -> None:
        self._server_info: dict[str, Any] = {
            "name": "stub",
            "schema_versions": ["predictions_v1"],
            "supports_batching": True,
            "max_batch_size": 8,
        }

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
        first = request.requests[0]
        return PredictionsV1Response.model_validate(
            {
                "schema_version": "predictions_v1",
                "responses": [
                    {"sample_id": first.sample_id, "predictions": ["ok"]}
                ],
            }
        )


def test_render_error_recorded_in_checkpoint_failures(tmp_path: Path) -> None:
    bad_id = "bad"
    good_id = "good"
    examples = [
        DatasetExample(sample_id=bad_id, task_id="t0", labels="x", metadata={}),
        DatasetExample(sample_id=good_id, task_id="t0", labels="x", metadata={}),
    ]
    client = _StubHttpClient()
    renderer = _SometimesFailingRenderer(fail_sample_id=bad_id)
    cfg = RunnerConfig(output_dir=tmp_path, run_id="test-run", prompt_version="pv1")
    BenchmarkRunner(client=client, renderer=renderer, config=cfg).run(examples)

    ck = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    failures = ck.get("failures", [])
    render_failures = [f for f in failures if f.get("stage") == "render"]
    assert any(f["sample_id"] == bad_id for f in render_failures)


def test_partial_server_response_raises_value_error(tmp_path: Path) -> None:
    examples = [
        DatasetExample(sample_id="p-0", labels="x", metadata={}),
        DatasetExample(sample_id="p-1", labels="x", metadata={}),
    ]
    client = _PartialResponseHttpClient()
    renderer = _SometimesFailingRenderer(fail_sample_id="__never__")
    cfg = RunnerConfig(output_dir=tmp_path, run_id="test-run", prompt_version="pv1")
    runner = BenchmarkRunner(client=client, renderer=renderer, config=cfg)

    with pytest.raises(ValueError, match="missing"):
        runner.run(examples)
