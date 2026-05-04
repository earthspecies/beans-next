"""Tests for Increment I6-A optional two-layer SQLite cache."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from beans_next.api.http_schemas import (
    PredictionsV1Request,
    PredictionsV1Response,
    PredictionsV1ResponseItem,
)
from beans_next.api.types import ChatMessage, DatasetExample, ModelRequest
from beans_next.runner.runner import BenchmarkRunner, RunnerConfig


class _StubRenderer:
    """Minimal renderer stub."""

    def __init__(self, *, prompt_id: str = "test-prompt") -> None:
        self._prompt_id = prompt_id

    @property
    def prompt_id(self) -> str:
        """Prompt identifier."""
        return self._prompt_id

    def render(self, example: DatasetExample) -> ModelRequest:
        return ModelRequest(
            sample_id=example.sample_id,
            messages=[ChatMessage(role="user", content=f"sample={example.sample_id}")],
            audio_inputs=[],
            generation_config=None,
        )


class _StubHttpClient:
    """HttpClient-shaped stub counting ``generate`` invocations."""

    def __init__(self, *, max_batch_size: int = 8) -> None:
        self._predict_url = "http://stub.local/predict"
        self._server_info: dict[str, Any] = {
            "name": "stub",
            "schema_versions": ["predictions_v1"],
            "supports_batching": True,
            "max_batch_size": max_batch_size,
        }
        self.generate_calls = 0
        self.requested_sample_ids: list[str] = []

    @property
    def predict_url(self) -> str:
        """Predict URL."""
        return self._predict_url

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
        self.generate_calls += 1
        ids = [item.sample_id for item in request.requests]
        self.requested_sample_ids.extend(ids)
        out = [
            PredictionsV1ResponseItem(sample_id=sid, predictions=[f"pred:{sid}"])
            for sid in reversed(ids)
        ]
        return PredictionsV1Response(responses=out)


def test_inference_cache_avoids_second_http_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    examples = [DatasetExample(sample_id="a-1", labels="dog")]

    client = _StubHttpClient(max_batch_size=4)
    renderer = _StubRenderer()
    cfg1 = RunnerConfig(
        output_dir=out1,
        run_id="r1",
        cache_dir=cache_dir,
        prompt_version="pv1",
    )
    BenchmarkRunner(client, renderer, cfg1, scorer=lambda *_: {}).run(examples)
    assert client.generate_calls == 1

    client2 = _StubHttpClient(max_batch_size=4)
    cfg2 = RunnerConfig(
        output_dir=out2,
        run_id="r2",
        cache_dir=cache_dir,
        prompt_version="pv1",
    )
    BenchmarkRunner(client2, renderer, cfg2, scorer=lambda *_: {}).run(examples)
    assert client2.generate_calls == 0
    assert (cache_dir / "inference.sqlite").is_file()


def test_scoring_cache_avoids_second_metrics_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import beans_next.metrics as metrics_mod

    cache_dir = tmp_path / "cache"
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    examples = [DatasetExample(sample_id="b-1", labels="dog")]

    calls = {"n": 0}
    real = metrics_mod.score_sample

    def _wrapped(*args: object, **kwargs: object) -> Mapping[str, float]:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics_mod, "score_sample", _wrapped)

    client = _StubHttpClient(max_batch_size=4)
    renderer = _StubRenderer()
    cfg1 = RunnerConfig(
        output_dir=out1,
        run_id="r1",
        cache_dir=cache_dir,
        prompt_version="test-prompt-v1",
    )
    BenchmarkRunner(client, renderer, cfg1).run(examples)
    first = calls["n"]

    client2 = _StubHttpClient(max_batch_size=4)
    cfg2 = RunnerConfig(
        output_dir=out2,
        run_id="r2",
        cache_dir=cache_dir,
        prompt_version="test-prompt-v1",
    )
    BenchmarkRunner(client2, renderer, cfg2).run(examples)

    assert first >= 1
    assert calls["n"] == first
    assert (cache_dir / "scoring.sqlite").is_file()


def test_scoring_cache_disabled_for_custom_scorer(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    examples = [DatasetExample(sample_id="c-1", labels="dog")]
    scorer_calls = {"n": 0}

    def _scorer(*_args: object, **_kwargs: object) -> dict[str, float]:
        scorer_calls["n"] += 1
        return {"custom": 1.0}

    client = _StubHttpClient(max_batch_size=4)
    renderer = _StubRenderer()
    cfg1 = RunnerConfig(
        output_dir=out1,
        run_id="r1",
        cache_dir=cache_dir,
        prompt_version="test-prompt-v1",
    )
    BenchmarkRunner(client, renderer, cfg1, scorer=_scorer).run(examples)
    n1 = scorer_calls["n"]

    client2 = _StubHttpClient(max_batch_size=4)
    cfg2 = RunnerConfig(
        output_dir=out2,
        run_id="r2",
        cache_dir=cache_dir,
        prompt_version="test-prompt-v1",
    )
    BenchmarkRunner(client2, renderer, cfg2, scorer=_scorer).run(examples)

    assert n1 == 1
    assert scorer_calls["n"] == 2


def test_run_records_round_trip_latency(tmp_path: Path) -> None:
    client = _StubHttpClient(max_batch_size=4)
    renderer = _StubRenderer()
    examples = [DatasetExample(sample_id="lat-1", labels="dog")]
    cfg = RunnerConfig(
        output_dir=tmp_path / "out", run_id="r-lat", prompt_version="pv1"
    )
    summary = BenchmarkRunner(client, renderer, cfg).run(examples)
    latency = summary.metrics.get("latency")
    assert latency is not None
    assert latency["mean_roundtrip_sec"] >= 0.0
    assert latency["n_batches"] >= 1
