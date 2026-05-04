"""Tests for Increment 6 resume + workers behavior.

These tests are CPU-only and use local stubs for HTTP inference (no external
FastAPI servers).
"""

from __future__ import annotations

import inspect
import json
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import pytest

from beans_next.api.http_schemas import (
    PredictionsV1Request,
    PredictionsV1Response,
    PredictionsV1ResponseItem,
)
from beans_next.api.types import ChatMessage, DatasetExample, ModelRequest
from beans_next.runner.runner import (
    BenchmarkRunner,
    RunnerConfig,
    run_from_cli_namespace,
)


class _StubRenderer:
    """Minimal renderer stub producing `ModelRequest` rows per sample."""

    def __init__(self, *, prompt_id: str = "test-prompt") -> None:
        self._prompt_id = prompt_id

    def render(self, example: DatasetExample) -> ModelRequest:
        return ModelRequest(
            sample_id=example.sample_id,
            messages=[ChatMessage(role="user", content=f"sample={example.sample_id}")],
            audio_inputs=[],
            generation_config=None,
        )


class _StubHttpClient:
    """Local `HttpClient`-shaped stub with deterministic, out-of-order responses."""

    def __init__(self, *, max_batch_size: int = 8) -> None:
        self._server_info: dict[str, Any] = {
            "name": "stub",
            "schema_versions": ["predictions_v1"],
            "supports_batching": True,
            "max_batch_size": max_batch_size,
        }
        self.requested_sample_ids: list[str] = []

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
        # Intentionally shuffle response order to ensure caller doesn't rely on it.
        ids = [item.sample_id for item in request.requests]
        self.requested_sample_ids.extend(ids)
        out = [
            PredictionsV1ResponseItem(sample_id=sid, predictions=[f"pred:{sid}"])
            for sid in reversed(ids)
        ]
        return PredictionsV1Response(responses=out)


def _runner_config(
    output_dir: Path,
    *,
    resume: bool,
    workers: int | None,
) -> RunnerConfig:
    cfg_fields = {f.name for f in dataclass_fields(RunnerConfig)}
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "run_id": "test-run",
        # Avoid touching renderer internals; summary should use this.
        "prompt_version": "test-prompt-v1",
    }

    # Resume naming is still in-flight across I6-B1/I6-B2; set whichever exists.
    for key in (
        "resume",
        "resume_from_checkpoint",
        "resume_from_output_dir",
        "resume_from_existing",
    ):
        if key in cfg_fields:
            kwargs[key] = bool(resume)
            break

    if workers is not None:
        for key in ("workers", "n_workers", "max_workers"):
            if key in cfg_fields:
                kwargs[key] = int(workers)
                break

    return RunnerConfig(**kwargs)


def _resume_skip_supported() -> bool:
    # `BenchmarkArtifactWriter` can append without duplication, but true resume
    # requires skipping completed ids before HTTP calls. Feature-detect by
    # checking whether the runner references the checkpoint helper.
    import beans_next.runner.runner as runner_mod

    try:
        src = inspect.getsource(runner_mod)
    except OSError:  # pragma: no cover
        return False
    return "completed_sample_ids_from_checkpoint_json" in src


def _run_with_optional_kwargs(
    runner: BenchmarkRunner,
    examples: list[DatasetExample],
    *,
    resume: bool,
    workers: int | None,
) -> None:
    sig = inspect.signature(runner.run)
    kwargs: dict[str, Any] = {}
    if "resume" in sig.parameters:
        kwargs["resume"] = bool(resume)
    if workers is not None:
        for key in ("workers", "n_workers", "max_workers"):
            if key in sig.parameters:
                kwargs[key] = int(workers)
                break
    runner.run(examples, **kwargs)


def _read_jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        ids.append(str(obj["sample_id"]))
    return ids


def _write_minimal_existing_artifacts(
    output_dir: Path,
    *,
    completed_ids: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    proc_path = output_dir / "processed_predictions.jsonl"
    scored_path = output_dir / "scored_predictions.jsonl"

    for p in (pred_path, proc_path, scored_path):
        p.write_text("", encoding="utf-8")

    with (
        pred_path.open("a", encoding="utf-8") as pred_f,
        proc_path.open("a", encoding="utf-8") as proc_f,
        scored_path.open("a", encoding="utf-8") as scored_f,
    ):
        for sid in completed_ids:
            pred_f.write(
                json.dumps(
                    {
                        "schema_version": "predictions_v1",
                        "sample_id": sid,
                        "predictions": [f"pred:{sid}"],
                        "finish_reason": None,
                        "usage": None,
                        "latency_sec": None,
                        "error": None,
                        "server_info": {"name": "stub"},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )
            base_scored = {
                "schema_version": "beans_next.scored_prediction.v1",
                "sample_id": sid,
                "task_id": None,
                "predictions": [f"pred:{sid}"],
                "processed_prediction": f"pred:{sid}",
                "targets": None,
                "scores": None,
                "postprocess_version": None,
                "error": None,
            }
            line = json.dumps(
                base_scored,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            proc_f.write(line + "\n")
            scored_f.write(line + "\n")

    (output_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": "beans_next.checkpoint.v1",
                "run_id": "test-run",
                "completed_sample_ids": list(completed_ids),
                "n_predictions_written": len(completed_ids),
                "n_errors": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_resume_skips_completed_and_no_duplicate_artifact_lines(tmp_path: Path) -> None:
    if not _resume_skip_supported():
        pytest.xfail("Runner resume-skip behavior not implemented yet (I6-B1/I6-B2).")

    output_dir = tmp_path / "out"
    completed = ["id-000", "id-001"]
    remaining = ["id-002", "id-003"]
    all_ids = completed + remaining

    _write_minimal_existing_artifacts(output_dir, completed_ids=completed)

    client = _StubHttpClient(max_batch_size=4)
    renderer = _StubRenderer()
    cfg = _runner_config(output_dir, resume=True, workers=None)
    runner = BenchmarkRunner(client, renderer, cfg, scorer=lambda *_args: {})

    examples = [DatasetExample(sample_id=sid) for sid in all_ids]
    _run_with_optional_kwargs(runner, examples, resume=True, workers=None)

    # Resume should not re-request completed ids.
    assert set(client.requested_sample_ids) == set(remaining)

    # Artifacts should contain each sample_id exactly once.
    pred_ids = _read_jsonl_ids(output_dir / "predictions.jsonl")
    proc_ids = _read_jsonl_ids(output_dir / "processed_predictions.jsonl")
    scored_ids = _read_jsonl_ids(output_dir / "scored_predictions.jsonl")

    for ids in (pred_ids, proc_ids, scored_ids):
        assert sorted(ids) == sorted(all_ids)
        assert len(ids) == len(set(ids))


def test_workers_preserve_deterministic_artifact_order(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    ids = [f"id-{i:03d}" for i in range(12)]
    examples = [DatasetExample(sample_id=sid) for sid in reversed(ids)]

    client = _StubHttpClient(max_batch_size=3)
    renderer = _StubRenderer()
    cfg = _runner_config(output_dir, resume=False, workers=4)
    runner = BenchmarkRunner(client, renderer, cfg, scorer=lambda *_args: {})

    _run_with_optional_kwargs(runner, examples, resume=False, workers=4)

    # Regardless of input order and response order, artifacts must be written in
    # deterministic sample_id order.
    pred_ids = _read_jsonl_ids(output_dir / "predictions.jsonl")
    assert pred_ids == sorted(ids)


def test_run_from_cli_namespace_suite_with_resume_from_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suite runs may resume from a prior base output dir (I6-B5)."""
    import beans_next.runner.runner as runner_mod

    base_prior = tmp_path / "prior_suite_run"
    base_prior.mkdir()
    (base_prior / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": "beans_next.checkpoint.v1",
                "run_id": "recovered-run-id",
                "completed_sample_ids": [],
                "n_predictions_written": 0,
                "n_errors": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    class _FakeHttp:
        """Minimal HTTP client stub for suite resume CLI tests."""

        server_info = {"name": "stub", "schema_versions": ["predictions_v1"]}

        def __enter__(self) -> _FakeHttp:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
            return PredictionsV1Response(
                responses=[
                    PredictionsV1ResponseItem(
                        sample_id=item.sample_id,
                        predictions=["ok"],
                    )
                    for item in request.requests
                ]
            )

    monkeypatch.setattr(runner_mod, "HttpClient", lambda *_a, **_k: _FakeHttp())

    stub_audio = tmp_path / "stub.wav"
    stub_audio.write_bytes(b"")

    def _fake_load(
        _eval_task: Mapping[str, Any], *, args: Namespace
    ) -> list[DatasetExample]:
        return [
            DatasetExample(
                sample_id="resume-suite-1",
                labels="dog",
                metadata={"audio_path": str(stub_audio)},
            )
        ]

    monkeypatch.setattr(runner_mod, "_load_examples_for_eval_task", _fake_load)

    args = Namespace(
        config=None,
        predict_url="http://127.0.0.1:9/predict",
        run_id="ignored",
        workers=1,
        suite="beans_zero_smoke",
        resume=False,
        resume_from=base_prior,
        task_id=None,
        limit=1,
        output_dir=None,
        prompt_yaml=None,
        hf_path="EarthSpeciesProject/BEANS-Zero",
        hf_config="BEANS-Zero",
        split="test",
        dataset_name=None,
    )

    run_from_cli_namespace(args)

    suite_out = base_prior / "suite" / "beans_zero_smoke"
    assert suite_out.is_dir()
    for task in ("beans_zero_esc50", "beans_zero_enabirds", "beans_zero_captioning"):
        ck = suite_out / task / "checkpoint.json"
        assert ck.is_file(), f"missing checkpoint for {task}"

    suite_summary_path = suite_out / "suite_summary.json"
    assert suite_summary_path.is_file()
    suite_summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
    assert suite_summary.get("suite_id") == "beans_zero_smoke"


class TestRunnerUtils:
    """Unit tests for aggregate_score_means and per_task_score_means."""

    def test_aggregate_empty(self) -> None:
        from beans_next.runner._utils import aggregate_score_means

        assert aggregate_score_means([]) == {}

    def test_aggregate_averages_per_key(self) -> None:
        from beans_next.runner._utils import aggregate_score_means

        rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        result = aggregate_score_means(rows)
        assert result["a"] == pytest.approx(2.0)
        assert result["b"] == pytest.approx(3.0)

    def test_aggregate_skips_missing_keys(self) -> None:
        from beans_next.runner._utils import aggregate_score_means

        rows: list[Mapping[str, float]] = [{"a": 1.0}, {"b": 2.0}]
        result = aggregate_score_means(rows)
        assert result["a"] == pytest.approx(1.0)
        assert result["b"] == pytest.approx(2.0)

    def test_per_task_groups_by_task_id(self) -> None:
        from beans_next.runner._utils import per_task_score_means

        examples = [
            DatasetExample(sample_id="x1", task_id="t1"),
            DatasetExample(sample_id="x2", task_id="t2"),
            DatasetExample(sample_id="x3", task_id="t1"),
        ]
        rows: list[Mapping[str, float]] = [{"acc": 1.0}, {"acc": 0.0}, {"acc": 0.0}]
        result = per_task_score_means(examples, rows)
        assert result["t1"]["acc"] == pytest.approx(0.5)
        assert result["t2"]["acc"] == pytest.approx(0.0)

    def test_per_task_none_becomes_default(self) -> None:
        from beans_next.runner._utils import per_task_score_means

        examples = [DatasetExample(sample_id="x1")]
        rows: list[Mapping[str, float]] = [{"acc": 1.0}]
        result = per_task_score_means(examples, rows)
        assert "default" in result
        assert result["default"]["acc"] == pytest.approx(1.0)


def test_map_ordered_with_persistent_executor() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from beans_next.runner.parallel import map_ordered

    items = list(range(6))
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = map_ordered(lambda x: x * 2, items, workers=2, executor=pool)

    assert result == [x * 2 for x in items]
