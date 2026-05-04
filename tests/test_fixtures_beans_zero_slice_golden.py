"""Golden regression test harness for fixture bundle `beans_zero_slice_v1`.

This test is intentionally **opt-in** and skips by default to keep the default
test suite fast and CPU-only.

Activation requires:
- bundle `manifest.yaml` is Phase B (`phase_b_golden_captured`)
- env `BEANS_PRO_RUN_GOLDEN_TESTS=1`
- env `BEANS_PRO_GOLDEN_PREDICT_URL` points to a live `POST /predict` endpoint
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest
import yaml

from beans_next.api.http_schemas import PredictionsV1Request, PredictionsV1RequestItem
from beans_next.api.types import (
    DatasetExample,
    ModelPrediction,
    RunSummary,
    ScoredPrediction,
)
from beans_next.models.http import HttpClient
from beans_next.post_process.pipeline import (
    PostProcessPipelineError,
    PostProcessResult,
    StepSpec,
    run_post_process_pipeline,
)
from beans_next.results.store import dumps_canonical
from beans_next.runner.batching import effective_max_batch_size, iter_batches
from beans_next.runner.runner import wire_response_item_to_model_prediction


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bundle_dir() -> Path:
    return _repo_root() / "tests" / "fixtures" / "beans_zero_slice_v1"


def _read_jsonl_dicts_nonempty(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            raise AssertionError(f"{path}: blank line at {i}")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path}: invalid JSON at line {i}: {exc}") from exc
        if not isinstance(obj, dict):
            raise AssertionError(f"{path}: expected JSON object at line {i}")
        rows.append(obj)
    if not rows:
        raise AssertionError(f"{path}: zero rows")
    return rows


def _write_jsonl_models(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_canonical(row.model_dump(mode="json")) + "\n")
            n += 1
    if n <= 0:
        raise AssertionError(f"{path}: wrote zero rows")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_canonical(obj) + "\n", encoding="utf-8")


def _canonical_jsonl_text(path: Path) -> str:
    lines = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            raise AssertionError(f"{path}: blank line at {i}")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path}: invalid JSON at line {i}: {exc}") from exc
        # Golden comparisons must be stable across runs. Some fields are inherently
        # runtime-derived (e.g. measured latency) and should not be compared bit-exact.
        if isinstance(obj, dict):
            obj.pop("latency_sec", None)
        lines.append(dumps_canonical(obj))
    if not lines:
        raise AssertionError(f"{path}: zero rows")
    return "\n".join(lines) + "\n"


def _canonical_json_text(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{path}: invalid JSON: {exc}") from exc
    return dumps_canonical(obj) + "\n"


def _dataset_examples_from_slice(
    slice_obj: dict[str, Any],
) -> dict[str, DatasetExample]:
    samples = slice_obj.get("samples")
    if not isinstance(samples, list) or not samples:
        raise AssertionError("inputs/slice.json: missing or empty 'samples' list")

    out: dict[str, DatasetExample] = {}
    for i, item in enumerate(samples):
        if not isinstance(item, dict):
            raise AssertionError(f"inputs/slice.json: samples[{i}] must be an object")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise AssertionError(
                f"inputs/slice.json: samples[{i}].sample_id must be a string"
            )
        task = item.get("task")
        task_id = None
        if isinstance(task, dict):
            tid = task.get("eval_task_id")
            if isinstance(tid, str) and tid.strip():
                task_id = tid.strip()
        labels = item.get("labels")
        out[sample_id] = DatasetExample(
            sample_id=sample_id,
            task_id=task_id,
            split=str(slice_obj.get("split") or "test"),
            labels=labels,
            metadata={},
        )
    return out


def _postprocess_steps(
    examples: list[DatasetExample],
) -> tuple[tuple[StepSpec, ...], tuple[StepSpec, ...]]:
    parsers = (StepSpec("parse_labels_comma", {}),)
    cleaners: list[StepSpec] = [
        StepSpec("normalize_whitespace", {}),
        StepSpec("strip_eos", {}),
    ]

    seen: set[str] = set()
    vocab: list[str] = []
    for ex in examples:
        labels = ex.labels
        if isinstance(labels, str) and labels.strip():
            for part in labels.split(","):
                tok = part.strip()
                if tok and tok not in seen:
                    seen.add(tok)
                    vocab.append(tok)
        elif isinstance(labels, list):
            for item in labels:
                if isinstance(item, str) and item.strip() and item not in seen:
                    seen.add(item)
                    vocab.append(item)
    if vocab:
        cleaners.append(StepSpec("fuzzy_match_to_labels", {"labels": tuple(vocab)}))
    return parsers, tuple(cleaners)


def _score_sample_fn() -> Callable[..., dict[str, float]] | None:
    try:
        import beans_next.metrics as metrics_mod  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(metrics_mod, "score_sample", None)
    return fn if callable(fn) else None


def _package_version() -> str:
    try:
        return importlib.metadata.version("beans-next")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _aggregate_score_means(score_rows: list[dict[str, float]]) -> dict[str, float]:
    if not score_rows:
        return {}
    keys: set[str] = set()
    for row in score_rows:
        keys.update(row.keys())
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals = [row[key] for row in score_rows if key in row]
        if vals:
            out[key] = sum(vals) / float(len(vals))
    return out


def _per_task_score_means(
    examples: list[DatasetExample],
    score_rows: list[dict[str, float]],
) -> dict[str, Any]:
    buckets: dict[str | None, list[dict[str, float]]] = {}
    for ex, scores in zip(examples, score_rows, strict=True):
        buckets.setdefault(ex.task_id, []).append(scores)
    out: dict[str, Any] = {}
    for tid, rows in sorted(
        buckets.items(),
        key=lambda kv: (kv[0] is None, kv[0] or ""),
    ):
        out[tid if tid is not None else "default"] = _aggregate_score_means(rows)
    return out


def _assert_nonempty_file(path: Path) -> None:
    assert path.is_file(), f"Missing file: {path}"
    assert path.stat().st_size > 0, f"Empty file: {path}"


def test_beans_zero_slice_v1_golden_regression(tmp_path: Path) -> None:
    bundle_dir = _bundle_dir()
    if not bundle_dir.is_dir():
        pytest.skip(f"Missing fixture bundle directory: {bundle_dir}")

    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.is_file():
        pytest.skip(f"Missing fixture manifest: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        pytest.skip("manifest.yaml root must be a mapping")

    phase = str(manifest.get("phase") or "")
    if phase != "phase_b_golden_captured":
        pytest.skip(
            "Golden regression test requires Phase B goldens. "
            f"manifest.phase={phase!r} (expected 'phase_b_golden_captured')."
        )

    if os.environ.get("BEANS_PRO_RUN_GOLDEN_TESTS") != "1":
        pytest.skip(
            "Golden regression test is opt-in. Set BEANS_PRO_RUN_GOLDEN_TESTS=1 to run."
        )

    predict_url = os.environ.get("BEANS_PRO_GOLDEN_PREDICT_URL")
    if not predict_url:
        pytest.skip(
            "Golden regression test requires BEANS_PRO_GOLDEN_PREDICT_URL "
            "(full URL to POST /predict)."
        )

    inputs = manifest.get("inputs")
    expected = manifest.get("expected")
    if not isinstance(inputs, dict) or not isinstance(expected, dict):
        raise AssertionError("manifest.yaml: inputs and expected must be mappings")

    req_rel = inputs.get("requests_path")
    slice_rel = inputs.get("slice_path")
    if req_rel != "inputs/requests.jsonl":
        raise AssertionError(
            "manifest.yaml: inputs.requests_path must be 'inputs/requests.jsonl'"
        )
    if slice_rel != "inputs/slice.json":
        raise AssertionError(
            "manifest.yaml: inputs.slice_path must be 'inputs/slice.json'"
        )

    expected_paths = {
        "predictions.jsonl": expected.get("predictions_path"),
        "processed_predictions.jsonl": expected.get("processed_predictions_path"),
        "scored_predictions.jsonl": expected.get("scored_predictions_path"),
        "summary.json": expected.get("summary_path"),
        "model_identity.json": expected.get("model_identity_path"),
    }
    for name, rel in expected_paths.items():
        if not isinstance(rel, str) or not rel:
            raise AssertionError(f"manifest.yaml: expected path for {name} is missing")
        _assert_nonempty_file(bundle_dir / rel)

    requests_path = bundle_dir / "inputs" / "requests.jsonl"
    slice_path = bundle_dir / "inputs" / "slice.json"
    assert requests_path.is_file(), f"Missing requests.jsonl: {requests_path}"
    assert slice_path.is_file(), f"Missing slice.json: {slice_path}"

    slice_obj = json.loads(slice_path.read_text(encoding="utf-8"))
    assert isinstance(slice_obj, dict), f"{slice_path}: expected a JSON object"
    by_sample_id = _dataset_examples_from_slice(slice_obj)

    request_dicts = _read_jsonl_dicts_nonempty(requests_path)
    wire_items: list[PredictionsV1RequestItem] = []
    for i, obj in enumerate(request_dicts, start=1):
        try:
            wire_items.append(PredictionsV1RequestItem.model_validate(obj))
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"{requests_path}: invalid request item at line {i}: {exc}"
            ) from exc
    wire_items = sorted(wire_items, key=lambda it: it.sample_id)

    examples: list[DatasetExample] = []
    for item in wire_items:
        ex = by_sample_id.get(item.sample_id)
        if ex is None:
            raise AssertionError(
                "inputs/requests.jsonl references a sample_id not present in "
                f"inputs/slice.json: {item.sample_id!r}"
            )
        examples.append(ex)

    out_dir = tmp_path / "golden_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_variant_id = str(expected.get("variant_id") or "").strip()
    if not expected_variant_id:
        raise AssertionError(
            "manifest.yaml: expected.variant_id must be a non-empty string"
        )
    run_id = f"fixture_capture__{expected_variant_id}"

    with HttpClient(str(predict_url), probe_on_init=True) as client:
        info = client.server_info or {}
        for k in ("name", "model", "model_revision"):
            v = info.get(k)
            if not isinstance(v, str) or not v.strip():
                raise AssertionError(
                    f"GET /info missing required string field {k!r}: got {v!r}"
                )

        _write_json(out_dir / "model_identity.json", dict(info))

        batch_size = effective_max_batch_size(info)
        parser_steps, cleaner_steps = _postprocess_steps(examples)
        score_sample = _score_sample_fn()

        preds_out: list[ModelPrediction] = []
        processed_out: list[ScoredPrediction] = []
        scored_out: list[ScoredPrediction] = []
        score_rows: list[dict[str, float]] = []
        n_errors = 0

        for batch in iter_batches(wire_items, batch_size):
            envelope = PredictionsV1Request(requests=list(batch))
            response = client.generate(envelope)
            for resp_item in response.responses:
                if resp_item.error is not None:
                    raise AssertionError(
                        "Launcher returned a per-sample error; reproduce on this "
                        "single request item. "
                        f"sample_id={resp_item.sample_id!r} error={resp_item.error!r}"
                    )
                pred = wire_response_item_to_model_prediction(
                    resp_item, server_info=client.server_info
                )
                preds_out.append(pred)

        preds_out = sorted(preds_out, key=lambda p: p.sample_id)
        pred_by_id = {p.sample_id: p for p in preds_out}

        for ex in sorted(examples, key=lambda e: e.sample_id):
            pred = pred_by_id[ex.sample_id]
            raw_text = pred.predictions[0] if pred.predictions else ""
            post_err: str | None = None
            try:
                post = run_post_process_pipeline(
                    raw_text, parser_steps=parser_steps, cleaner_steps=cleaner_steps
                )
            except PostProcessPipelineError as exc:
                post = PostProcessResult(segments=[], text="", warnings=(str(exc),))
                post_err = str(exc)

            row_err = pred.error or post_err
            processed_row = ScoredPrediction(
                sample_id=ex.sample_id,
                task_id=ex.task_id,
                predictions=list(pred.predictions),
                processed_prediction=post.text,
                targets=ex.labels,
                scores=None,
                postprocess_version=None,
                error=row_err,
            )

            if row_err is not None:
                scores: dict[str, float] = {}
                n_errors += 1
            elif score_sample is None:
                scores = {}
            else:
                raw_scores = score_sample(
                    ex,
                    post=post,
                    raw_predictions=list(pred.predictions),
                )
                scores = {
                    str(k): float(v)
                    for k, v in dict(raw_scores).items()
                    if isinstance(v, (int, float))
                }

            scored_row = processed_row.model_copy(update={"scores": scores or None})
            processed_out.append(processed_row)
            scored_out.append(scored_row)
            score_rows.append(scores)

        _write_jsonl_models(out_dir / "predictions.jsonl", preds_out)
        _write_jsonl_models(out_dir / "processed_predictions.jsonl", processed_out)
        _write_jsonl_models(out_dir / "scored_predictions.jsonl", scored_out)

        summary = RunSummary(
            run_id=run_id,
            library_version=_package_version(),
            code_git_sha=None,
            run_config_hash=None,
            prompt_version=None,
            postprocess_version=None,
            scorer_versions=None,
            model_identity=dict(info),
            seed=None,
            n_samples=len(examples),
            n_errors=n_errors,
            metrics={
                "mean": _aggregate_score_means(score_rows),
                "per_task_mean": _per_task_score_means(
                    sorted(examples, key=lambda e: e.sample_id),
                    score_rows,
                ),
            },
            task_results=None,
        )
        _write_json(out_dir / "summary.json", summary.model_dump(mode="json"))

    # Enforce artifact non-emptiness (stop-on-error discipline).
    for name in (
        "predictions.jsonl",
        "processed_predictions.jsonl",
        "scored_predictions.jsonl",
        "summary.json",
        "model_identity.json",
    ):
        _assert_nonempty_file(out_dir / name)

    # Strict comparison: canonical JSONL/JSON equality.
    # Normalizes whitespace + key order to avoid diff noise.
    comparisons = [
        ("predictions.jsonl", "jsonl"),
        ("processed_predictions.jsonl", "jsonl"),
        ("scored_predictions.jsonl", "jsonl"),
        ("summary.json", "json"),
        ("model_identity.json", "json"),
    ]
    for name, kind in comparisons:
        expected_path = bundle_dir / str(expected_paths[name])
        actual_path = out_dir / name
        if kind == "jsonl":
            assert _canonical_jsonl_text(actual_path) == _canonical_jsonl_text(
                expected_path
            ), f"Mismatch in {name}"
        else:
            assert _canonical_json_text(actual_path) == _canonical_json_text(
                expected_path
            ), f"Mismatch in {name}"
