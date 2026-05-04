"""Capture Phase B golden artifacts for a regression fixture bundle.

This tool replays the exact request items in ``inputs/requests.jsonl`` against a live
launcher ``POST /predict`` endpoint using :class:`beans_next.models.http.HttpClient`,
then writes canonical BEANS-Next artifacts under ``expected/``:

- ``predictions.jsonl``
- ``processed_predictions.jsonl``
- ``scored_predictions.jsonl``
- ``summary.json``
- ``model_identity.json``

It also updates ``manifest.yaml`` to mark the bundle as Phase B and to record the
launcher identity fields from ``GET /info``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

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


def _utc_now_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _package_version() -> str:
    try:
        return importlib.metadata.version("beans-next")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            raise SystemExit(f"{path}: blank line at {i}")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: invalid JSON at line {i}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SystemExit(f"{path}: expected JSON object at line {i}")
        rows.append(obj)
    if not rows:
        raise SystemExit(f"{path}: zero rows")
    return rows


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl_models(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        n = 0
        for row in rows:
            f.write(dumps_canonical(row.model_dump(mode="json")) + "\n")
            n += 1
    if n <= 0:
        raise SystemExit(f"{path}: wrote zero rows (invalid golden artifact)")


def _write_json(path: Path, obj: object) -> None:
    _write_text(path, dumps_canonical(obj) + "\n")


def _validate_manifest(manifest: dict[str, Any], *, bundle_dir: Path) -> None:
    required_top = [
        "fixture_format_version",
        "bundle_id",
        "phase",
        "model_identity",
        "inputs",
        "expected",
    ]
    for k in required_top:
        if k not in manifest:
            raise SystemExit(f"manifest.yaml missing required key: {k!r}")

    if str(manifest.get("fixture_format_version")) != "1":
        raise SystemExit("manifest.yaml: fixture_format_version must be '1'")

    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        raise SystemExit("manifest.yaml: bundle_id must be a non-empty string")
    if bundle_dir.name != bundle_id:
        raise SystemExit(
            "manifest.yaml: bundle_id "
            f"{bundle_id!r} does not match directory {bundle_dir.name!r}"
        )

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit("manifest.yaml: inputs must be a mapping")
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise SystemExit("manifest.yaml: expected must be a mapping")

    slice_path = inputs.get("slice_path")
    req_path = inputs.get("requests_path")
    if slice_path != "inputs/slice.json":
        raise SystemExit("manifest.yaml: inputs.slice_path must be 'inputs/slice.json'")
    if req_path != "inputs/requests.jsonl":
        raise SystemExit(
            "manifest.yaml: inputs.requests_path must be 'inputs/requests.jsonl'"
        )

    exp_paths = {
        "predictions_path": "expected/predictions.jsonl",
        "processed_predictions_path": "expected/processed_predictions.jsonl",
        "scored_predictions_path": "expected/scored_predictions.jsonl",
        "summary_path": "expected/summary.json",
        "model_identity_path": "expected/model_identity.json",
    }
    for k, v in exp_paths.items():
        if expected.get(k) != v:
            raise SystemExit(f"manifest.yaml: expected.{k} must be {v!r}")


def _dataset_examples_from_slice(
    slice_obj: dict[str, Any],
) -> dict[str, DatasetExample]:
    samples = slice_obj.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit("inputs/slice.json: missing or empty 'samples' list")

    out: dict[str, DatasetExample] = {}
    for i, item in enumerate(samples):
        if not isinstance(item, dict):
            raise SystemExit(f"inputs/slice.json: samples[{i}] must be an object")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise SystemExit(
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


def _raw_prediction_text(pred: ModelPrediction) -> str:
    return pred.predictions[0] if pred.predictions else ""


def _merge_row_error(pred: ModelPrediction, post_err: str | None) -> str | None:
    return pred.error or post_err


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


def _score_sample_fn() -> Callable[..., dict[str, float]] | None:
    try:
        import beans_next.metrics as metrics_mod  # noqa: PLC0415
    except Exception:
        return None
    fn = getattr(metrics_mod, "score_sample", None)
    return fn if callable(fn) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Path to a fixture bundle directory containing manifest.yaml",
    )
    parser.add_argument(
        "--predict-url",
        required=True,
        help="Full URL to POST /predict (e.g. http://127.0.0.1:8000/predict)",
    )
    parser.add_argument(
        "--out-variant-id",
        required=True,
        help="Variant id to store in manifest.yaml under expected.variant_id",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite expected/* outputs even if they are non-empty",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional run id stored in summary.json "
            "(default derives from out-variant-id)"
        ),
    )
    args = parser.parse_args(argv)

    bundle_dir: Path = args.bundle.expanduser().resolve()
    if not bundle_dir.is_dir():
        raise SystemExit(f"--bundle must be a directory: {bundle_dir}")
    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest.yaml: {manifest_path}")

    try:
        manifest_obj = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SystemExit(f"Invalid YAML ({manifest_path}): {exc}") from exc
    if not isinstance(manifest_obj, dict):
        raise SystemExit("manifest.yaml must be a YAML mapping")
    manifest = dict(manifest_obj)
    _validate_manifest(manifest, bundle_dir=bundle_dir)

    inputs_dir = bundle_dir / "inputs"
    expected_dir = bundle_dir / "expected"
    slice_path = inputs_dir / "slice.json"
    requests_path = inputs_dir / "requests.jsonl"

    if not slice_path.is_file():
        raise SystemExit(f"missing slice.json: {slice_path}")
    if not requests_path.is_file():
        raise SystemExit(f"missing requests.jsonl: {requests_path}")

    try:
        slice_obj = json.loads(slice_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{slice_path}: invalid JSON: {exc}") from exc
    if not isinstance(slice_obj, dict):
        raise SystemExit(f"{slice_path}: expected a JSON object")
    by_sample_id = _dataset_examples_from_slice(slice_obj)

    request_dicts = _read_jsonl(requests_path)
    wire_items: list[PredictionsV1RequestItem] = []
    for i, obj in enumerate(request_dicts):
        try:
            wire_items.append(PredictionsV1RequestItem.model_validate(obj))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"{requests_path}: invalid request item at line {i + 1}: {exc}"
            ) from exc
    wire_items = sorted(wire_items, key=lambda it: it.sample_id)

    examples: list[DatasetExample] = []
    for item in wire_items:
        ex = by_sample_id.get(item.sample_id)
        if ex is None:
            raise SystemExit(
                "inputs/requests.jsonl references a sample_id not present in "
                f"inputs/slice.json: {item.sample_id!r}"
            )
        examples.append(ex)

    if not args.force:
        nonempty: list[str] = []
        for rel in (
            "predictions.jsonl",
            "processed_predictions.jsonl",
            "scored_predictions.jsonl",
            "summary.json",
            "model_identity.json",
        ):
            p = expected_dir / rel
            if p.is_file() and p.stat().st_size > 0:
                nonempty.append(str(p))
        if nonempty:
            raise SystemExit(
                "Refusing to overwrite non-empty expected artifacts without --force: "
                f"{nonempty}"
            )

    run_id = (args.run_id or f"fixture_capture__{args.out_variant_id}").strip()
    if not run_id:
        raise SystemExit("--run-id resolved to empty string")

    with HttpClient(str(args.predict_url), probe_on_init=True) as client:
        info = client.server_info or {}
        for k in ("name", "model", "model_revision"):
            v = info.get(k)
            if not isinstance(v, str) or not v.strip():
                raise SystemExit(
                    f"GET /info missing required string field {k!r}: got {v!r}"
                )

        _write_json(expected_dir / "model_identity.json", dict(info))

        batch_size = effective_max_batch_size(info)
        parsed_steps, cleaner_steps = _postprocess_steps(examples)
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
                    raise SystemExit(
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
            raw_text = _raw_prediction_text(pred)
            post_err: str | None = None
            try:
                post = run_post_process_pipeline(
                    raw_text,
                    parser_steps=parsed_steps,
                    cleaner_steps=cleaner_steps,
                )
            except PostProcessPipelineError as exc:
                post = PostProcessResult(segments=[], text="", warnings=(str(exc),))
                post_err = str(exc)

            row_err = _merge_row_error(pred, post_err)
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
                try:
                    raw_scores = score_sample(
                        ex,
                        post=post,
                        raw_predictions=list(pred.predictions),
                    )
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(
                        f"Scoring failed for sample_id={ex.sample_id!r}: {exc}"
                    ) from exc
                scores = {
                    str(k): float(v)
                    for k, v in dict(raw_scores).items()
                    if isinstance(v, (int, float))
                }

            scored_row = processed_row.model_copy(update={"scores": scores or None})
            processed_out.append(processed_row)
            scored_out.append(scored_row)
            score_rows.append(scores)

        _write_jsonl_models(expected_dir / "predictions.jsonl", preds_out)
        _write_jsonl_models(expected_dir / "processed_predictions.jsonl", processed_out)
        _write_jsonl_models(expected_dir / "scored_predictions.jsonl", scored_out)

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
        _write_json(expected_dir / "summary.json", summary.model_dump(mode="json"))

    # Update manifest.yaml last (only after artifacts succeeded).
    captured_at = _utc_now_iso_z()
    manifest["phase"] = "phase_b_golden_captured"
    expected_block = dict(manifest.get("expected") or {})
    expected_block["variant_id"] = str(args.out_variant_id)
    manifest["expected"] = expected_block

    mi = dict(manifest.get("model_identity") or {})
    mi["source"] = "info_endpoint"
    mi["info"] = {
        "name": info["name"],
        "model": info["model"],
        "model_revision": info["model_revision"],
    }
    mi["info_captured_at_utc"] = captured_at
    manifest["model_identity"] = mi

    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    # Validate non-empty outputs (stop-on-error discipline).
    for p in (
        expected_dir / "predictions.jsonl",
        expected_dir / "processed_predictions.jsonl",
        expected_dir / "scored_predictions.jsonl",
    ):
        if not p.is_file() or p.stat().st_size <= 0:
            raise SystemExit(f"invalid golden artifact (empty): {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
