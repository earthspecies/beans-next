"""CPU-only validation for Increment-7 fixture slice bundle.

This test file is intentionally high-signal and fast:
- validates the on-disk fixture bundle structure and manifest invariants
- runs a minimal end-to-end execution against the dummy launcher over real HTTP

It owns the dummy server lifecycle via `scripts/with_uvicorn.py` as required by
`AGENT_SPEC.md` testing discipline.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bundle_dir() -> Path:
    return _repo_root() / "tests" / "fixtures" / "beans_zero_slice_v1"


def _read_jsonl_nonempty(path: Path) -> list[object]:
    raw = path.read_text(encoding="utf-8").splitlines()
    rows: list[object] = []
    for i, line in enumerate(raw, start=1):
        if not line.strip():
            raise AssertionError(f"Empty JSONL line at {path}:{i}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Invalid JSON at {path}:{i}: {exc}") from exc
    if not rows:
        raise AssertionError(f"JSONL file is empty: {path}")
    return rows


@pytest.fixture(scope="session")
def beans_zero_slice_bundle() -> Path:
    bundle = _bundle_dir()
    if not bundle.is_dir():
        pytest.skip(
            "Fixture bundle is missing. Expected directory: "
            f"{bundle}. (Owned by I7-A2.)"
        )
    manifest_path = bundle / "manifest.yaml"
    if not manifest_path.is_file():
        pytest.skip(
            "Fixture bundle is incomplete (missing manifest). Expected file: "
            f"{manifest_path}. (Owned by I7-A2.)"
        )
    return bundle


def test_fixture_manifest_and_inputs_sanity(beans_zero_slice_bundle: Path) -> None:
    manifest_path = beans_zero_slice_bundle / "manifest.yaml"
    assert manifest_path.is_file(), f"Missing manifest: {manifest_path}"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), "manifest.yaml root must be a mapping"

    required_top = {
        "fixture_format_version",
        "bundle_id",
        "created_at_utc",
        "description",
        "phase",
        "model_identity",
        "inputs",
        "expected",
        "regenerate",
    }
    missing = sorted(required_top - set(manifest))
    assert not missing, f"manifest.yaml missing keys: {missing}"

    bundle_id = manifest.get("bundle_id")
    assert bundle_id == beans_zero_slice_bundle.name, (
        "manifest.yaml bundle_id must match directory name. "
        f"bundle_id={bundle_id!r} dir={beans_zero_slice_bundle.name!r}"
    )

    inputs = manifest["inputs"]
    assert isinstance(inputs, dict), "manifest.yaml inputs must be a mapping"
    slice_rel = inputs.get("slice_path")
    req_rel = inputs.get("requests_path")
    assert slice_rel == "inputs/slice.json"
    assert req_rel == "inputs/requests.jsonl"

    slice_path = beans_zero_slice_bundle / str(slice_rel)
    req_path = beans_zero_slice_bundle / str(req_rel)
    assert slice_path.is_file(), f"Missing slice.json: {slice_path}"
    assert req_path.is_file(), f"Missing requests.jsonl: {req_path}"

    # Structural check: `requests.jsonl` must be non-empty and valid JSONL.
    req_rows = _read_jsonl_nonempty(req_path)
    assert all(isinstance(r, dict) for r in req_rows)

    # Structural check: slice.json must parse as JSON and declare at least one sample.
    slice_obj = json.loads(slice_path.read_text(encoding="utf-8"))
    assert isinstance(slice_obj, dict), "inputs/slice.json must be a JSON object"
    samples = slice_obj.get("samples")
    assert isinstance(samples, list) and samples, (
        "slice.json must define non-empty samples"
    )
    for i, sample in enumerate(samples):
        assert isinstance(sample, dict), f"slice.json samples[{i}] must be an object"
        sid = sample.get("sample_id")
        assert isinstance(sid, str) and sid.strip(), (
            f"Missing sample_id at samples[{i}]"
        )

    # If Phase B goldens exist and include JSONL, they must not be empty.
    phase = str(manifest.get("phase"))
    expected = manifest["expected"]
    assert isinstance(expected, dict), "manifest.yaml expected must be a mapping"
    expected_paths = [
        expected.get("predictions_path"),
        expected.get("processed_predictions_path"),
        expected.get("scored_predictions_path"),
    ]
    if phase == "phase_b_golden_captured":
        for rel in expected_paths:
            assert isinstance(rel, str) and rel, (
                "expected.*_path must be non-empty strings"
            )
            p = beans_zero_slice_bundle / rel
            assert p.is_file(), f"Missing expected artifact: {p}"
            _read_jsonl_nonempty(p)


def test_fixture_minimal_end_to_end_against_dummy_launcher(
    beans_zero_slice_bundle: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "fixture_run_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["BEANS_PRO_FIXTURE_DIR"] = str(beans_zero_slice_bundle)
    env["BEANS_PRO_OUTPUT_DIR"] = str(out_dir)
    env["BEANS_PRO_PREDICT_URL"] = "http://127.0.0.1:19091/predict"

    # Own the dummy launcher lifecycle via the wrapper.
    # Run the harness as the child command.
    harness = textwrap.dedent(
        r"""
        import json
        import os
        from pathlib import Path

        import yaml

        from beans_next.api.http_schemas import (
            PredictionsV1Request,
            PredictionsV1RequestItem,
        )
        from beans_next.api.types import DatasetExample, RunSummary, ScoredPrediction
        from beans_next.models.http import HttpClient
        from beans_next.post_process.pipeline import run_post_process_pipeline
        from beans_next.results.store import BenchmarkArtifactWriter

        try:
            from beans_next.metrics import score_sample
        except Exception:  # noqa: BLE001
            score_sample = None

        bundle = Path(os.environ["BEANS_PRO_FIXTURE_DIR"]).resolve()
        out_dir = Path(os.environ["BEANS_PRO_OUTPUT_DIR"]).resolve()
        predict_url = os.environ["BEANS_PRO_PREDICT_URL"]

        manifest = yaml.safe_load(
            (bundle / "manifest.yaml").read_text(encoding="utf-8")
        )
        req_path = bundle / manifest["inputs"]["requests_path"]
        slice_path = bundle / manifest["inputs"]["slice_path"]

        by_id = {}
        slice_obj = json.loads(slice_path.read_text(encoding="utf-8"))
        for item in slice_obj.get("samples", []):
            if isinstance(item, dict) and isinstance(item.get("sample_id"), str):
                sid = item["sample_id"]
                task = item.get("task") if isinstance(item.get("task"), dict) else {}
                eval_task_id = task.get("eval_task_id")
                task_id = eval_task_id if isinstance(eval_task_id, str) else None
                labels = item.get("labels") if "labels" in item else item.get("targets")
                by_id[sid] = DatasetExample(
                    sample_id=sid,
                    task_id=task_id,
                    split=(
                        slice_obj.get("split")
                        if isinstance(slice_obj.get("split"), str)
                        else None
                    ),
                    labels=labels,
                    metadata=(
                        item.get("source")
                        if isinstance(item.get("source"), dict)
                        else {}
                    ),
                )

        req_items = []
        for line in req_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            req_items.append(PredictionsV1RequestItem.model_validate(obj))
        if not req_items:
            raise SystemExit(f"inputs/requests.jsonl is empty: {req_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        with HttpClient(predict_url, probe_on_init=True) as client:
            resp = client.generate(PredictionsV1Request(requests=req_items))
            resp_by_id = {r.sample_id: r for r in resp.responses}

            n_errors = 0
            score_rows = []
            with BenchmarkArtifactWriter(out_dir) as writer:
                for req in req_items:
                    item = resp_by_id.get(req.sample_id)
                    if item is None:
                        raise SystemExit(
                            f"Missing response for sample_id={req.sample_id!r}"
                        )
                    pred = item.model_dump(mode="json")
                    # Map to core-ish shape.
                    # Roundtrip through artifact-writer-compatible types.
                    # Use runner's artifact schemas via explicit Pydantic construction.
                    from beans_next.runner.runner import (
                        wire_response_item_to_model_prediction,
                    )

                    mp = wire_response_item_to_model_prediction(
                        item,
                        server_info=client.server_info,
                    )
                    writer.append_prediction_record(mp)

                    raw_text = mp.predictions[0] if mp.predictions else ""
                    post = run_post_process_pipeline(
                        raw_text, parser_steps=(), cleaner_steps=()
                    )
                    ex = by_id.get(
                        req.sample_id, DatasetExample(sample_id=req.sample_id)
                    )
                    processed = ScoredPrediction(
                        sample_id=req.sample_id,
                        task_id=ex.task_id,
                        predictions=list(mp.predictions),
                        processed_prediction=post.text,
                        targets=ex.labels,
                        scores=None,
                        postprocess_version=None,
                        error=mp.error,
                    )
                    writer.append_processed_prediction(processed)

                    scores = {}
                    if mp.error is None and score_sample is not None:
                        try:
                            scores = dict(
                                score_sample(
                                    ex,
                                    post=post,
                                    raw_predictions=list(mp.predictions),
                                )
                            )
                        except Exception:  # noqa: BLE001
                            scores = {}
                    scored = processed.model_copy(update={"scores": (scores or None)})
                    writer.append_scored_prediction(scored)
                    score_rows.append(scores)
                    if mp.error is not None:
                        n_errors += 1

                # Minimal summary: ensure required fields exist; structure can evolve.
                means = {}
                keys = sorted({k for row in score_rows for k in row.keys()})
                for k in keys:
                    vals = [row[k] for row in score_rows if k in row]
                    if vals:
                        means[k] = sum(vals) / float(len(vals))
                summary = RunSummary(
                    run_id="beans_zero_slice_fixture_dummy",
                    library_version="0.0.0",
                    model_identity=dict(client.server_info or {}),
                    n_samples=len(req_items),
                    n_errors=n_errors,
                    metrics={"mean": means},
                )
                writer.write_summary(summary)
                writer.write_model_identity(dict(client.server_info or {}))

        required = [
            "predictions.jsonl",
            "processed_predictions.jsonl",
            "scored_predictions.jsonl",
            "summary.json",
            "model_identity.json",
        ]
        for name in required:
            p = out_dir / name
            if not p.is_file():
                raise SystemExit(f"Missing artifact: {p}")
            if p.stat().st_size <= 0:
                raise SystemExit(f"Empty artifact: {p}")
        """
    ).strip()

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/with_uvicorn.py",
        "--cwd",
        "examples/servers/dummy",
        "--app",
        "serve:app",
        "--host",
        "127.0.0.1",
        "--port",
        "19091",
        "--",
        "uv",
        "run",
        "python",
        "-c",
        harness,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-50:])
        err_tail = "\n".join((proc.stderr or "").splitlines()[-50:])
        raise AssertionError(
            "End-to-end fixture run failed.\n"
            f"returncode={proc.returncode}\n"
            "--- stdout (tail) ---\n"
            f"{tail}\n"
            "--- stderr (tail) ---\n"
            f"{err_tail}\n"
        )

    # Validate artifacts exist and are non-empty (the harness checks this too).
    for name in (
        "predictions.jsonl",
        "processed_predictions.jsonl",
        "scored_predictions.jsonl",
        "summary.json",
        "model_identity.json",
    ):
        p = out_dir / name
        assert p.is_file(), f"Missing artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact: {p}"
