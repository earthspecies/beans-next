"""Integration-style tests for `beans-next score-from-file` (offline, CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beans_next.api.types import ModelPrediction, ScoredPrediction
from beans_next.cli import main


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def test_score_from_file_writes_artifacts_and_scores_classification(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    preds_path = run_dir / "predictions.jsonl"
    proc_path = run_dir / "processed_predictions.jsonl"
    out_dir = tmp_path / "rescored"

    preds = [
        ModelPrediction(
            sample_id="s1",
            predictions=["dog"],
            error=None,
            server_info={"name": "dummy"},
        ).model_dump(mode="json"),
        ModelPrediction(
            sample_id="s2",
            predictions=["dog"],
            error=None,
            server_info={"name": "dummy"},
        ).model_dump(mode="json"),
    ]
    _write_jsonl(preds_path, preds)

    proc_rows = [
        ScoredPrediction(
            sample_id="s1",
            task_id="t1",
            predictions=["dog"],
            processed_prediction="dog",
            targets="dog",
            scores=None,
            error=None,
        ).model_dump(mode="json"),
        ScoredPrediction(
            sample_id="s2",
            task_id="t1",
            predictions=["dog"],
            processed_prediction="dog",
            targets="cat",
            scores=None,
            error=None,
        ).model_dump(mode="json"),
    ]
    _write_jsonl(proc_path, proc_rows)

    rc = main(["score-from-file", str(preds_path), "-o", str(out_dir)])
    assert rc == 0

    assert (out_dir / "processed_predictions.jsonl").is_file()
    assert (out_dir / "scored_predictions.jsonl").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "model_identity.json").is_file()

    scored = _read_jsonl(out_dir / "scored_predictions.jsonl")
    by_id = {row["sample_id"]: row for row in scored}
    assert by_id["s1"]["scores"]["accuracy"] == pytest.approx(1.0)
    assert by_id["s2"]["scores"]["accuracy"] == pytest.approx(0.0)

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_samples"] == 2
    assert summary["n_errors"] == 0
    assert summary["metrics"]["mean"]["accuracy"] == pytest.approx(0.5)


def test_score_from_file_scores_multilabel_average_precision(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    preds_path = run_dir / "predictions.jsonl"
    proc_path = run_dir / "processed_predictions.jsonl"

    preds = [
        ModelPrediction(sample_id="s1", predictions=["a, c"], error=None).model_dump(
            mode="json"
        ),
        ModelPrediction(sample_id="s2", predictions=["b"], error=None).model_dump(
            mode="json"
        ),
    ]
    _write_jsonl(preds_path, preds)

    proc_rows = [
        ScoredPrediction(
            sample_id="s1",
            task_id="t1",
            predictions=["a, c"],
            processed_prediction="a, c",
            targets=["a", "b"],
            scores=None,
            error=None,
        ).model_dump(mode="json"),
        ScoredPrediction(
            sample_id="s2",
            task_id="t1",
            predictions=["b"],
            processed_prediction="b",
            targets=["a", "b"],
            scores=None,
            error=None,
        ).model_dump(mode="json"),
    ]
    _write_jsonl(proc_path, proc_rows)

    out_dir = tmp_path / "rescored"
    rc = main(["score-from-file", str(preds_path), "-o", str(out_dir)])
    assert rc == 0

    scored = _read_jsonl(out_dir / "scored_predictions.jsonl")
    ap_vals = [row["scores"]["average_precision"] for row in scored]
    assert all(isinstance(v, (int, float)) for v in ap_vals)
