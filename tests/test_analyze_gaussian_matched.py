"""Tests for the matched Gaussian-noise analysis utility."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def analysis_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "analyze_gaussian_matched.py"
    spec = importlib.util.spec_from_file_location("analyze_gaussian_matched", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_arm(
    root: Path,
    *,
    arm: str,
    rows: list[dict[str, object]],
    metrics: dict[str, float],
    task_id: str,
) -> Path:
    task_dir = root / arm / "suite" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "scored_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (task_dir / "summary.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "n_samples": len(rows),
                "n_errors": sum(row.get("error") not in (None, "") for row in rows),
                "metrics": {"mean": metrics, "per_task_mean": {task_id: metrics}},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_pair_reports_per_task_deltas_and_diagnostics(
    analysis_module: ModuleType, tmp_path: Path
) -> None:
    task_id = "beans_next_numeric_demo"
    real_rows = [
        {
            "sample_id": "a",
            "targets": "1 Hz",
            "error": None,
            "processed_prediction": "1 Hz",
            "predictions": ["1 Hz"],
            "scores": {"absolute_error": 0.0, "numeric_parse_success": 1.0},
        },
        {
            "sample_id": "b",
            "targets": "2 Hz",
            "error": None,
            "processed_prediction": "",
            "predictions": ["I cannot determine the frequency."],
            "scores": {"absolute_error": 2.0, "numeric_parse_success": 0.0},
        },
        {
            "sample_id": "c",
            "targets": "3 Hz",
            "error": "paired failure",
            "predictions": [],
            "scores": None,
        },
    ]
    noise_rows = [
        {**row, "scores": {"absolute_error": 1.0, "numeric_parse_success": 1.0}}
        for row in real_rows[:2]
    ] + [{**real_rows[2], "error": "paired failure"}]
    real = _write_arm(
        tmp_path / "real",
        arm="audio",
        rows=real_rows,
        metrics={"absolute_error": 1.0},
        task_id=task_id,
    )
    noise = _write_arm(
        tmp_path / "noise",
        arm="noise",
        rows=noise_rows,
        metrics={"absolute_error": 2.0},
        task_id=task_id,
    )

    report = analysis_module.analyze_pair(real, noise)
    task = report["tasks"][task_id]
    assert task["metrics"]["absolute_error"] == {
        "real": 1.0,
        "noise": 2.0,
        "audio_minus_noise": -1.0,
    }
    assert task["real_diagnostics"]["numeric_parse_denominator"] == 2
    assert task["real_diagnostics"]["numeric_parse_failure_count"] == 1
    assert task["real_diagnostics"]["numeric_parse_failure_rate"] == pytest.approx(0.5)
    assert task["real_diagnostics"]["refusal_candidate_sample_ids"] == ["b"]
    assert task["real_diagnostics"]["metric_denominators"]["absolute_error"] == 2
    assert report["paired_error_count"] == 1
    assert report["successful_real_samples"] == 2


def test_frequency_iou_uses_full_denominator_and_missing_is_zero(
    analysis_module: ModuleType, tmp_path: Path
) -> None:
    task_id = "beans_next_t3_frequency_range_description"
    real_rows = [
        {
            "sample_id": "a",
            "targets": "x",
            "error": None,
            "scores": {"freq_mean_iou": 1.0},
        },
        {"sample_id": "b", "targets": "x", "error": None, "scores": None},
        {
            "sample_id": "c",
            "targets": "x",
            "error": None,
            "scores": {"freq_mean_iou": 0.5},
        },
    ]
    noise_rows = [
        {
            "sample_id": "a",
            "targets": "x",
            "error": None,
            "scores": {"freq_mean_iou": 0.0},
        },
        {
            "sample_id": "b",
            "targets": "x",
            "error": None,
            "scores": {"freq_mean_iou": 1.0},
        },
        {"sample_id": "c", "targets": "x", "error": None, "scores": None},
    ]
    real = _write_arm(
        tmp_path / "real",
        arm="audio",
        rows=real_rows,
        metrics={"freq_mean_iou": 0.7},
        task_id=task_id,
    )
    noise = _write_arm(
        tmp_path / "noise",
        arm="noise",
        rows=noise_rows,
        metrics={"freq_mean_iou": 0.3},
        task_id=task_id,
    )

    task = analysis_module.analyze_pair(real, noise)["tasks"][task_id]
    assert task["coverage_denominator"] == 3
    assert task["metrics"]["coverage_aware_freq_mean_iou"]["real"] == pytest.approx(0.5)
    assert task["metrics"]["coverage_aware_freq_mean_iou"]["noise"] == pytest.approx(
        1 / 3
    )


def test_alignment_mismatch_is_rejected(
    analysis_module: ModuleType, tmp_path: Path
) -> None:
    task_id = "beans_next_demo"
    rows = [
        {
            "sample_id": "a",
            "targets": "x",
            "error": None,
            "scores": {"accuracy": 1.0},
        }
    ]
    real = _write_arm(
        tmp_path / "real",
        arm="audio",
        rows=rows,
        metrics={"accuracy": 1.0},
        task_id=task_id,
    )
    noise = _write_arm(
        tmp_path / "noise",
        arm="noise",
        rows=[{**rows[0], "targets": "different"}],
        metrics={"accuracy": 1.0},
        task_id=task_id,
    )
    with pytest.raises(
        analysis_module.MatchedAnalysisError, match="sample/target mismatch"
    ):
        analysis_module.analyze_pair(real, noise)
