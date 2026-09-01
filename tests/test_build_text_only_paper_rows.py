"""Focused tests for the text-only paper-row export utility."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def row_builder() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "build_text_only_paper_rows.py"
    )
    spec = importlib.util.spec_from_file_location("build_text_only_paper_rows", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _task(
    task_id: str,
    mean: dict[str, float],
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    task: dict[str, object] = {
        "eval_task_id": task_id,
        "summary": {"metrics": {"mean": mean}},
    }
    if output_dir is not None:
        task["output_dir"] = str(output_dir)
    return task


def _write_fixture(tmp_path: Path) -> Path:
    freq_dir = tmp_path / "frequency"
    freq_dir.mkdir()
    (freq_dir / "scored_predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"scores": {"freq_mean_iou": 0.8}}),
                json.dumps({"scores": None}),
                json.dumps({"scores": {"freq_mean_iou": 0.4}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tasks: list[dict[str, object]] = [
        _task("beans_zero_cbi", {"accuracy": 0.0123}),
        _task("beans_zero_humbugdb", {"accuracy": 0.0456}),
        _task("beans_zero_esc50", {"accuracy": 0.0789}),
        _task("beans_zero_call_type", {"accuracy": 0.5}),
        _task("beans_zero_lifestage", {"accuracy": 0.25}),
        _task("beans_zero_zf_indiv", {"accuracy": 0.75}),
        _task("beans_next_t1_caption", {"cider": 0.012}),
        _task("beans_next_t1_description_mcq", {"top1_accuracy": 0.345}),
        _task("beans_next_t1_snr_regression", {"absolute_error": 3.21}),
        _task("beans_next_bird_presence", {"top1_accuracy": 0.5}),
        _task("beans_next_call_type_fixed_vocab", {"macro_f1": 0.25}),
        _task("beans_next_t2_behavior", {"top1_accuracy": 0.75}),
        _task("beans_next_t2_captioning", {"cider": 0.007}),
        _task(
            "beans_next_t3_species_count_oe",
            {"absolute_error": 1.25},
        ),
        _task(
            "beans_next_t3_vocalization_count_per_species_oe",
            {"count_mae": 2.5},
        ),
        _task(
            "beans_next_t3_species_by_vocalization_order_oe",
            {"top1_accuracy": 0.33},
        ),
        _task(
            "beans_next_t3_species_by_vocalization_order_mcq",
            {"top1_accuracy": 0.22},
        ),
        _task(
            "beans_next_t3_frequency_range_description",
            # This matched-only summary value must not be used.
            {"freq_mean_iou": 0.99},
            output_dir=freq_dir,
        ),
        _task("beans_next_t3_ordered_species_summary", {"species_f1": 0.5}),
        _task("beans_next_t3_structural_captioning", {"cider": 0.03}),
        _task(
            "beans_next_gibbon_fewshot_detection_balanced",
            {"accuracy": 0.456, "macro_f1": 0.123},
        ),
    ]
    path = tmp_path / "suite_summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "beans_next.suite_summary.v1",
                "suite_id": "fixture",
                "eval_tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_rows_scales_units_and_preserves_missing_tasks(
    row_builder: ModuleType,
    tmp_path: Path,
) -> None:
    result = row_builder.build_paper_rows(_write_fixture(tmp_path), "Test model")

    beans_zero = result["tables"]["beans_zero"]
    assert beans_zero["values"]["cbi"] == pytest.approx(0.0123)
    assert beans_zero["values"]["humbugdb"] == pytest.approx(0.0456)
    assert beans_zero["values"]["individual_count"] == pytest.approx(0.75)

    acoustic = result["tables"]["acoustic_v2"]
    assert acoustic["values"]["caption"] == pytest.approx(1.2)
    assert acoustic["values"]["description_mcq"] == pytest.approx(34.5)
    assert acoustic["values"]["snr_regression"] == pytest.approx(3.21)

    semantic = result["tables"]["semantic_v2"]
    assert semantic["values"]["bird"] == pytest.approx(50.0)
    assert semantic["values"]["fixed"] == pytest.approx(25.0)
    assert semantic["values"]["behavior"] == pytest.approx(75.0)
    assert semantic["values"]["caption"] == pytest.approx(0.007)
    assert semantic["values"]["insect"] is None
    assert semantic["values"]["begging"] is None
    assert "--" in semantic["latex_row"]

    structural = result["tables"]["structural_v3"]
    assert structural["values"]["species_count"] == pytest.approx(1.25)
    assert structural["values"]["frequency"] == pytest.approx(40.0)
    assert structural["values"]["summary"] == pytest.approx(50.0)
    assert structural["values"]["caption"] == pytest.approx(3.0)
    assert (
        structural["subtables"]["species_id"]["values"]["order_oe"]
        == pytest.approx(33.0)
    )
    assert (
        structural["subtables"]["species_id"]["values"]["order_mcq"]
        == pytest.approx(22.0)
    )

    assert result["tables"]["tier4"]["values"]["gibbons"] == pytest.approx(0.123)
    # The result must be directly machine-readable JSON.
    json.dumps(result)


def test_frequency_mean_counts_rows_without_scores_as_zero(
    row_builder: ModuleType,
    tmp_path: Path,
) -> None:
    summary_path = _write_fixture(tmp_path)
    scored_path = tmp_path / "frequency" / "scored_predictions.jsonl"
    scored_path.write_text(
        json.dumps({"scores": {"freq_mean_iou": 0.5}})
        + "\n"
        + json.dumps({"scores": {}})
        + "\n",
        encoding="utf-8",
    )
    result = row_builder.build_rows(summary_path, "Test model")
    assert result["tables"]["structural_v3"]["values"]["frequency"] == pytest.approx(
        25.0
    )


def test_frequency_prefers_portable_sibling_over_stale_output_dir(
    row_builder: ModuleType,
    tmp_path: Path,
) -> None:
    summary_path = _write_fixture(tmp_path)
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    frequency = next(
        task
        for task in raw["eval_tasks"]
        if task["eval_task_id"] == "beans_next_t3_frequency_range_description"
    )
    portable = summary_path.parent / "beans_next_t3_frequency_range_description"
    portable.mkdir()
    (portable / "scored_predictions.jsonl").write_text(
        json.dumps({"scores": {"freq_mean_iou": 0.25}}) + "\n",
        encoding="utf-8",
    )
    frequency["output_dir"] = "/missing/cluster/path"
    summary_path.write_text(json.dumps(raw), encoding="utf-8")
    result = row_builder.build_paper_rows(summary_path, "Test model")
    assert result["tables"]["structural_v3"]["values"]["frequency"] == 25.0


def test_wrong_metric_name_is_a_clear_error(
    row_builder: ModuleType,
    tmp_path: Path,
) -> None:
    path = tmp_path / "suite_summary.json"
    path.write_text(
        json.dumps(
            {
                "eval_tasks": [
                    _task("beans_next_t1_caption", {"accuracy": 0.5}),
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="beans_next_t1_caption.*expected"):
        row_builder.build_paper_rows(path, "Test model")


def test_latex_cli_emits_six_concise_fragments(
    row_builder: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert row_builder.main(
        [str(_write_fixture(tmp_path)), "--model-label", "Test model", "--latex"]
    ) == 0
    output = capsys.readouterr().out
    assert output.count(r"\\") == 6
    assert "Test model & 0.012 & 0.046 & 0.079 & 0.500 & 0.250 & 0.750" in output
    assert "Test model & 1.2 & 34.5" in output
    assert "Test model & 50.0" in output
    assert "& 0.007 \\\\" in output
    assert "Test model & 0.123 & -- & -- & -- & --" in output
