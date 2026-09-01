#!/usr/bin/env python3
"""Build paper-table rows from one BEANS-Next suite summary.

The paper tables intentionally contain a small, fixed subset of the tasks in a
full BEANS-Next suite.  This module keeps that mapping in one place and emits
values in the units used by the tables.  A missing task is represented by
``None`` in JSON and ``--`` in the LaTeX fragments.

The frequency-range row is a special case.  The suite summary contains a
matched-species mean for that task, while the paper reports a coverage-aware
mean.  Consequently, its value is recomputed from every row in the sibling
``scored_predictions.jsonl`` file, treating a missing ``freq_mean_iou`` as
zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ColumnSpec:
    """Description of one paper-table column."""

    key: str
    task_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    scale: float
    number_format: str
    source: str = "summary"

    @property
    def expected_task_id(self) -> str:
        return self.task_ids[0]

    @property
    def expected_metric_name(self) -> str:
        return self.metric_names[0]


def _c(
    key: str,
    task_id: str,
    metric: str,
    *,
    scale: float = 1.0,
    number_format: str = ".1f",
    aliases: Sequence[str] = (),
    metric_aliases: Sequence[str] = (),
    source: str = "summary",
) -> ColumnSpec:
    return ColumnSpec(
        key=key,
        task_ids=(task_id, *aliases),
        metric_names=(metric, *metric_aliases),
        scale=scale,
        number_format=number_format,
        source=source,
    )


# The task IDs below are the IDs used by the current beans-next registry.  The
# two tasks not yet present in that registry are included deliberately: they
# correspond to the Insect and Begging columns in semantic_v2 and remain null
# until a future suite exports them.
TABLE_SPECS: dict[str, tuple[ColumnSpec, ...]] = {
    "acoustic_v2": (
        _c("caption", "beans_next_t1_caption", "cider", scale=100.0),
        _c(
            "description_mcq",
            "beans_next_t1_description_mcq",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "crow",
            "beans_next_crow_description",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "snr_mcq",
            "beans_next_t1_snr_mcq",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "snr_regression",
            "beans_next_t1_snr_regression",
            "absolute_error",
            number_format=".2f",
            metric_aliases=("mean_absolute_error",),
        ),
        _c(
            "f0_seen",
            "beans_next_f0_mean_seen_taxa",
            "absolute_error",
            number_format=".2f",
            metric_aliases=("mean_absolute_error",),
        ),
        _c(
            "f0_heldout",
            "beans_next_f0_mean_heldout_taxa",
            "absolute_error",
            number_format=".2f",
            metric_aliases=("mean_absolute_error",),
        ),
    ),
    "semantic_v2": (
        _c(
            "bird",
            "beans_next_bird_presence",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "mammal",
            "beans_next_mammal_presence",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "amphibian",
            "beans_next_amphibian_presence",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "insect",
            "beans_next_insect_presence",
            "top1_accuracy",
            scale=100.0,
            aliases=("beans_next_insect_presence_binary",),
        ),
        _c(
            "alarm",
            "beans_next_alarm_call_presence",
            "top1_accuracy",
            scale=100.0,
            # Older summaries used the registry's ``accuracy`` name for this
            # binary detection task; it is numerically the same quantity.
            metric_aliases=("accuracy",),
        ),
        _c(
            "flight",
            "beans_next_flight_call_presence",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "begging",
            "beans_next_begging_call_presence",
            "top1_accuracy",
            scale=100.0,
            aliases=("beans_next_begging_presence",),
        ),
        _c(
            "fixed",
            "beans_next_call_type_fixed_vocab",
            "macro_f1",
            scale=100.0,
        ),
        _c(
            "behavior",
            "beans_next_t2_behavior",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "caption",
            "beans_next_t2_captioning",
            "cider",
            number_format=".3f",
        ),
    ),
    "structural_v3": (
        _c(
            "species_count",
            "beans_next_t3_species_count_oe",
            "absolute_error",
            number_format=".2f",
            metric_aliases=("mean_absolute_error",),
        ),
        _c(
            "vocalization_count",
            "beans_next_t3_vocalization_count_total_oe",
            "absolute_error",
            number_format=".2f",
            metric_aliases=("mean_absolute_error",),
        ),
        _c(
            "vocalization_count_per_species",
            "beans_next_t3_vocalization_count_per_species_oe",
            "count_mae",
            number_format=".2f",
            metric_aliases=("mean_absolute_error",),
        ),
        _c(
            "referring",
            "beans_next_t3_vocalization_referring_mcq",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "cooccurrence",
            "beans_next_t3_vocalization_cooccurrence_binary",
            "top1_accuracy",
            scale=100.0,
        ),
        _c(
            "frequency",
            "beans_next_t3_frequency_range_description",
            "freq_mean_iou",
            scale=100.0,
            source="scored_predictions.jsonl",
        ),
        _c(
            "summary",
            "beans_next_t3_ordered_species_summary",
            "species_f1",
            scale=100.0,
        ),
        _c(
            "caption",
            "beans_next_t3_structural_captioning",
            "cider",
            scale=100.0,
        ),
    ),
    "tier4": (
        _c(
            "gibbons",
            "beans_next_gibbon_fewshot_detection_balanced",
            "macro_f1",
            number_format=".3f",
            aliases=("beans_next_multiaudio_gibbon_fewshot_detection_balanced",),
        ),
        _c(
            "otters",
            "beans_next_giant_otter_4way",
            "accuracy",
            number_format=".3f",
            metric_aliases=("top1_accuracy",),
            aliases=("beans_next_multiaudio_giant_otter_4way",),
        ),
        _c(
            "dcase",
            "beans_next_dcase_fewshot_detection_balanced",
            "macro_f1",
            number_format=".3f",
            aliases=(
                "beans_next_multiaudio_dcase_fewshot_detection_balanced",
            ),
        ),
        _c(
            "crows",
            "beans_next_crow_4way",
            "accuracy",
            number_format=".3f",
            metric_aliases=("top1_accuracy",),
            aliases=("beans_next_multiaudio_crow_4way",),
        ),
        _c(
            "unseen",
            "beans_next_unseen_species_4way",
            "accuracy",
            number_format=".3f",
            metric_aliases=("top1_accuracy",),
            aliases=("beans_next_multiaudio_unseen_species_4way",),
        ),
    ),
}

# The second structural table has the same ordering as structural_v3.tex.
STRUCTURAL_SPECIES_SPECS: tuple[ColumnSpec, ...] = (
    _c(
        "order_oe",
        "beans_next_t3_species_by_vocalization_order_oe",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "highest_pitch_oe",
        "beans_next_t3_species_by_highest_pitch_oe",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "lowest_pitch_oe",
        "beans_next_t3_species_by_lowest_pitch_oe",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "longest_vocalization_oe",
        "beans_next_t3_species_by_longest_vocalization_oe",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "frequency_oe",
        "beans_next_t3_species_by_vocalization_frequency_oe",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "order_mcq",
        "beans_next_t3_species_by_vocalization_order_mcq",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "highest_pitch_mcq",
        "beans_next_t3_species_by_highest_pitch_mcq",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "lowest_pitch_mcq",
        "beans_next_t3_species_by_lowest_pitch_mcq",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "longest_vocalization_mcq",
        "beans_next_t3_species_by_longest_vocalization_mcq",
        "top1_accuracy",
        scale=100.0,
    ),
    _c(
        "frequency_mcq",
        "beans_next_t3_species_by_vocalization_frequency_mcq",
        "top1_accuracy",
        scale=100.0,
    ),
)


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Missing suite summary: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in suite summary {path}: {exc}") from exc


def _load_suite_tasks(
    summary_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    raw = _load_json(summary_path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Suite summary {summary_path} must contain a JSON object")
    eval_tasks = raw.get("eval_tasks")
    if not isinstance(eval_tasks, list):
        raise ValueError(
            f"Suite summary {summary_path} must contain an eval_tasks list"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, task in enumerate(eval_tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"eval_tasks[{index}] must be a JSON object")
        task_id = task.get("eval_task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"eval_tasks[{index}] is missing a non-empty eval_task_id")
        if task_id in by_id:
            raise ValueError(f"Duplicate eval_task_id in suite summary: {task_id!r}")
        by_id[task_id] = task
    return raw, by_id


def _task_mean(task_id: str, task: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = task.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"Task {task_id!r} is missing its summary object")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Task {task_id!r} summary is missing metrics")
    mean = metrics.get("mean")
    if not isinstance(mean, Mapping):
        raise ValueError(f"Task {task_id!r} summary is missing metrics.mean")
    return mean


def _task_output_dir(summary_path: Path, task_id: str, task: Mapping[str, Any]) -> Path:
    # A copied result tree retains the cluster's absolute ``output_dir`` in its
    # summary.  Prefer the colocated task directory so exports stay portable.
    sibling = summary_path.parent / task_id
    if sibling.is_dir():
        return sibling
    output_dir = task.get("output_dir")
    if output_dir is None and isinstance(task.get("summary"), Mapping):
        output_dir = task["summary"].get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        path = Path(output_dir)
        return path if path.is_absolute() else summary_path.parent / path
    return summary_path.parent / task_id


def _number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return result


def _frequency_iou_mean(
    summary_path: Path,
    task_id: str,
    task: Mapping[str, Any],
) -> float:
    path = _task_output_dir(summary_path, task_id, task) / "scored_predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Frequency task {task_id!r} is present but its sibling "
            "scored_predictions.jsonl "
            f"is missing: {path}"
        )
    total = 0.0
    n_rows = 0
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"JSONL row at {path}:{line_no} must be an object")
        scores = row.get("scores")
        value = 0.0
        if isinstance(scores, Mapping) and "freq_mean_iou" in scores:
            raw_value = scores["freq_mean_iou"]
            # Match the paper definition exactly: a missing or null score is
            # the zero contribution of a missed species band.
            if raw_value is not None:
                value = _number(
                    raw_value,
                    context=f"scores.freq_mean_iou at {path}:{line_no}",
                )
        total += value
        n_rows += 1
    if n_rows == 0:
        raise ValueError(f"Frequency task scored_predictions.jsonl is empty: {path}")
    return total / n_rows


def _find_task(
    spec: ColumnSpec,
    tasks: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, Mapping[str, Any] | None]:
    matches = [
        (task_id, tasks[task_id])
        for task_id in spec.task_ids
        if task_id in tasks
    ]
    if len(matches) > 1:
        ids = ", ".join(task_id for task_id, _ in matches)
        raise ValueError(
            f"Multiple task IDs match paper column {spec.key!r}: {ids}. "
            "Keep one expected task ID in the suite summary."
        )
    return matches[0] if matches else (None, None)


def _column_value(
    spec: ColumnSpec,
    *,
    summary_path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
) -> tuple[float | None, dict[str, Any]]:
    task_id, task = _find_task(spec, tasks)
    if task is None or task_id is None:
        return None, {
            "task_id": None,
            "metric_name": spec.expected_metric_name,
            "raw_value": None,
            "value": None,
            "scale": spec.scale,
            "source": spec.source,
            "expected_task_id": spec.expected_task_id,
        }

    if spec.source == "scored_predictions.jsonl":
        raw_value = _frequency_iou_mean(summary_path, task_id, task)
        metric_name = spec.expected_metric_name
    else:
        mean = _task_mean(task_id, task)
        metric_name = next((name for name in spec.metric_names if name in mean), None)
        if metric_name is None:
            expected = ", ".join(repr(name) for name in spec.metric_names)
            available = ", ".join(sorted(str(name) for name in mean)) or "(none)"
            raise ValueError(
                f"Task {task_id!r} has no expected metric for paper column "
                f"{spec.key!r}; expected one of {expected}, found {available}"
            )
        raw_value = _number(
            mean[metric_name],
            context=f"metrics.mean.{metric_name} for task {task_id!r}",
        )
    value = raw_value * spec.scale
    return value, {
        "task_id": task_id,
        "metric_name": metric_name,
        "raw_value": raw_value,
        "value": value,
        "scale": spec.scale,
        "source": spec.source,
        "expected_task_id": spec.expected_task_id,
    }


def _latex_label(label: str) -> str:
    # Labels are commonly supplied as ``\\textit{...}`` or ``\\model``.  Keep
    # LaTeX commands usable while protecting the one separator with meaning in
    # a table row.
    return label.replace("&", r"\&").replace("\n", " ")


def _format_latex_value(value: float | None, spec: ColumnSpec) -> str:
    if value is None:
        return "--"
    return format(value, spec.number_format)


def _table_result(
    table_name: str,
    specs: Sequence[ColumnSpec],
    *,
    label: str,
    summary_path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values: dict[str, float | None] = {}
    task_results: dict[str, dict[str, Any]] = {}
    fragments: list[str] = [_latex_label(label)]
    for spec in specs:
        value, record = _column_value(
            spec,
            summary_path=summary_path,
            tasks=tasks,
        )
        values[spec.key] = value
        task_results[spec.key] = record
        fragments.append(_format_latex_value(value, spec))
    latex_row = " & ".join(fragments) + r" \\"
    return {
        "model_label": label,
        "columns": [spec.key for spec in specs],
        "values": values,
        "tasks": task_results,
        "row": [label, *values.values()],
        "latex_row": latex_row,
    }


def build_paper_rows(
    suite_summary: str | Path,
    model_label: str,
) -> dict[str, Any]:
    """Build JSON-ready rows for the active paper tables.

    Parameters
    ----------
    suite_summary
        Path to one full ``suite_summary.json``.
    model_label
        Label to place in each emitted row.

    Returns
    -------
    dict[str, Any]
        JSON-serializable table rows, metric provenance, and LaTeX fragments.

    Raises
    ------
    ValueError
        If the suite summary is malformed or a present task lacks its expected
        metric.
    """
    summary_path = Path(suite_summary).expanduser().resolve()
    raw, tasks = _load_suite_tasks(summary_path)
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be a non-empty string")

    tables: dict[str, Any] = {}
    for name, specs in TABLE_SPECS.items():
        tables[name] = _table_result(
            name,
            specs,
            label=model_label,
            summary_path=summary_path,
            tasks=tasks,
        )

    species = _table_result(
        "structural_v3_species_id",
        STRUCTURAL_SPECIES_SPECS,
        label=model_label,
        summary_path=summary_path,
        tasks=tasks,
    )
    main = tables["structural_v3"]
    tables["structural_v3"]["subtables"] = {
        "main": {
            key: value for key, value in main.items() if key != "subtables"
        },
        "species_id": species,
    }
    latex_rows = {
        "acoustic_v2": tables["acoustic_v2"]["latex_row"],
        "semantic_v2": tables["semantic_v2"]["latex_row"],
        "structural_v3_main": tables["structural_v3"]["latex_row"],
        "structural_v3_species_id": species["latex_row"],
        "tier4": tables["tier4"]["latex_row"],
    }
    # ``rows`` is a compact, table-oriented view for consumers that do not
    # need provenance.  ``tables.*.tasks`` retains the metric audit trail.
    rows = {
        "acoustic_v2": tables["acoustic_v2"]["row"],
        "semantic_v2": tables["semantic_v2"]["row"],
        "structural_v3_main": tables["structural_v3"]["row"],
        "structural_v3_species_id": species["row"],
        "tier4": tables["tier4"]["row"],
    }
    return {
        "schema_version": "beans_next.text_only_paper_rows.v1",
        "model_label": model_label,
        "suite_id": raw.get("suite_id"),
        "suite_summary": str(summary_path),
        "tables": tables,
        "rows": rows,
        "latex_rows": latex_rows,
    }


# Short aliases make the small utility convenient to import in notebooks and
# tests without making callers depend on the internal table representation.
build_rows = build_paper_rows


def _latex_output(result: Mapping[str, Any]) -> str:
    latex_rows = result.get("latex_rows")
    if not isinstance(latex_rows, Mapping):
        raise ValueError("Result is missing latex_rows")
    order = (
        "acoustic_v2",
        "semantic_v2",
        "structural_v3_main",
        "structural_v3_species_id",
        "tier4",
    )
    return "\n".join(str(latex_rows[name]) for name in order) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build active BEANS-Next text-only paper rows from suite_summary.json."
        )
    )
    parser.add_argument("suite_summary", type=Path)
    parser.add_argument("--model-label", required=True, help="LaTeX/JSON model label")
    parser.add_argument(
        "--format",
        choices=("json", "latex"),
        default="json",
        help="Output JSON (default) or concise LaTeX row fragments",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Alias for --format latex",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file (stdout is used when omitted)",
    )
    args = parser.parse_args(argv)
    try:
        result = build_paper_rows(args.suite_summary, args.model_label)
        output = (
            _latex_output(result)
            if args.latex or args.format == "latex"
            else json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        if args.output is None:
            sys.stdout.write(output)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
