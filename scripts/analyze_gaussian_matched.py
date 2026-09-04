#!/usr/bin/env python3
"""Analyze matched real-audio and Gaussian-noise benchmark runs.

The input to :func:`analyze_matched_runs` is a mapping from model name to a
``(real_root, noise_root)`` pair.  Each root is searched recursively for one
``summary.json`` per task.  The paired ``scored_predictions.jsonl`` files are
checked before summary metrics are compared, so a report cannot silently mix
different samples or targets.

This is an analysis utility, not a scorer.  Native summary metrics are copied
from each run.  For the frequency-range description task, the report also
computes a full-denominator IoU: missing row-level ``freq_mean_iou`` values
contribute zero.  Numeric parse rates use only rows that expose
``numeric_parse_success``; the denominator is reported explicitly.  Refusal
matches are heuristic candidates for review, not adjudicated refusals.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_FREQUENCY_TASK_IDS",
    "DEFAULT_REFUSAL_PATTERN",
    "analyze_matched_runs",
    "analyze_pair",
    "find_task_summaries",
]

DEFAULT_FREQUENCY_TASK_IDS = frozenset(
    {"beans_next_t3_frequency_range_description"}
)

# Keep this deliberately narrow.  Phrases such as "not mentioned" are often
# valid answers to the task and must not be labelled as refusals automatically.
DEFAULT_REFUSAL_PATTERN = re.compile(
    r"\b(?:"
    r"i\s+(?:cannot|can't|am\s+unable\s+to)\s+(?:determine|identify|answer|analy[sz]e)"
    r"|unable\s+to\s+(?:determine|identify|answer|analy[sz]e)"
    r"|cannot\s+(?:determine|identify|answer|analy[sz]e)"
    r"|can't\s+(?:determine|identify|answer|analy[sz]e)"
    r")\b",
    re.IGNORECASE,
)

_MISSING = object()


class MatchedAnalysisError(ValueError):
    """Raised when a purported real/noise pair is not aligned."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedAnalysisError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, dict):
        raise MatchedAnalysisError(f"JSON artifact is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MatchedAnalysisError(f"cannot read JSONL artifact {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MatchedAnalysisError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise MatchedAnalysisError(
                f"JSONL row is not an object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def find_task_summaries(root: str | Path) -> dict[str, Path]:
    """Find task summaries below a run root.

    Parameters
    ----------
    root
        Run directory containing task subdirectories, possibly below suite
        and modality directories.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from task ID to its unique summary path.

    Raises
    ------
    MatchedAnalysisError
        If the root is missing, has no summaries, or contains duplicate task
        IDs.
    """

    base = Path(root).expanduser()
    if not base.is_dir():
        raise MatchedAnalysisError(f"run root is not a directory: {base}")
    summaries: dict[str, Path] = {}
    paths = sorted(base.rglob("summary.json"))
    if not paths:
        raise MatchedAnalysisError(f"no summary.json found below {base}")
    for path in paths:
        summary = _load_json(path)
        task_id = _task_id(summary, path)
        if task_id in summaries:
            raise MatchedAnalysisError(
                f"duplicate task {task_id!r}: {summaries[task_id]} and {path}"
            )
        summaries[task_id] = path
    return summaries


def _task_id(summary: Mapping[str, Any], summary_path: Path) -> str:
    summary_metrics = summary.get("metrics")
    per_task = (
        summary_metrics.get("per_task_mean")
        if isinstance(summary_metrics, Mapping)
        else None
    )
    if isinstance(per_task, Mapping) and len(per_task) == 1:
        only = next(iter(per_task))
        if isinstance(only, str) and only:
            return only
    task_id = summary.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    return summary_path.parent.name


def _is_error(row: Mapping[str, Any]) -> bool:
    value = row.get("error", _MISSING)
    return value not in (_MISSING, None, "", [], {})


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _scores(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("scores")
    return value if isinstance(value, Mapping) else {}


def _prediction_text(row: Mapping[str, Any]) -> str:
    values = row.get("predictions", row.get("prediction", row.get("text", [])))
    if isinstance(values, str):
        return values
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        return " ".join(str(value) for value in values)
    return str(values) if values not in (None, _MISSING) else ""


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("sample_id", row.get("id", row.get("example_id", _MISSING)))
    return str(index) if value in (_MISSING, None) else str(value)


def _metric_means(summary: Mapping[str, Any], task_id: str) -> Mapping[str, Any]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise MatchedAnalysisError(f"{task_id}: summary has no metrics.mean")
    per_task = metrics.get("per_task_mean")
    if isinstance(per_task, Mapping) and isinstance(per_task.get(task_id), Mapping):
        return per_task[task_id]
    mean = metrics.get("mean")
    if isinstance(mean, Mapping):
        return mean
    raise MatchedAnalysisError(f"{task_id}: summary has no metric means")


def _model_revision(summary: Mapping[str, Any]) -> str | None:
    identity = summary.get("model_identity")
    if not isinstance(identity, Mapping):
        return None
    value = identity.get("model_revision")
    return str(value) if value is not None else None


def _check_rows(
    summary: Mapping[str, Any], rows: list[dict[str, Any]], path: Path, task_id: str
) -> None:
    declared = _finite_number(summary.get("n_samples"))
    if declared is not None and int(declared) != len(rows):
        raise MatchedAnalysisError(
            f"{task_id}: n_samples={int(declared)} but {path} has {len(rows)} rows"
        )
    errors = sum(_is_error(row) for row in rows)
    declared_errors = _finite_number(summary.get("n_errors"))
    if declared_errors is not None and int(declared_errors) != errors:
        raise MatchedAnalysisError(
            f"{task_id}: n_errors={int(declared_errors)} but {path} has {errors} errors"
        )


def _diagnostics(
    rows: list[dict[str, Any]],
    refusal_pattern: re.Pattern[str],
) -> dict[str, Any]:
    usable = [row for row in rows if not _is_error(row)]
    parse_values = [
        _finite_number(_scores(row)["numeric_parse_success"])
        for row in usable
        if "numeric_parse_success" in _scores(row)
    ]
    parse_values = [value for value in parse_values if value is not None]
    scorer_values = [
        _finite_number(_scores(row)["parse_success"])
        for row in usable
        if "parse_success" in _scores(row)
    ]
    scorer_values = [value for value in scorer_values if value is not None]
    metric_denominators: dict[str, int] = {}
    for row in usable:
        for name, value in _scores(row).items():
            if _finite_number(value) is not None:
                key = str(name)
                metric_denominators[key] = metric_denominators.get(key, 0) + 1
    candidates = [
        _sample_id(row, index)
        for index, row in enumerate(usable)
        if refusal_pattern.search(_prediction_text(row))
    ]
    return {
        "non_error_denominator": len(usable),
        "empty_processed_prediction_count": sum(
            not str(row.get("processed_prediction", "") or "").strip()
            for row in usable
        ),
        "metric_denominators": dict(sorted(metric_denominators.items())),
        "numeric_parse_denominator": len(parse_values),
        "numeric_parse_failure_count": sum(value != 1.0 for value in parse_values),
        "numeric_parse_failure_rate": (
            sum(value != 1.0 for value in parse_values) / len(parse_values)
            if parse_values
            else None
        ),
        # This scorer flag is intentionally exposed separately.  For species
        # dictionaries it describes target/scorer parsing, not model parsing.
        "scorer_parse_flag_denominator": len(scorer_values),
        "scorer_parse_flag_zero_count": sum(value == 0.0 for value in scorer_values),
        "scorer_parse_flag_zero_rate": (
            sum(value == 0.0 for value in scorer_values) / len(scorer_values)
            if scorer_values
            else None
        ),
        "refusal_heuristic_count": len(candidates),
        "refusal_heuristic_rate": len(candidates) / len(usable) if usable else None,
        "refusal_candidate_sample_ids": candidates,
    }


def _coverage_iou(rows: Iterable[Mapping[str, Any]]) -> tuple[float, int]:
    rows_list = list(rows)
    values = [
        _finite_number(_scores(row).get("freq_mean_iou")) or 0.0
        for row in rows_list
    ]
    return (sum(values) / len(values) if values else 0.0, len(values))


def _paired_error_diagnostics(
    task_id: str,
    real_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    real = {_sample_id(row, index): row for index, row in enumerate(real_rows)}
    noise = {_sample_id(row, index): row for index, row in enumerate(noise_rows)}
    if set(real) != set(noise):
        raise MatchedAnalysisError(f"{task_id}: sample IDs differ between paired rows")
    errors: list[dict[str, str]] = []
    for sample_id in real:
        real_error = real[sample_id].get("error")
        noise_error = noise[sample_id].get("error")
        if _is_error(real[sample_id]) or _is_error(noise[sample_id]):
            errors.append(
                {
                    "task_id": task_id,
                    "sample_id": sample_id,
                    "real_error": (
                        "" if real_error in (None, _MISSING) else str(real_error)
                    ),
                    "noise_error": (
                        "" if noise_error in (None, _MISSING) else str(noise_error)
                    ),
                }
            )
    return errors


def analyze_pair(
    real_root: str | Path,
    noise_root: str | Path,
    *,
    frequency_task_ids: Iterable[str] = DEFAULT_FREQUENCY_TASK_IDS,
    refusal_pattern: re.Pattern[str] = DEFAULT_REFUSAL_PATTERN,
) -> dict[str, Any]:
    """Analyze one matched real/noise pair and return a JSON-ready report.

    Returns
    -------
    dict
        JSON-ready report containing per-task metric deltas and diagnostics.

    Raises
    ------
    MatchedAnalysisError
        If task sets, sample IDs, targets, declared counts, or summary metrics
        are not aligned.
    """

    real_summaries = find_task_summaries(real_root)
    noise_summaries = find_task_summaries(noise_root)
    if set(real_summaries) != set(noise_summaries):
        raise MatchedAnalysisError(
            "task sets differ: "
            f"real-only={sorted(set(real_summaries) - set(noise_summaries))}, "
            f"noise-only={sorted(set(noise_summaries) - set(real_summaries))}"
        )
    frequency_ids = set(frequency_task_ids)
    tasks: dict[str, Any] = {}
    paired_errors: list[dict[str, str]] = []
    total_samples = 0
    real_revisions: set[str] = set()
    noise_revisions: set[str] = set()
    for task_id in sorted(real_summaries):
        real_summary_path = real_summaries[task_id]
        noise_summary_path = noise_summaries[task_id]
        real_summary = _load_json(real_summary_path)
        noise_summary = _load_json(noise_summary_path)
        real_revision = _model_revision(real_summary)
        noise_revision = _model_revision(noise_summary)
        if real_revision is not None:
            real_revisions.add(real_revision)
        if noise_revision is not None:
            noise_revisions.add(noise_revision)
        if (
            real_revision is not None
            and noise_revision is not None
            and real_revision != noise_revision
        ):
            raise MatchedAnalysisError(f"{task_id}: model revisions differ")
        real_scored = real_summary_path.with_name("scored_predictions.jsonl")
        noise_scored = noise_summary_path.with_name("scored_predictions.jsonl")
        real_rows = _load_jsonl(real_scored)
        noise_rows = _load_jsonl(noise_scored)
        _check_rows(real_summary, real_rows, real_scored, task_id)
        _check_rows(noise_summary, noise_rows, noise_scored, task_id)
        if len(real_rows) != len(noise_rows):
            raise MatchedAnalysisError(f"{task_id}: paired row counts differ")
        for index, (real_row, noise_row) in enumerate(
            zip(real_rows, noise_rows, strict=True)
        ):
            real_key = (_sample_id(real_row, index), real_row.get("targets"))
            noise_key = (_sample_id(noise_row, index), noise_row.get("targets"))
            if real_key != noise_key:
                raise MatchedAnalysisError(
                    f"{task_id}: sample/target mismatch at row {index}: "
                    f"real={real_key!r}, noise={noise_key!r}"
                )
        real_metrics = _metric_means(real_summary, task_id)
        noise_metrics = _metric_means(noise_summary, task_id)
        if set(real_metrics) != set(noise_metrics):
            raise MatchedAnalysisError(f"{task_id}: summary metric sets differ")
        metrics: dict[str, Any] = {}
        for name in sorted(real_metrics):
            real_value = _finite_number(real_metrics[name])
            noise_value = _finite_number(noise_metrics[name])
            if real_value is None or noise_value is None:
                raise MatchedAnalysisError(f"{task_id}: non-finite metric {name!r}")
            metrics[name] = {
                "real": real_value,
                "noise": noise_value,
                "audio_minus_noise": real_value - noise_value,
            }
        coverage_denominator = None
        if task_id in frequency_ids or task_id.endswith("frequency_range_description"):
            real_coverage, coverage_denominator = _coverage_iou(real_rows)
            noise_coverage, noise_denominator = _coverage_iou(noise_rows)
            if coverage_denominator != noise_denominator:
                raise MatchedAnalysisError(f"{task_id}: coverage denominators differ")
            metrics["coverage_aware_freq_mean_iou"] = {
                "real": real_coverage,
                "noise": noise_coverage,
                "audio_minus_noise": real_coverage - noise_coverage,
            }
        errors = _paired_error_diagnostics(task_id, real_rows, noise_rows)
        paired_errors.extend(errors)
        n_samples = len(real_rows)
        total_samples += n_samples
        tasks[task_id] = {
            "n_samples": n_samples,
            "real_errors": sum(_is_error(row) for row in real_rows),
            "noise_errors": sum(_is_error(row) for row in noise_rows),
            "metrics": metrics,
            "real_diagnostics": _diagnostics(real_rows, refusal_pattern),
            "noise_diagnostics": _diagnostics(noise_rows, refusal_pattern),
            **(
                {"coverage_denominator": coverage_denominator}
                if coverage_denominator is not None
                else {}
            ),
        }
    real_errors = sum(task["real_errors"] for task in tasks.values())
    noise_errors = sum(task["noise_errors"] for task in tasks.values())
    return {
        "n_tasks": len(tasks),
        "n_samples": total_samples,
        "real_errors": real_errors,
        "noise_errors": noise_errors,
        "successful_real_samples": total_samples - real_errors,
        "successful_noise_samples": total_samples - noise_errors,
        "real_model_revisions": sorted(real_revisions),
        "noise_model_revisions": sorted(noise_revisions),
        "paired_error_count": len(paired_errors),
        "paired_error_samples": paired_errors,
        "tasks": tasks,
    }


def analyze_matched_runs(
    pairs: Mapping[str, tuple[str | Path, str | Path]],
    *,
    frequency_task_ids: Iterable[str] = DEFAULT_FREQUENCY_TASK_IDS,
    refusal_pattern: re.Pattern[str] = DEFAULT_REFUSAL_PATTERN,
) -> dict[str, Any]:
    """Analyze all named matched pairs in deterministic key order.

    Returns
    -------
    dict
        Mapping from model name to its matched-pair report.
    """

    return {
        model: analyze_pair(
            real_root,
            noise_root,
            frequency_task_ids=frequency_task_ids,
            refusal_pattern=refusal_pattern,
        )
        for model, (real_root, noise_root) in sorted(pairs.items())
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        metavar=("MODEL", "REAL_ROOT", "NOISE_ROOT"),
        action="append",
        required=True,
        help="named pair; repeat for each model",
    )
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matched-analysis CLI and emit a stable JSON report.

    Returns
    -------
    int
        Zero after writing the report.

    Raises
    ------
    SystemExit
        If duplicate model names are supplied.
    """

    args = _build_parser().parse_args(argv)
    pairs = {model: (real, noise) for model, real, noise in args.pair}
    if len(pairs) != len(args.pair):
        raise SystemExit("duplicate model names in --pair arguments")
    report = analyze_matched_runs(pairs)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
