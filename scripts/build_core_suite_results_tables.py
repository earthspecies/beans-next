#!/usr/bin/env python3
"""Build Markdown result tables for ``birdset_core`` and ``beans_zero_core`` ingests.

Scans ``--results-root`` for ``suite/<suite_id>/<task_id>/predictions.jsonl``,
selects the best 1:1-aligned predictions/processed pair per (model, task) using
the same rules as ``consolidate_beans_zero_results.py`` (with partial runs
allowed), reads ``summary.json`` ``metrics.mean``, and writes a Markdown table.

Cells show the primary metric from each task's eval registry (first listed metric,
with fallbacks). ``(incomplete)`` is appended only when rows are missing or the
chosen run is short of the canonical test split. A bare ``—`` means full row
coverage but no usable score in ``summary.json`` / ``metrics.mean``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _REPO_ROOT / "beans_next" / "registry"
_REGISTRY_EVAL = _REGISTRY / "eval_task"
_REGISTRY_SUITE = _REGISTRY / "suite"


def _load_suite_task_ids(suite_key: str) -> list[str]:
    path = _REGISTRY_SUITE / f"{suite_key}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data[suite_key]
    if not isinstance(block, dict) or "eval_tasks" not in block:
        msg = f"Malformed suite registry: {path}"
        raise ValueError(msg)
    tasks = block["eval_tasks"]
    if not isinstance(tasks, list):
        msg = f"eval_tasks must be a list in {path}"
        raise ValueError(msg)
    return [str(t) for t in tasks]


def _load_row_counts(filename: str) -> dict[str, int]:
    path = _REGISTRY / filename
    if not path.is_file():
        msg = f"Missing row-count registry: {path}"
        raise FileNotFoundError(msg)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{filename} must be a JSON object"
        raise ValueError(msg)
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, int) or v < 0:
            msg = f"Invalid entry in {filename}: {k!r}"
            raise ValueError(msg)
        out[k] = v
    return out


def _model_key(pred_path: Path, results_root: Path) -> str:
    rel = pred_path.resolve().relative_to(results_root.resolve())
    parts = rel.parts
    if len(parts) >= 2 and parts[1] == "suite":
        return parts[0]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return "unknown"


def _is_suite_path(pred_path: Path, suite_dir: str) -> bool:
    parts = pred_path.parts
    for i in range(len(parts) - 1):
        if parts[i] == suite_dir and i > 0 and parts[i - 1] == "suite":
            return True
    return False


def _count_jsonl_rows(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def _read_summary_mean(pred_dir: Path) -> dict[str, Any] | None:
    summary_path = pred_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    mean = metrics.get("mean")
    return dict(mean) if isinstance(mean, dict) else None


def _jsonl_sample_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSONL at {path}:{line_no}: {exc}"
            raise ValueError(msg) from exc
        sid = obj.get("sample_id")
        if not isinstance(sid, str) or not sid.strip():
            msg = f"Missing sample_id at {path}:{line_no}"
            raise ValueError(msg)
        ids.append(sid)
    return ids


def _pred_proc_1to1_report(pred_path: Path, proc_path: Path) -> dict[str, Any]:
    try:
        pred_ids = _jsonl_sample_ids(pred_path)
        proc_ids = _jsonl_sample_ids(proc_path)
    except ValueError as exc:
        return {"aligned": False, "n_rows": 0, "reason": str(exc)}
    sp, sq = set(pred_ids), set(proc_ids)
    if len(pred_ids) != len(sp):
        return {
            "aligned": False,
            "n_rows": len(pred_ids),
            "reason": "duplicate_sample_id_in_predictions",
        }
    if len(proc_ids) != len(sq):
        return {
            "aligned": False,
            "n_rows": len(pred_ids),
            "reason": "duplicate_sample_id_in_processed_predictions",
        }
    if len(pred_ids) != len(proc_ids):
        return {
            "aligned": False,
            "n_rows": len(pred_ids),
            "reason": "row_count_mismatch",
            "n_pred": len(pred_ids),
            "n_proc": len(proc_ids),
        }
    if sp != sq:
        return {
            "aligned": False,
            "n_rows": len(pred_ids),
            "reason": "sample_id_set_mismatch",
        }
    return {"aligned": True, "n_rows": len(pred_ids), "reason": None}


def _select_winner(
    paths: list[Path],
    *,
    task_id: str,
    expected_n: int,
    allow_partial: bool,
) -> tuple[Path | None, dict[str, Any]]:
    full: list[Path] = []
    partial: list[tuple[int, Path]] = []
    candidate_reports: list[dict[str, Any]] = []
    for pred_path in paths:
        proc_path = pred_path.parent / "processed_predictions.jsonl"
        rep = _pred_proc_1to1_report(pred_path, proc_path)
        n_rows = int(rep.get("n_rows", 0))
        entry: dict[str, Any] = {
            "predictions_path": str(pred_path),
            **rep,
            "matches_expected_rows": rep.get("aligned") and n_rows == expected_n,
        }
        candidate_reports.append(entry)
        if not rep["aligned"]:
            continue
        if n_rows == expected_n:
            full.append(pred_path)
        else:
            partial.append((n_rows, pred_path))

    meta: dict[str, Any] = {
        "task_id": task_id,
        "expected_rows": expected_n,
        "allow_partial": allow_partial,
        "candidates": candidate_reports,
    }

    def _tiebreak(candidates: list[Path]) -> Path:
        return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))

    if full:
        chosen = _tiebreak(full)
        meta["selection"] = "full_coverage"
        meta["chosen_predictions_path"] = str(chosen)
        meta["actual_rows"] = expected_n
        meta["full_coverage"] = True
        return chosen, meta

    if allow_partial and partial:
        max_n = max(t[0] for t in partial)
        tied = [p for n, p in partial if n == max_n]
        chosen = _tiebreak(tied)
        meta["selection"] = "partial_max_rows"
        meta["chosen_predictions_path"] = str(chosen)
        meta["actual_rows"] = max_n
        meta["full_coverage"] = False
        meta["warning"] = (
            f"No run with {expected_n} rows; using partial run with {max_n} rows."
        )
        return chosen, meta

    meta["selection"] = "none"
    meta["full_coverage"] = False
    if not partial and not full:
        meta["reason"] = (
            "No candidate with 1:1 predictions/processed alignment "
            "(row counts, duplicate-free ids, identical id sets)."
        )
    elif not allow_partial and partial and not full:
        meta["reason"] = (
            f"No full run ({expected_n} rows); only partial aligned runs exist."
        )
    else:
        meta["reason"] = "No qualifying candidate."
    return None, meta


def _iter_suite_predictions(
    results_root: Path,
    *,
    suite_dir: str,
    parent_prefix: str,
) -> Iterator[Path]:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        msg = f"Not a directory: {results_root}"
        raise FileNotFoundError(msg)
    for path in results_root.rglob("predictions.jsonl"):
        if not path.is_file():
            continue
        if not path.parent.name.startswith(parent_prefix):
            continue
        if not _is_suite_path(path, suite_dir):
            continue
        yield path


def _preferred_metric_names(task_id: str) -> list[str]:
    path = _REGISTRY_EVAL / f"{task_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data[task_id]
    metrics = block.get("metrics", []) if isinstance(block, dict) else []
    names: list[str] = []
    if isinstance(metrics, list):
        for m in metrics:
            if isinstance(m, dict):
                n = m.get("name")
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
    fallbacks = (
        "top1_accuracy",
        "accuracy",
        "f1",
        "average_precision",
        "cider",
        "spider",
        "macro_f1",
        "precision",
        "recall",
    )
    for fb in fallbacks:
        if fb not in names:
            names.append(fb)
    return names


def _pick_primary_metric(mean: Mapping[str, Any], preferred: list[str]) -> float | None:
    for k in preferred:
        v = mean.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    for _k, v in mean.items():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _short_column_header(suite_key: str, task_id: str) -> str:
    if suite_key == "beans_zero_core":
        return task_id.removeprefix("beans_zero_")
    if suite_key == "birdset_core":
        return task_id.removeprefix("birdset_").replace("_test_5s", "")
    return task_id


def _md_escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _build_table_for_suite(
    *,
    suite_key: str,
    row_counts_file: str,
    suite_dir: str,
    parent_prefix: str,
    results_root: Path,
    models_filter: set[str] | None,
    exclude_models: set[str],
) -> tuple[str, list[dict[str, Any]]]:
    tasks = _load_suite_task_ids(suite_key)
    expected = _load_row_counts(row_counts_file)
    for tid in tasks:
        if tid not in expected:
            msg = f"Task {tid!r} missing from {row_counts_file}"
            raise KeyError(msg)

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    paths = _iter_suite_predictions(
        results_root,
        suite_dir=suite_dir,
        parent_prefix=parent_prefix,
    )
    for pred_path in sorted(paths):
        tid = pred_path.parent.name
        if tid not in expected:
            continue
        mk = _model_key(pred_path, results_root)
        if models_filter is not None and mk not in models_filter:
            continue
        if mk in exclude_models:
            continue
        proc_path = pred_path.parent / "processed_predictions.jsonl"
        if proc_path.is_file() and _count_jsonl_rows(pred_path) > 0:
            grouped[(mk, tid)].append(pred_path)

    winners: dict[tuple[str, str], Path] = {}
    selection: dict[tuple[str, str], dict[str, Any]] = {}
    for key, paths in grouped.items():
        tid = key[1]
        chosen, meta = _select_winner(
            paths,
            task_id=tid,
            expected_n=expected[tid],
            allow_partial=True,
        )
        selection[key] = meta
        if chosen is not None:
            winners[key] = chosen

    models = sorted({mk for (mk, _) in grouped} | {mk for (mk, _) in winners})

    json_rows: list[dict[str, Any]] = []
    short_headers = [_short_column_header(suite_key, t) for t in tasks]
    header = ["Model"] + short_headers + ["coverage"]
    md_lines = [
        f"# {suite_key} — results (primary metric per task)",
        "",
        "Canonical row counts come from esp_data metadata registries. "
        "*(incomplete)* = fewer rows than the full test split or no aligned artifact. "
        "A bare — = full row count but no primary metric in `summary.json`.",
        "",
        "| " + " | ".join(_md_escape_cell(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]

    for mk in models:
        pref_by_task = {t: _preferred_metric_names(t) for t in tasks}
        cells: list[str] = [_md_escape_cell(mk)]
        complete_ct = 0
        row_json: dict[str, Any] = {
            "model_key": mk,
            "tasks": {},
            "coverage_complete": None,
        }
        for tid in tasks:
            key = (mk, tid)
            exp_n = expected[tid]
            if key not in winners:
                cells.append("— (incomplete)")
                row_json["tasks"][tid] = {
                    "primary_metric": None,
                    "complete": False,
                    "reason": selection.get(key, {}).get("reason")
                    if key in selection
                    else "no_artifacts",
                }
                continue

            pred_path = winners[key]
            meta = selection[key]
            actual = int(meta.get("actual_rows", _count_jsonl_rows(pred_path)))
            is_complete = bool(meta.get("full_coverage")) and actual == exp_n
            if is_complete:
                complete_ct += 1

            mean = _read_summary_mean(pred_path.parent)
            pref = pref_by_task[tid]
            val = _pick_primary_metric(mean or {}, pref) if mean else None
            metric_name_used: str | None = None
            if mean:
                for name in pref:
                    v = mean.get(name)
                    if isinstance(v, (int, float)):
                        metric_name_used = name
                        break
                if metric_name_used is None:
                    for name, v in mean.items():
                        if isinstance(v, (int, float)):
                            metric_name_used = str(name)
                            break

            if not is_complete:
                if val is None:
                    cell = "— (incomplete)"
                else:
                    cell = f"{val:.4f} (incomplete)"
            elif val is None:
                cell = "—"
            else:
                cell = f"{val:.4f}"

            cells.append(_md_escape_cell(cell))
            row_json["tasks"][tid] = {
                "primary_metric_name": metric_name_used,
                "primary_metric_value": val,
                "complete": is_complete,
                "expected_rows": exp_n,
                "actual_rows": actual,
                "predictions_path": str(pred_path),
            }

        cov = f"{complete_ct}/{len(tasks)}"
        cells.append(_md_escape_cell(cov))
        row_json["coverage_complete"] = cov
        json_rows.append(row_json)
        md_lines.append("| " + " | ".join(cells) + " |")

    md_lines.append("")
    return "\n".join(md_lines), json_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build birdset_core and beans_zero_core Markdown tables from ingests."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "results" / "ingested",
        help="Root directory to scan (e.g. results/ingested).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "tables",
        help="Directory for .md and .json outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="If set, only include these model_key values.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help="model_key values to exclude.",
    )
    args = parser.parse_args()
    results_root: Path = args.results_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_filter = set(args.models) if args.models else None
    exclude_models = set(args.exclude_models) if args.exclude_models else set()

    suites: list[tuple[str, str, str, str]] = [
        (
            "birdset_core",
            "birdset_core_test_row_counts.json",
            "birdset_core",
            "birdset_",
        ),
        (
            "beans_zero_core",
            "beans_zero_core_test_row_counts.json",
            "beans_zero_core",
            "beans_zero_",
        ),
    ]

    index: dict[str, Any] = {"results_root": str(results_root.resolve()), "suites": {}}

    for suite_key, counts_file, suite_dir, prefix in suites:
        md, json_rows = _build_table_for_suite(
            suite_key=suite_key,
            row_counts_file=counts_file,
            suite_dir=suite_dir,
            parent_prefix=prefix,
            results_root=results_root,
            models_filter=model_filter,
            exclude_models=exclude_models,
        )
        md_path = out_dir / f"{suite_key}_results.md"
        md_path.write_text(md + "\n", encoding="utf-8")
        json_path = out_dir / f"{suite_key}_results.json"
        json_path.write_text(
            json.dumps(json_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        index["suites"][suite_key] = {
            "markdown": str(md_path.resolve()),
            "json": str(json_path.resolve()),
        }

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
