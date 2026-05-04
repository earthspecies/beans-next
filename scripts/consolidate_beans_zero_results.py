#!/usr/bin/env python3
"""Inventory BEANS-Zero core results, merge predictions per model, rescore per task.

Walks ``results/ingested/**/suite/beans_zero_core/beans_zero_*/predictions.jsonl``,
records any existing ``summary.json`` means, then for each (model, task) **selects a
single run** that satisfies:

* **1:1 alignment**: same row count in ``predictions.jsonl`` and
  ``processed_predictions.jsonl``, identical ``sample_id`` sets, and no duplicate
  ``sample_id`` within either file.
* **Full subset coverage** (default): row count equals the canonical test size from
  ``beans_next/registry/beans_zero_core_test_row_counts.json`` (from esp_data
  metadata).  This avoids treating a 3-sample smoke run as comparable to a full
  evaluation (where sparse correct predictions can inflate means toward 1.0).

Use ``--allow-partial`` only for development when no full run exists.

Then writes one merged JSONL pair per model and runs
:func:`beans_next.runner.rescorer.rescore_predictions_file` per task with
``task_type`` from the eval-task registry.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from beans_next.runner.rescorer import rescore_predictions_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_EVAL = _REPO_ROOT / "beans_next" / "registry" / "eval_task"
_REGISTRY_ROW_COUNTS = (
    _REPO_ROOT / "beans_next" / "registry" / "beans_zero_core_test_row_counts.json"
)


def _task_id_from_pred_path(pred_path: Path) -> str | None:
    """Return eval task id from a ``beans_zero_*`` predictions path.

    Returns
    -------
    str or None
        Parent directory name when it starts with ``beans_zero_``, else ``None``.
    """
    name = pred_path.parent.name
    if not name.startswith("beans_zero_"):
        return None
    return name


def _model_key(pred_path: Path, results_root: Path) -> str:
    """Derive a stable model grouping key from the path under ``results_root``.

    Flat ingests use ``<run_name>/suite/beans_zero_core/...`` (only one segment
    before ``suite``). Nested layouts use ``<campaign>/<model>/.../suite/...``.

    Returns
    -------
    str
        ``run_name`` when the second segment is ``suite``, else ``a/b`` for the
        first two path components.
    """
    rel = pred_path.resolve().relative_to(results_root.resolve())
    parts = rel.parts
    if len(parts) >= 2 and parts[1] == "suite":
        return parts[0]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return "unknown"


def _is_beans_zero_core_path(pred_path: Path) -> bool:
    parts = pred_path.parts
    for i in range(len(parts) - 1):
        if parts[i] == "beans_zero_core" and i > 0 and parts[i - 1] == "suite":
            return True
    return False


def _count_jsonl_rows(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return sum(1 for ln in text.splitlines() if ln.strip())


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


def _load_expected_row_counts() -> dict[str, int]:
    """Load canonical beans_zero_core test row counts per eval task id.

    Returns
    -------
    dict[str, int]
        Task id → number of test rows (esp_data metadata; no audio I/O).

    Raises
    ------
    FileNotFoundError
        When the registry JSON is missing.
    ValueError
        When the file is malformed.
    """
    if not _REGISTRY_ROW_COUNTS.is_file():
        msg = (
            f"Missing beans_zero_core test row counts registry: {_REGISTRY_ROW_COUNTS}"
        )
        raise FileNotFoundError(msg)
    raw = json.loads(_REGISTRY_ROW_COUNTS.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "beans_zero_core_test_row_counts.json must be a JSON object"
        raise ValueError(msg)
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, int) or v < 0:
            msg = f"Invalid entry in beans_zero_core_test_row_counts.json: {k!r}"
            raise ValueError(msg)
        out[k] = v
    return out


def _jsonl_sample_ids(path: Path) -> list[str]:
    """Parse ``sample_id`` from each JSONL row.

    Returns
    -------
    list[str]
        Ordered ids as stored in the file.

    Raises
    ------
    ValueError
        On invalid JSON, missing ``sample_id``, or blank non-empty parse issues.
    """
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
    """Check predictions vs processed targets file for pairwise consistency.

    Returns
    -------
    dict[str, Any]
        Keys include ``aligned`` (bool), ``n_rows``, ``reason`` (when not aligned).
    """
    try:
        pred_ids = _jsonl_sample_ids(pred_path)
        proc_ids = _jsonl_sample_ids(proc_path)
    except ValueError as exc:
        return {
            "aligned": False,
            "n_rows": 0,
            "reason": str(exc),
        }
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
    """Pick one predictions path per coverage and alignment rules.

    Returns
    -------
    pathlib.Path or None
        Selected ``predictions.jsonl``, or ``None`` if no candidate qualifies.
    dict[str, Any]
        Selection diagnostics (candidates, selection mode, etc.).
    """
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
        return max(
            candidates,
            key=lambda p: (p.stat().st_mtime, str(p)),
        )

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
            f"No full run ({expected_n} rows); only partial aligned runs exist. "
            "Pass --allow-partial to pick the largest partial run."
        )
    else:
        meta["reason"] = "No qualifying candidate."
    return None, meta


def _load_task_type(task_id: str) -> str:
    path = _REGISTRY_EVAL / f"{task_id}.yaml"
    if not path.is_file():
        msg = f"Missing eval_task registry file for {task_id!r}: {path}"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or task_id not in data:
        msg = f"Malformed registry entry (expected top-level {task_id!r}): {path}"
        raise ValueError(msg)
    block = data[task_id]
    if not isinstance(block, dict) or "task_type" not in block:
        msg = f"Registry entry missing task_type: {path}"
        raise ValueError(msg)
    tt = block["task_type"]
    if not isinstance(tt, str) or not tt.strip():
        msg = f"Invalid task_type in {path}"
        raise ValueError(msg)
    return tt.strip()


def _iter_prediction_paths(results_root: Path) -> Iterator[Path]:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        msg = f"Not a directory: {results_root}"
        raise FileNotFoundError(msg)
    for path in results_root.rglob("predictions.jsonl"):
        if not path.is_file():
            continue
        if _task_id_from_pred_path(path) is None:
            continue
        if not _is_beans_zero_core_path(path):
            continue
        yield path


def _safe_dir_name(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_key.replace("/", "__"))


def _write_jsonl(objects: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in objects:
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSONL at {path}:{line_no}: {exc}"
            raise ValueError(msg) from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate beans_zero_core predictions per model under results/ingested, "
            "then rescore each eval task with the correct task_type."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "results" / "ingested",
        help="Root that contains campaign/model/.../suite/beans_zero_core/...",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "consolidated_beans_zero_core",
        help="Output directory for merged JSONL and per-task rescored summaries.",
    )
    parser.add_argument(
        "--skip-rescore",
        action="store_true",
        help="Only write inventory + merged files; do not run metrics.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model_key filters (e.g. i29/af3). Default: all discovered.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help="Optional model_key values to skip (e.g. a broken ingest run id).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow selecting the largest 1:1-aligned run when no full test split "
            "(registry row count) is available."
        ),
    )
    args = parser.parse_args()
    results_root: Path = args.results_root
    out_root: Path = args.out_dir
    model_filter = set(args.models) if args.models else None
    exclude_models = set(args.exclude_models) if args.exclude_models else set()
    expected_counts = _load_expected_row_counts()
    canonical_tasks = sorted(expected_counts.keys())

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []

    for pred_path in sorted(_iter_prediction_paths(results_root)):
        task_id = _task_id_from_pred_path(pred_path)
        assert task_id is not None
        mk = _model_key(pred_path, results_root)
        if model_filter is not None and mk not in model_filter:
            continue
        if mk in exclude_models:
            continue
        proc_path = pred_path.parent / "processed_predictions.jsonl"
        n_pred = _count_jsonl_rows(pred_path)
        n_proc = _count_jsonl_rows(proc_path) if proc_path.is_file() else -1
        prior = _read_summary_mean(pred_path.parent)
        exp_n = expected_counts.get(task_id)
        inventory.append(
            {
                "model_key": mk,
                "task_id": task_id,
                "expected_test_rows": exp_n,
                "predictions_path": str(pred_path),
                "processed_predictions_path": str(proc_path)
                if proc_path.is_file()
                else None,
                "n_predictions": n_pred,
                "n_processed": n_proc,
                "prior_metrics_mean": prior,
            }
        )
        if proc_path.is_file() and n_pred > 0:
            grouped[(mk, task_id)].append(pred_path)

    winners: dict[tuple[str, str], Path] = {}
    selection_details: dict[tuple[str, str], dict[str, Any]] = {}
    for key, paths in grouped.items():
        _mk, task_id = key
        if task_id not in expected_counts:
            msg = f"No expected row count for task {task_id!r} in registry JSON."
            raise KeyError(msg)
        chosen, meta = _select_winner(
            paths,
            task_id=task_id,
            expected_n=expected_counts[task_id],
            allow_partial=args.allow_partial,
        )
        selection_details[key] = meta
        if chosen is not None:
            winners[key] = chosen

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    model_keys = sorted({mk for (mk, _) in grouped} | {mk for (mk, _) in winners})
    metrics_table: list[dict[str, Any]] = []

    for mk in model_keys:
        slug = _safe_dir_name(mk)
        model_dir = out_root / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        merged_preds: list[dict[str, Any]] = []
        merged_proc: list[dict[str, Any]] = []
        manifest: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for task_id in canonical_tasks:
            key = (mk, task_id)
            exp_n = expected_counts[task_id]
            if key not in grouped:
                manifest.append(
                    {
                        "task_id": task_id,
                        "expected_test_rows": exp_n,
                        "skipped": True,
                        "reason": "no_prediction_artifacts_found",
                    }
                )
                continue
            detail = selection_details[key]
            if key not in winners:
                manifest.append(
                    {
                        "task_id": task_id,
                        "expected_test_rows": exp_n,
                        "skipped": True,
                        "selection": detail.get("selection"),
                        "reason": detail.get("reason"),
                        "candidates": detail.get("candidates"),
                    }
                )
                continue

            pred_path = winners[key]
            proc_path = pred_path.parent / "processed_predictions.jsonl"
            n_rows = _count_jsonl_rows(pred_path)
            manifest.append(
                {
                    "task_id": task_id,
                    "expected_test_rows": exp_n,
                    "actual_rows": n_rows,
                    "full_coverage": n_rows == exp_n,
                    "selection": detail.get("selection"),
                    "chosen_predictions_path": str(pred_path),
                    "chosen_processed_path": str(proc_path),
                    "warning": detail.get("warning"),
                }
            )
            for row in _read_jsonl(pred_path):
                sid = row["sample_id"]
                if sid in seen_ids:
                    msg = (
                        f"Duplicate sample_id {sid!r} when merging {mk} "
                        f"(task {task_id})"
                    )
                    raise ValueError(msg)
                seen_ids.add(sid)
                merged_preds.append(row)
            if not proc_path.is_file():
                msg = f"Missing processed_predictions for winner {pred_path}"
                raise FileNotFoundError(msg)
            merged_proc.extend(_read_jsonl(proc_path))

        (model_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pred_out = model_dir / "predictions.jsonl"
        proc_out = model_dir / "processed_predictions.jsonl"
        if merged_preds:
            _write_jsonl(merged_preds, pred_out)
            _write_jsonl(merged_proc, proc_out)
        else:
            for p in (pred_out, proc_out):
                if p.is_file():
                    p.unlink()

        if args.skip_rescore:
            for task_id in canonical_tasks:
                key = (mk, task_id)
                exp_n = expected_counts[task_id]
                if key not in winners:
                    metrics_table.append(
                        {
                            "model_key": mk,
                            "task_id": task_id,
                            "task_type": _load_task_type(task_id),
                            "expected_test_rows": exp_n,
                            "n_samples": 0,
                            "full_coverage": False,
                            "source_predictions_path": None,
                            "metrics_mean": None,
                            "prior_metrics_mean": None,
                            "rescored": False,
                            "skipped": True,
                            "skip_reason": selection_details.get(key, {}).get("reason")
                            if key in selection_details
                            else "no_prediction_artifacts_found",
                        }
                    )
                    continue
                wpred = winners[key]
                det = selection_details[key]
                metrics_table.append(
                    {
                        "model_key": mk,
                        "task_id": task_id,
                        "task_type": _load_task_type(task_id),
                        "expected_test_rows": exp_n,
                        "n_samples": _count_jsonl_rows(wpred),
                        "full_coverage": det.get("full_coverage", False),
                        "source_predictions_path": str(wpred),
                        "metrics_mean": None,
                        "prior_metrics_mean": _read_summary_mean(wpred.parent),
                        "rescored": False,
                        "skipped": False,
                    }
                )
            continue

        per_task_root = model_dir / "per_task_rescore"
        per_task_root.mkdir(parents=True, exist_ok=True)

        for task_id in canonical_tasks:
            key = (mk, task_id)
            exp_n = expected_counts[task_id]
            task_type = _load_task_type(task_id)
            if key not in winners:
                metrics_table.append(
                    {
                        "model_key": mk,
                        "task_id": task_id,
                        "task_type": task_type,
                        "expected_test_rows": exp_n,
                        "n_samples": 0,
                        "full_coverage": False,
                        "skipped": True,
                        "skip_reason": selection_details.get(key, {}).get("reason")
                        if key in selection_details
                        else "no_prediction_artifacts_found",
                    }
                )
                continue

            tdir = per_task_root / task_id
            tdir.mkdir(parents=True, exist_ok=True)
            winner_pred = winners[key]
            winner_proc = winner_pred.parent / "processed_predictions.jsonl"
            det = selection_details[key]
            if not winner_proc.is_file():
                metrics_table.append(
                    {
                        "model_key": mk,
                        "task_id": task_id,
                        "task_type": task_type,
                        "expected_test_rows": exp_n,
                        "skipped": True,
                        "error": "missing_processed_at_winner",
                    }
                )
                continue
            tp = tdir / "predictions.jsonl"
            pp = tdir / "processed_predictions.jsonl"
            shutil.copy2(winner_pred, tp)
            shutil.copy2(winner_proc, pp)
            rescore_predictions_file(tp, output_dir=tdir, task_type=task_type)
            summary_path = tdir / "summary.json"
            mean: dict[str, Any] | None = None
            n_task = _count_jsonl_rows(tp)
            if summary_path.is_file():
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                m = s.get("metrics", {})
                if isinstance(m, dict):
                    mm = m.get("mean")
                    if isinstance(mm, dict):
                        mean = dict(mm)
            metrics_table.append(
                {
                    "model_key": mk,
                    "task_id": task_id,
                    "task_type": task_type,
                    "expected_test_rows": exp_n,
                    "n_samples": n_task,
                    "full_coverage": bool(det.get("full_coverage")),
                    "selection": det.get("selection"),
                    "partial_warning": det.get("warning"),
                    "source_predictions_path": str(winner_pred),
                    "metrics_mean": mean,
                    "prior_metrics_mean": _read_summary_mean(winner_pred.parent),
                    "rescored": True,
                    "skipped": False,
                }
            )

    (out_root / "metrics_per_task.json").write_text(
        json.dumps(metrics_table, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
