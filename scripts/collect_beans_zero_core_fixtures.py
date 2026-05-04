"""Collect local fixture runs for BEANS-Zero core.

This script selects a small, curated slice of examples per BEANS-Zero core task and
runs inference against a provided launcher endpoint. The resulting artifacts are
saved locally so they can be promoted into stable test fixtures later.

Selection policy
----------------
- For all tasks: select 20 examples in dataset order.
- For detection tasks: ensure at least 10/20 examples have non-empty targets
  (i.e., not empty and not ``"None"``), scanning forward until enough positives
  are found.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beans_next.api.types import DatasetExample
from beans_next.datasets.esp_data import iter_esp_data_beans_zero_examples
from beans_next.models.http import HttpClient
from beans_next.prompts.renderer import PromptRenderer, builtin_prompt_registry_path
from beans_next.prompts.renderer import load_prompt_spec_from_path as load_prompt_spec
from beans_next.runner.runner import (
    BenchmarkRunner,
    RunnerConfig,
    _coerce_eval_task_mapping,
    _eval_task_yaml_path,
    _labels_for_eval_task,
    _load_yaml_mapping,
    _postprocess_steps_for_examples,
    _suite_yaml_path,
)


@dataclass(frozen=True)
class TaskSpec:
    """Resolved eval-task metadata needed for selection and inference."""

    eval_task_id: str
    subset: str
    split: str
    task_type: str | None
    prompt: str


def _iter_suite_eval_task_ids(*, suite_id: str) -> list[str]:
    suite_path = _suite_yaml_path(suite_id)
    suite_raw = _load_yaml_mapping(suite_path)
    suite = suite_raw.get(suite_id)
    if not isinstance(suite, Mapping):
        raise SystemExit(f"Suite YAML missing key {suite_id!r}: {suite_path}")
    eval_tasks = suite.get("eval_tasks")
    if not isinstance(eval_tasks, list) or not eval_tasks:
        raise SystemExit(f"Suite YAML has no eval_tasks list: {suite_path}")
    out: list[str] = []
    for item in eval_tasks:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    if not out:
        raise SystemExit(
            f"Suite eval_tasks list was empty after filtering: {suite_path}"
        )
    return out


def _load_task_spec(eval_task_id: str) -> TaskSpec:
    path = _eval_task_yaml_path(eval_task_id)
    raw = _load_yaml_mapping(path)
    cfg = _coerce_eval_task_mapping(raw, source=path)
    subset = cfg.get("subset")
    split = cfg.get("split")
    prompt = cfg.get("prompt")
    if not isinstance(subset, str) or not subset.strip():
        raise SystemExit(f"Eval task {eval_task_id!r} missing subset: {path}")
    if not isinstance(split, str) or not split.strip():
        raise SystemExit(f"Eval task {eval_task_id!r} missing split: {path}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SystemExit(f"Eval task {eval_task_id!r} missing prompt: {path}")
    task_type = cfg.get("task_type")
    task_type_s = (
        task_type.strip()
        if isinstance(task_type, str) and task_type.strip()
        else None
    )
    return TaskSpec(
        eval_task_id=eval_task_id,
        subset=subset.strip(),
        split=split.strip(),
        task_type=task_type_s,
        prompt=prompt.strip(),
    )


def _labels_non_empty(labels: object) -> bool:
    if labels is None:
        return False
    if isinstance(labels, str):
        toks = [t.strip() for t in labels.split(",")]
        return any(t and t.lower() != "none" for t in toks)
    if isinstance(labels, list):
        toks = [str(t).strip() for t in labels]
        return any(t and t.lower() != "none" for t in toks)
    return bool(str(labels).strip())


def _select_examples(
    it: Iterable[DatasetExample],
    *,
    n_total: int,
    min_positive: int,
    max_scan: int,
) -> list[DatasetExample]:
    positives: list[DatasetExample] = []
    negatives: list[DatasetExample] = []
    scanned = 0
    for ex in it:
        scanned += 1
        if scanned > max_scan:
            break
        if _labels_non_empty(ex.labels):
            if len(positives) < n_total:
                positives.append(ex)
        else:
            if len(negatives) < n_total:
                negatives.append(ex)
        if len(positives) >= min_positive and (
            len(positives) + len(negatives)
        ) >= n_total:
            break

    selected: list[DatasetExample] = []
    if min_positive > 0:
        selected.extend(positives[:min_positive])
        selected.extend(negatives[: max(0, n_total - len(selected))])
        if len(selected) < n_total:
            # Not enough negatives; fill with remaining positives.
            selected.extend(positives[len(selected) : n_total])
    else:
        selected.extend((positives + negatives)[:n_total])
    return selected[:n_total]


def _write_selected_jsonl(path: Path, examples: Sequence[DatasetExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row: dict[str, Any] = {
                "sample_id": ex.sample_id,
                "task_id": ex.task_id,
                "split": ex.split,
                "labels": ex.labels,
            }
            # Keep metadata light; audio paths can be large but are useful for auditing.
            if isinstance(ex.metadata, dict):
                audio_path = ex.metadata.get("audio_path")
                if isinstance(audio_path, str) and audio_path:
                    row["audio_path"] = audio_path
                dataset_name = ex.metadata.get("dataset_name")
                if isinstance(dataset_name, str) and dataset_name:
                    row["dataset_name"] = dataset_name
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _predict_url_from_args(args: argparse.Namespace) -> str:
    url = str(getattr(args, "predict_url", "") or "").strip()
    if url:
        return url
    url_file = getattr(args, "url_file", None)
    if url_file is None:
        raise SystemExit("--predict-url or --url-file is required")
    p = Path(url_file).expanduser()
    if not p.is_file():
        raise SystemExit(f"--url-file not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"--url-file was empty: {p}")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collect-beans-zero-core-fixtures",
        description="Select 20 examples per BEANS-Zero core task and run inference.",
    )
    parser.add_argument(
        "--suite-id",
        default="beans_zero_core",
        help="Suite id under beans_next/registry/suite (default: beans_zero_core).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/fixtures/beans_zero_core_20each"),
        help="Output directory root for all fixture runs.",
    )
    parser.add_argument(
        "--predict-url",
        default="",
        help="Launcher POST /predict endpoint URL.",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        default=None,
        help="File containing the launcher predict URL (first line).",
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=20,
        help="Number of examples to select per task (default: 20).",
    )
    parser.add_argument(
        "--min-detection-positive",
        type=int,
        default=10,
        help='Minimum non-"None" targets for detection tasks (default: 10).',
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=2000,
        help="Max examples to scan per task when searching for positives.",
    )
    parser.add_argument(
        "--esp-workers",
        type=int,
        default=1,
        help="esp_data audio download workers (default: 1).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    n_total = max(1, int(args.n_total))
    min_det = max(0, int(args.min_detection_positive))
    max_scan = max(1, int(args.max_scan))
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    predict_url = _predict_url_from_args(args)

    task_ids = _iter_suite_eval_task_ids(suite_id=str(args.suite_id))
    task_specs: list[TaskSpec] = [_load_task_spec(tid) for tid in task_ids]

    with HttpClient(predict_url, probe_on_init=True) as client:
        for spec in task_specs:
            task_out = out_root / spec.eval_task_id
            task_out.mkdir(parents=True, exist_ok=True)

            examples_it = iter_esp_data_beans_zero_examples(
                subset=spec.subset,
                split=spec.split,
                task_id=spec.eval_task_id,
                limit=None,
                workers=max(1, int(args.esp_workers)),
            )
            want_pos = (
                min_det
                if (spec.task_type or "").lower().find("detection") >= 0
                else 0
            )
            selected = _select_examples(
                examples_it,
                n_total=n_total,
                min_positive=min(want_pos, n_total),
                max_scan=max_scan,
            )
            _write_selected_jsonl(task_out / "selected_examples.jsonl", selected)

            # Prompt + post-process pipeline for this task.
            prompt_path = (
                builtin_prompt_registry_path() / f"{spec.prompt}.yaml"
            ).resolve()
            prompt_spec = load_prompt_spec(prompt_path)
            renderer = PromptRenderer(prompt_spec)

            task_yaml = _load_yaml_mapping(_eval_task_yaml_path(spec.eval_task_id))
            task_cfg = _coerce_eval_task_mapping(
                task_yaml,
                source=_eval_task_yaml_path(spec.eval_task_id),
            )
            labels_override = _labels_for_eval_task(task_cfg)
            parsers, cleaners = _postprocess_steps_for_examples(
                selected,
                task_type=spec.task_type,
                labels_override=labels_override,
            )
            cfg = RunnerConfig(
                output_dir=task_out,
                run_id=str(out_root.name),
                task_type=spec.task_type,
                parser_steps=parsers,
                cleaner_steps=cleaners,
            )
            runner = BenchmarkRunner(client, renderer, cfg)
            runner.run(list(selected))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
