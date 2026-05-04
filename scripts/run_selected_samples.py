"""Run inference for a pre-selected set of sample_ids and save prompt+answers.

This is intended for building stable fixtures: pick a set of samples once, then
rerun the same sample_ids across multiple models/servers and store comparable
artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beans_next.api.http_schemas import PredictionsV1Request
from beans_next.api.types import DatasetExample
from beans_next.datasets.esp_data import iter_esp_data_beans_zero_examples
from beans_next.models.http import HttpClient
from beans_next.post_process.pipeline import run_post_process_pipeline
from beans_next.prompts.renderer import (
    PromptRenderer,
    builtin_prompt_registry_path,
    load_prompt_spec_from_path,
)
from beans_next.runner.runner import (
    _coerce_eval_task_mapping,
    _eval_task_yaml_path,
    _labels_for_eval_task,
    _load_yaml_mapping,
    _postprocess_steps_for_examples,
    model_request_to_wire_item,
    wire_response_item_to_model_prediction,
)


@dataclass(frozen=True)
class SelectedRow:
    """One selected dataset row to run inference on."""

    task_id: str
    subset: str
    split: str
    sample_id: str


def _load_eval_task_cfg(task_id: str) -> Mapping[str, Any]:
    path = _eval_task_yaml_path(task_id)
    raw = _load_yaml_mapping(path)
    return _coerce_eval_task_mapping(raw, source=path)


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


def _read_selected(path: Path) -> list[SelectedRow]:
    rows: list[SelectedRow] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        task_id = str(obj["task_id"])
        subset = str(obj["subset"])
        split = str(obj.get("split") or "test")
        sample_id = str(obj["sample_id"])
        rows.append(
            SelectedRow(
                task_id=task_id,
                subset=subset,
                split=split,
                sample_id=sample_id,
            )
        )
    if not rows:
        raise SystemExit(f"No rows in selected file: {path}")
    return rows


def _find_examples(
    rows: Sequence[SelectedRow],
    *,
    workers: int,
) -> list[DatasetExample]:
    # Group by (subset, split, task_id) so we can scan each dataset stream once.
    by_key: dict[tuple[str, str, str], list[SelectedRow]] = {}
    for r in rows:
        by_key.setdefault((r.subset, r.split, r.task_id), []).append(r)

    found: dict[str, DatasetExample] = {}
    for (subset, split, task_id), wanted in by_key.items():
        wanted_ids = {w.sample_id for w in wanted}
        it = iter_esp_data_beans_zero_examples(
            subset=subset,
            split=split,
            task_id=task_id,
            limit=None,
            workers=workers,
        )
        for ex in it:
            if ex.sample_id in wanted_ids:
                found[ex.sample_id] = ex
                if len(found) >= len(rows):
                    break
        missing = wanted_ids - set(found.keys())
        if missing:
            raise SystemExit(
                f"Missing {len(missing)} sample_id(s) for task_id={task_id!r} "
                f"subset={subset!r}: "
                f"{sorted(list(missing))[:5]!r}"
            )

    # Return in the same order as input file.
    return [found[r.sample_id] for r in rows]


def _write_jsonl(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(dict(obj), ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-selected-samples",
        description=(
            "Run inference for pre-selected sample_ids and store prompt+answers."
        ),
    )
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--predict-url", default="")
    parser.add_argument("--url-file", type=Path, default=None)
    parser.add_argument("--esp-workers", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    predict_url = _predict_url_from_args(args)
    selected = _read_selected(Path(args.selected_jsonl))
    examples = _find_examples(selected, workers=max(1, int(args.esp_workers)))

    # Per-task wiring: prompt + postprocess depends on eval-task YAML.
    per_task: dict[str, dict[str, Any]] = {}
    for row in selected:
        if row.task_id in per_task:
            continue
        cfg = _load_eval_task_cfg(row.task_id)
        per_task[row.task_id] = dict(cfg)

    out_rows: list[dict[str, Any]] = []
    with HttpClient(predict_url, probe_on_init=True) as client:
        for i, (sel_row, ex) in enumerate(zip(selected, examples, strict=True), 1):
            print(
                f"[run_selected_samples] {i}/{len(selected)} "
                f"task_id={sel_row.task_id} sample_id={sel_row.sample_id}",
                flush=True,
            )
            cfg = per_task[sel_row.task_id]
            prompt_id = str(cfg["prompt"])
            task_type = cfg.get("task_type")
            prompt_path = (
                builtin_prompt_registry_path() / f"{prompt_id}.yaml"
            ).resolve()
            renderer = PromptRenderer(load_prompt_spec_from_path(prompt_path))

            labels_override = _labels_for_eval_task(cfg)
            parsers, cleaners = _postprocess_steps_for_examples(
                [ex],
                task_type=str(task_type) if isinstance(task_type, str) else None,
                labels_override=labels_override,
            )

            mr = renderer.render(ex)
            wire = model_request_to_wire_item(mr)
            resp = client.generate(PredictionsV1Request(requests=[wire]))
            pred = wire_response_item_to_model_prediction(
                resp.responses[0],
                server_info=client.server_info,
            )
            raw = pred.predictions[0] if pred.predictions else ""
            post = run_post_process_pipeline(
                raw,
                parser_steps=parsers,
                cleaner_steps=cleaners,
            )

            out_rows.append(
                {
                    "task_id": sel_row.task_id,
                    "subset": sel_row.subset,
                    "split": sel_row.split,
                    "sample_id": sel_row.sample_id,
                    "ground_truth": ex.labels,
                    "prompt": [
                        {"role": m.role, "content": m.content}
                        for m in mr.messages
                    ],
                    "raw_answer": raw,
                    "answer": post.text,
                    "warnings": list(post.warnings),
                    "server_info": dict(client.server_info or {}),
                }
            )
            print(
                f"[run_selected_samples] done {i}/{len(selected)} "
                f"processed={post.text!r}",
                flush=True,
            )

    _write_jsonl(Path(args.out_jsonl), out_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
