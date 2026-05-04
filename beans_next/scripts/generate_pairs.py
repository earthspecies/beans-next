"""Generate prompt/answer/ground-truth pairs for BeansPro subsets.

This script is meant for analysis and debugging: it saves, for each sampled
dataset row, the rendered prompt messages, the raw model output(s), the
post-processed prediction text, and the ground-truth label(s).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from beans_next.api.http_schemas import PredictionsV1Request
from beans_next.api.types import DatasetExample
from beans_next.models.http import HttpClient
from beans_next.post_process.pipeline import (
    PostProcessResult,
    run_post_process_pipeline,
)
from beans_next.prompts.renderer import PromptRenderer, load_builtin_prompt_yaml
from beans_next.runner.runner import (
    _postprocess_steps_for_examples,
    _raw_prediction_text,
    model_request_to_wire_item,
    wire_response_item_to_model_prediction,
)

_DEFAULT_K: Final[int] = 100
_DEFAULT_SEED: Final[int] = 0
_DEFAULT_BATCH_SIZE: Final[int] = 1
_PROGRESS_EVERY: Final[int] = 10


@dataclass(frozen=True, slots=True)
class SubsetTaskSpec:
    """Resolved eval-task metadata for one BeansPro subset."""

    subset: str
    split: str
    task_id: str | None
    task_type: str | None
    prompt_name: str


def _registry_root() -> Path:
    return Path(__file__).resolve().parent.parent / "registry"


def _load_yaml_file(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_beans_next_subsets_from_registry() -> list[str]:
    """Load BeansPro subset ids from the bundled dataset registry YAML."""
    path = _registry_root() / "dataset" / "beans_next_esp.yaml"
    raw = _load_yaml_file(path)
    if not isinstance(raw, Mapping) or "beans_next_esp" not in raw:
        raise ValueError(f"Unexpected dataset registry shape: {path}")
    doc = raw["beans_next_esp"]
    if not isinstance(doc, Mapping):
        raise ValueError(f"Unexpected dataset registry document: {path}")
    subsets = doc.get("subsets")
    if not isinstance(subsets, list) or not subsets:
        raise ValueError(f"beans_next_esp.subsets missing/empty in {path}")
    out: list[str] = []
    for item in subsets:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    if not out:
        raise ValueError(f"No valid subset strings in {path}")
    return out


def _load_beans_next_eval_task_specs() -> dict[str, Mapping[str, Any]]:
    """Load eval-task YAML docs for BeansPro into mapping keyed by subset."""
    root = _registry_root() / "eval_task"
    out: dict[str, Mapping[str, Any]] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.yaml")):
        raw = _load_yaml_file(path)
        if not isinstance(raw, Mapping) or len(raw) != 1:
            continue
        task_id, doc = next(iter(raw.items()))
        if not isinstance(task_id, str) or not isinstance(doc, Mapping):
            continue
        if doc.get("dataset") != "beans_next":
            continue
        subset = doc.get("subset")
        if not isinstance(subset, str) or not subset.strip():
            continue
        out[subset.strip()] = {"task_id": task_id, **doc}
    return out


def _resolve_subset_task_specs(
    subsets: Iterable[str],
    *,
    default_prompt_name: str,
) -> list[SubsetTaskSpec]:
    eval_specs = _load_beans_next_eval_task_specs()
    out: list[SubsetTaskSpec] = []
    for subset in subsets:
        doc = eval_specs.get(subset)
        if doc is None:
            out.append(
                SubsetTaskSpec(
                    subset=subset,
                    split=subset,
                    task_id=None,
                    task_type=None,
                    prompt_name=default_prompt_name,
                )
            )
            continue
        prompt = doc.get("prompt") or doc.get("prompt_yaml")
        prompt_name = (
            str(prompt).strip()
            if isinstance(prompt, str) and str(prompt).strip()
            else default_prompt_name
        )
        task_id = doc.get("task_id")
        out.append(
            SubsetTaskSpec(
                subset=subset,
                split=str(doc.get("split") or subset),
                task_id=str(task_id).strip() if isinstance(task_id, str) else None,
                task_type=str(doc.get("task_type")).strip()
                if isinstance(doc.get("task_type"), str)
                else None,
                prompt_name=prompt_name,
            )
        )
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl_append(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _ensure_dir_empty_or_create(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_existing_sample_ids(sample_ids_dir: Path) -> dict[str, set[str]]:
    """Load sample ids previously written under sample_ids/.

    Returns a mapping of subset -> set(sample_id).
    """
    out: dict[str, set[str]] = {}
    if not sample_ids_dir.is_dir():
        return out
    for path in sorted(sample_ids_dir.glob("*.jsonl")):
        subset = path.stem
        ids: set[str] = set()
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                sid = line.strip()
                if sid:
                    ids.add(sid)
        except OSError:
            continue
        out[subset] = ids
    return out


def _scan_pairs_for_completed_ids(pairs_path: Path) -> dict[str, set[str]]:
    """Fallback: scan pairs.jsonl for completed (subset, sample_id) entries."""
    out: dict[str, set[str]] = {}
    if not pairs_path.is_file():
        return out
    try:
        with pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                subset = obj.get("subset")
                sample_id = obj.get("sample_id")
                if isinstance(subset, str) and isinstance(sample_id, str):
                    out.setdefault(subset, set()).add(sample_id)
    except OSError:
        return out
    return out


def _print(msg: str) -> None:
    print(msg, flush=True)


def _sample_first_k(rows: Iterator[DatasetExample], *, k: int) -> list[DatasetExample]:
    out: list[DatasetExample] = []
    for ex in rows:
        out.append(ex)
        if len(out) >= k:
            break
    return out


def _sample_reservoir_k(
    rows: Iterator[DatasetExample],
    *,
    k: int,
    seed: int,
) -> list[DatasetExample]:
    rng = random.Random(int(seed))
    reservoir: list[DatasetExample] = []
    n = 0
    for ex in rows:
        n += 1
        if len(reservoir) < k:
            reservoir.append(ex)
            continue
        j = rng.randrange(n)
        if j < k:
            reservoir[j] = ex
    return reservoir


def _load_beans_next_examples(
    spec: SubsetTaskSpec,
    *,
    k: int,
    sample_strategy: str,
    seed: int,
    workers: int,
) -> list[DatasetExample]:
    from beans_next.datasets.esp_data import iter_esp_data_beans_next_examples

    it = iter_esp_data_beans_next_examples(
        subset=spec.subset,
        split=spec.split,
        task_id=spec.task_id,
        limit=None,
        workers=workers,
    )
    if sample_strategy == "first":
        return _sample_first_k(it, k=k)
    if sample_strategy == "reservoir":
        return _sample_reservoir_k(it, k=k, seed=seed)
    raise ValueError(f"Unknown sample strategy: {sample_strategy!r}")


def _postprocess_for_task(
    *,
    examples: list[DatasetExample],
    raw_text: str,
    task_type: str | None,
) -> PostProcessResult:
    parsers, cleaners = _postprocess_steps_for_examples(
        examples, task_type=task_type, labels_override=None
    )
    return run_post_process_pipeline(
        raw_text,
        parser_steps=parsers,
        cleaner_steps=cleaners,
    )


def generate_pairs_main(args: argparse.Namespace) -> int:
    """CLI entrypoint for `beans-next pairs`."""
    subsets_all = _load_beans_next_subsets_from_registry()
    if args.subsets:
        subset_set = {s.strip() for s in args.subsets.split(",") if s.strip()}
        subsets = [s for s in subsets_all if s in subset_set]
        missing = sorted(subset_set - set(subsets))
        if missing:
            raise SystemExit(f"Unknown BeansPro subset(s): {missing!r}")
    else:
        subsets = subsets_all

    run_id = (args.run_id or "").strip()
    if not run_id:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"beans_next_pairs_{args.model_tag}_{ts}"

    out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if out_root is None:
        out_root = (
            Path.cwd()
            / "results"
            / "prompt_answer_pairs"
            / "beans_next_v0_1_0"
            / run_id
        ).resolve()
    _ensure_dir_empty_or_create(out_root)

    sample_ids_dir = out_root / "sample_ids"
    pairs_path = out_root / "pairs.jsonl"
    manifest_path = out_root / "manifest.json"
    resume = bool(getattr(args, "resume", False))

    existing_ids = _load_existing_sample_ids(sample_ids_dir) if resume else {}
    if resume and not existing_ids:
        existing_ids = _scan_pairs_for_completed_ids(pairs_path)

    subset_specs = _resolve_subset_task_specs(
        subsets,
        default_prompt_name=str(args.prompt),
    )

    seed = int(args.seed)
    k = int(args.k)
    sample_strategy = str(args.sample_strategy)

    with HttpClient(
        str(args.predict_url),
        probe_on_init=True,
        timeout=float(args.http_timeout_sec),
    ) as client:
        server_info = client.server_info
        if server_info is None:
            server_info = client.probe_info()

        manifest = {
            "schema_version": "beans_next.pairs_manifest.v1",
            "run_id": run_id,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_tag": str(args.model_tag),
            "predict_url": client.predict_url,
            "server_info": server_info,
            "k_per_subset": k,
            "seed": seed,
            "sample_strategy": sample_strategy,
            "subsets": [s.subset for s in subset_specs],
            "argv": sys.argv[1:],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        batch_size = max(1, int(args.batch_size))
        for spec in subset_specs:
            _print(f"[pairs] subset={spec.subset} split={spec.split} task={spec.task_id}")
            examples = _load_beans_next_examples(
                spec,
                k=k,
                sample_strategy=sample_strategy,
                seed=seed,
                workers=int(args.workers),
            )
            if not examples:
                continue

            # Persist chosen sample ids for reproducibility.
            sid_path = sample_ids_dir / f"{spec.subset}.jsonl"
            sid_path.parent.mkdir(parents=True, exist_ok=True)
            if not sid_path.exists():
                with sid_path.open("w", encoding="utf-8") as f:
                    for ex in examples:
                        f.write(ex.sample_id + "\n")

            prompt_spec_filename = spec.prompt_name
            if not prompt_spec_filename.endswith(".yaml"):
                prompt_spec_filename = f"{prompt_spec_filename}.yaml"
            renderer = PromptRenderer(load_builtin_prompt_yaml(prompt_spec_filename))

            rendered: list[tuple[DatasetExample, Any]] = []
            completed_for_subset = existing_ids.get(spec.subset, set())
            for ex in examples:
                if ex.sample_id in completed_for_subset:
                    continue
                rendered.append((ex, renderer.render(ex)))

            if resume and completed_for_subset:
                _print(
                    f"[pairs] subset={spec.subset} resume_skipped={len(completed_for_subset)}"
                )
            _print(f"[pairs] subset={spec.subset} queued={len(rendered)}")

            # Send in batches.
            for start in range(0, len(rendered), batch_size):
                chunk = rendered[start : start + batch_size]
                wire_items = [model_request_to_wire_item(mr) for _ex, mr in chunk]
                envelope = PredictionsV1Request(requests=wire_items)
                t0 = time.perf_counter()
                response = client.generate(envelope)
                roundtrip = time.perf_counter() - t0

                by_id = {r.sample_id: r for r in response.responses}
                for ex, mr in chunk:
                    item = by_id.get(ex.sample_id)
                    if item is None:
                        continue
                    pred = wire_response_item_to_model_prediction(
                        item, server_info=client.server_info
                    )
                    raw_text = _raw_prediction_text(pred)
                    post = _postprocess_for_task(
                        examples=examples, raw_text=raw_text, task_type=spec.task_type
                    )

                    # Prefer saving paths/hashes instead of base64 payloads.
                    audio_inputs = []
                    for slot in mr.audio_inputs:
                        audio_rec: dict[str, Any] = {
                            "payload_type": slot.payload_type,
                            "sample_rate_hz": slot.sample_rate,
                        }
                        if slot.payload_type == "file_path":
                            try:
                                p = Path(str(slot.data)).expanduser().resolve()
                            except Exception:  # noqa: BLE001
                                p = None
                            if p is not None and p.is_file():
                                audio_rec["source_path"] = str(p)
                                audio_rec["sha256"] = _sha256_file(p)
                            else:
                                audio_rec["source_path"] = str(slot.data)
                        else:
                            audio_rec["source_path"] = None
                        audio_inputs.append(audio_rec)

                    rec = {
                        "schema_version": "beans_next.prompt_answer_pair.v1",
                        "run_id": run_id,
                        "model_tag": str(args.model_tag),
                        "predict_url": client.predict_url,
                        "server_info": client.server_info,
                        "subset": spec.subset,
                        "split": spec.split,
                        "task_id": ex.task_id,
                        "task_type": spec.task_type,
                        "sample_id": ex.sample_id,
                        "ground_truth": ex.labels,
                        "prompt_id": renderer.prompt_id,
                        "prompt_messages": [
                            {"role": m.role, "content": m.content} for m in mr.messages
                        ],
                        "audio_inputs": audio_inputs,
                        "raw_predictions": list(item.predictions),
                        "processed_prediction": post.text,
                        "postprocess_warnings": list(post.warnings),
                        "finish_reason": item.finish_reason,
                        "usage": item.usage.model_dump(mode="json") if item.usage else None,
                        "latency_sec": item.latency_sec,
                        "batch_roundtrip_sec": roundtrip,
                        "error": item.error,
                    }
                    _jsonl_append(pairs_path, rec)
                if (start // batch_size + 1) % _PROGRESS_EVERY == 0:
                    _print(
                        f"[pairs] subset={spec.subset} progress={min(start + batch_size, len(rendered))}/{len(rendered)}"
                    )
            _print(f"[pairs] subset={spec.subset} done")

    return 0


def build_pairs_arg_parser(parser: argparse.ArgumentParser) -> None:
    """Attach the `pairs` subcommand arguments to `parser`."""
    parser.add_argument(
        "--predict-url",
        required=True,
        help="Full URL to the launcher POST /predict endpoint.",
    )
    parser.add_argument(
        "--model-tag",
        required=True,
        help="Short label for the model endpoint (used in output naming).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id (directory name). Defaults to timestamped name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Append into an existing output dir and skip sample_ids already present "
            "under output_dir/sample_ids (or in pairs.jsonl as a fallback)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: results/prompt_answer_pairs/beans_next_v0_1_0/<run_id>/).",
    )
    parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated BeansPro subsets to include (default: all).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=_DEFAULT_K,
        help="Number of examples per subset (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Sampling seed (used by reservoir strategy).",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("first", "reservoir"),
        default="first",
        help="Sampling strategy per subset (default: first).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel threads for esp_data GCS audio downloads (default: 1).",
    )
    parser.add_argument(
        "--prompt",
        default="classification_beans_zero_official_v1",
        help="Prompt registry name (default: classification_beans_zero_official_v1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Batch size for POST /predict (default: 1).",
    )
    parser.add_argument(
        "--http-timeout-sec",
        type=float,
        default=float(os.environ.get("BEANS_PRO_HTTP_TIMEOUT_SEC", "120") or 120),
        help="HttpClient timeout seconds (default: env BEANS_PRO_HTTP_TIMEOUT_SEC else 120).",
    )


__all__ = ["build_pairs_arg_parser", "generate_pairs_main"]
