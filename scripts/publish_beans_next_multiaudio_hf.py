"""Publish BeansProMultiAudio rows to the Hugging Face Hub (BEANS-Next tier 4).

Builds **one** Hub dataset with **one** builder config (default) and a single
``test`` split. Each row includes ``tier`` and ``subset`` so consumers can filter
without separate Hub configs or splits.

Source metadata and audio roots match ``beans_next/datasets/beans_next_multiaudio.py``.

Notes
-----
- Intended for manual runs with GCS + Hub credentials.
- Audio is streamed to short-lived local files, then embedded on push
  (``embed_external_files=True``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import HfApi


@dataclass(frozen=True)
class SourceSplitSpec:
    """One multiaudio JSONL source (subset id + GCS paths).

    Attributes
    ----------
    subset
        Subset id stored in column ``subset`` (and ``task`` for compatibility).
    jsonl_path
        Source JSONL path (GCS).
    audio_root
        Base ``gs://`` prefix for ``audio_paths`` in that JSONL.
    """

    subset: str
    jsonl_path: str
    audio_root: str


_GCS_BASE = "gs://esp-data-ingestion/beans-next/v0.1.0/raw"
_BEANS_ZERO_GCS_BASE = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw"

_SOURCE_SPLITS: list[SourceSplitSpec] = [
    SourceSplitSpec(
        subset="gibbon-fewshot-detection-balanced",
        jsonl_path=f"{_GCS_BASE}/gibbon_fewshot_detection_balanced/test.jsonl",
        audio_root=f"{_BEANS_ZERO_GCS_BASE}/",
    ),
    SourceSplitSpec(
        subset="giant-otter-4way",
        jsonl_path=f"{_GCS_BASE}/giant_otter_4way/test.jsonl",
        audio_root=f"{_GCS_BASE}/",
    ),
    SourceSplitSpec(
        subset="dcase-fewshot-detection-balanced",
        jsonl_path=f"{_GCS_BASE}/dcase_fewshot_detection_balanced/test.jsonl",
        audio_root=f"{_BEANS_ZERO_GCS_BASE}/",
    ),
    SourceSplitSpec(
        subset="crow-4way",
        jsonl_path=f"{_GCS_BASE}/crow_4way/test.jsonl",
        audio_root=f"{_GCS_BASE}/carrion_crow_descriptions/",
    ),
    SourceSplitSpec(
        subset="zebra-4way",
        jsonl_path=f"{_GCS_BASE}/zebra_4way/test.jsonl",
        audio_root=f"{_GCS_BASE}/zebra_descriptions/",
    ),
    SourceSplitSpec(
        subset="unseen-species-4way",
        jsonl_path=f"{_GCS_BASE}/unseen_species_4way/test.jsonl",
        audio_root=f"{_BEANS_ZERO_GCS_BASE}/",
    ),
]


def _require_fsspec() -> Any:
    try:
        import fsspec  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "This publisher requires `fsspec` and `gcsfs` for `gs://` access. "
            "Install them in your environment (e.g. `uv pip install fsspec gcsfs`)."
        ) from exc
    return fsspec


def _iter_jsonl(path: str) -> Iterator[Mapping[str, Any]]:
    fsspec = _require_fsspec()
    with fsspec.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if isinstance(row, dict):
                yield row


def _download_to(
    *,
    gcs_abs: str,
    out_path: Path,
) -> None:
    """Stream one GCS object to a local file path."""
    fsspec = _require_fsspec()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(gcs_abs, "rb") as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def _normalized_relpath(p: str) -> str:
    s = str(p).strip().lstrip("/")
    if not s:
        raise ValueError("empty relative path")
    return s


def _build_records_for_source_split(
    spec: SourceSplitSpec,
    *,
    tier: str,
    work_dir: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    audio_dir = work_dir / "audio" / spec.subset
    records: list[dict[str, Any]] = []
    kept = 0
    for row in _iter_jsonl(spec.jsonl_path):
        audio_paths_raw = row.get("audio_paths")
        if not isinstance(audio_paths_raw, list) or not audio_paths_raw:
            continue

        local_audio_paths: list[str] = []
        rel_audio_paths: list[str] = []
        for rel in audio_paths_raw:
            if not isinstance(rel, str) or not rel.strip():
                continue
            rel_norm = _normalized_relpath(rel)
            rel_audio_paths.append(rel_norm)
            out_path = audio_dir / rel_norm
            if not out_path.exists():
                gcs_abs = spec.audio_root.rstrip("/") + "/" + rel_norm
                _download_to(gcs_abs=gcs_abs, out_path=out_path)
            local_audio_paths.append(str(out_path))

        if not local_audio_paths:
            continue

        rec: dict[str, Any] = dict(row)
        rec["tier"] = tier
        rec["subset"] = spec.subset
        rec["audio_paths"] = rel_audio_paths
        rec["audio"] = local_audio_paths
        rec["task"] = spec.subset
        for key in ("dataset_name", "source_dataset", "license", "template_path", "id"):
            val = rec.get(key)
            if val is None:
                rec[key] = ""
            elif not isinstance(val, str):
                rec[key] = str(val)
        records.append(rec)
        kept += 1
        if limit is not None and kept >= limit:
            break
    return records


def _dataset_card(repo_id: str) -> str:
    return (
        f"---\n"
        f"license: cc-by-nc-sa-4.0\n"
        f"---\n\n"
        f"# BEANS-Next (multiaudio tier)\n\n"
        f"This repository uses a **single** dataset config and a single ``test`` split. "
        f"Filter by columns ``tier`` and ``subset`` (for example "
        f"``tier=\"tier_4_in_context\"`` and ``subset=\"crow-4way\"``).\n\n"
        f"## Usage\n\n"
        f"```python\n"
        f"from datasets import load_dataset\n\n"
        f"ds = load_dataset(\"{repo_id}\", split=\"test\")\n"
        f"row = ds[0]\n"
        f"print(row[\"tier\"], row[\"subset\"])\n"
        f"print(len(row[\"audio\"]))\n"
        f"```\n"
    )


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id",
        default="EarthSpeciesProject/BEANS-Next",
        help="Target Hub dataset repo id.",
    )
    p.add_argument(
        "--tier",
        default="tier_4_in_context",
        help="Value written to the ``tier`` column for published rows.",
    )
    p.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Optional subset ids to publish (defaults to all multiaudio subsets).",
    )
    p.add_argument(
        "--limit-per-subset",
        type=int,
        default=None,
        help="Optional cap on examples per subset (smoke tests).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create/update the repo as private.",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var).",
    )
    p.add_argument(
        "--keep-work-dir",
        default=None,
        help=(
            "Optional directory to retain downloaded audio for debugging. "
            "When unset, a temporary directory is removed after upload."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    token = args.token
    if token is not None and not str(token).strip():
        token = None

    tier = str(args.tier).strip()
    if not tier:
        raise SystemExit("--tier must be non-empty")

    target = {s for s in (args.subsets or []) if str(s).strip()}
    split_specs = [
        s for s in _SOURCE_SPLITS if not target or s.subset in target
    ]
    if not split_specs:
        known = ", ".join(s.subset for s in _SOURCE_SPLITS)
        raise SystemExit(f"No subsets selected. Known subset ids: {known}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type="dataset",
        private=bool(args.private),
        exist_ok=True,
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        path_or_fileobj=_dataset_card(args.repo_id).encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Update dataset card",
    )

    def _collect_all_records(work_root: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for spec in split_specs:
            part = _build_records_for_source_split(
                spec,
                tier=tier,
                work_dir=work_root,
                limit=args.limit_per_subset,
            )
            out.extend(part)
        return out

    if args.keep_work_dir:
        work_root = Path(args.keep_work_dir).expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        all_records = _collect_all_records(work_root)
    else:
        with tempfile.TemporaryDirectory(prefix="beans-next-beans-next-publish-") as td:
            all_records = _collect_all_records(Path(td))

    if not all_records:
        raise SystemExit("No records built; check GCS paths and credentials.")

    features = datasets.Features(
        {
            "messages": datasets.Sequence(
                {
                    "role": datasets.Value("string"),
                    "content": datasets.Value("string"),
                }
            ),
            "audio": datasets.Sequence(datasets.Audio()),
            "audio_paths": datasets.Sequence(datasets.Value("string")),
            "tier": datasets.Value("string"),
            "subset": datasets.Value("string"),
            "task": datasets.Value("string"),
            "dataset_name": datasets.Value("string"),
            "source_dataset": datasets.Value("string"),
            "license": datasets.Value("string"),
            "template_path": datasets.Value("string"),
            "id": datasets.Value("string"),
        }
    )
    ds = datasets.Dataset.from_list(all_records, features=features)
    ds = ds.cast_column("audio", datasets.Sequence(datasets.Audio()))

    ds.push_to_hub(
        args.repo_id,
        config_name="default",
        split="test",
        set_default=True,
        private=bool(args.private),
        token=token,
        commit_message=f"Publish multiaudio rows (tier={tier})",
        embed_external_files=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
