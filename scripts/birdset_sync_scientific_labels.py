"""Sync BirdSet scientific label vocabularies into eval-task YAMLs.

This script materializes a per-subset scientific-name vocabulary from the
authoritative BirdSet metadata returned by ``esp_data.BirdSet`` and writes it
into the corresponding registry YAML under ``scientific_labels``.

Why:
- Many BirdSet subsets do not reliably populate ``species_common``.
- Open-set models output free-form text (timestamps, extra prose, casing),
  so later label matching needs a robust target vocabulary.
- The eval-task YAMLs are a convenient place to persist vocabularies used by
  post-processing and scoring pipelines.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from beans_next.datasets.esp_data import (
    _load_birdset_rows_via_reflection,
    require_esp_data,
)

_SCIENTIFIC_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z\\-\\.]+$")


class _YamlDumper(yaml.SafeDumper):
    """PyYAML dumper with indented sequences for readability."""

    # https://stackoverflow.com/a/39681672
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:  # noqa: ANN401
        return super().increase_indent(flow, False)


def _iter_text_candidates(value: object) -> Iterable[str]:
    """Yield non-empty strings from a scalar/list/JSON-list value."""
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    yield s
        return
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return
        if s.startswith("[") and s.endswith("]"):
            try:
                decoded = json.loads(s)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                yield from _iter_text_candidates(decoded)
                return
        yield s
        return
    yield str(value).strip()


def _looks_like_scientific_name(text: str) -> bool:
    """Heuristic filter for scientific-name strings.

    We accept 2+ tokens (genus + species [+ subspecies]) where each token looks
    like a scientific token (letters/hyphens/dots). This keeps vocabularies
    clean when upstream fields contain ids or URLs.
    """
    parts = [p for p in text.strip().split() if p]
    if len(parts) < 2:
        return False
    return all(_SCIENTIFIC_TOKEN_RE.match(p) is not None for p in parts)


def extract_scientific_labels_from_rows(rows: Iterable[Mapping[str, object]]) -> set[str]:
    """Extract scientific-name candidates from raw BirdSet rows."""
    keys = (
        # Most reliable across subsets (often scientific names live here).
        "species",
        "canonical_name_multispecies",
        "scientific_name_unified_original",
        "canonical_name",
        # Extra fallbacks seen in some variants.
        "scientific_name",
        "species_scientific",
    )
    out: set[str] = set()
    for row in rows:
        for key in keys:
            for cand in _iter_text_candidates(row.get(key)):
                if _looks_like_scientific_name(cand):
                    out.add(cand)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or len(obj) != 1:
        raise ValueError(f"Unexpected YAML shape in {path} (expected 1 top key).")
    return obj


def _dump_yaml(obj: dict[str, Any]) -> str:
    # Keep registry YAMLs human-readable and stable for diffs.
    return yaml.dump(
        obj,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        Dumper=_YamlDumper,
        indent=2,
        width=88,
    )


def sync_one_eval_task(*, path: Path, limit: int | None, dry_run: bool) -> tuple[str, int]:
    """Sync one BirdSet eval-task YAML, returning (task_id, n_labels)."""
    doc = _load_yaml(path)
    task_id = next(iter(doc.keys()))
    cfg = doc[task_id]
    if not isinstance(cfg, dict):
        raise ValueError(f"Unexpected task config type in {path}: {type(cfg)}")
    subset = cfg.get("subset")
    split = cfg.get("split")
    if not isinstance(subset, str) or not subset.strip():
        raise ValueError(f"Missing subset in {path}")
    if not isinstance(split, str) or not split.strip():
        raise ValueError(f"Missing split in {path}")

    esp = require_esp_data()
    rows = _load_birdset_rows_via_reflection(esp, split=split)
    if limit is not None:
        rows = (r for i, r in enumerate(rows) if i < limit)
    labels = extract_scientific_labels_from_rows(rows)
    cfg["scientific_labels"] = sorted(labels)

    if not dry_run:
        path.write_text(_dump_yaml(doc), encoding="utf-8")
    return task_id, len(labels)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("beans_next/registry/eval_task"),
        help="Eval-task registry directory (default: beans_next/registry/eval_task).",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="birdset_*_test_5s.yaml",
        help="Glob pattern for BirdSet eval-task YAMLs.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of rows scanned per subset (debug only).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute label sets but do not write any files.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    reg_dir: Path = args.registry_dir
    paths = sorted(reg_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {reg_dir / args.pattern}")

    updated: list[tuple[str, int, str]] = []
    for path in paths:
        task_id, n = sync_one_eval_task(path=path, limit=args.limit, dry_run=args.dry_run)
        updated.append((task_id, n, str(path)))

    for task_id, n, path in updated:
        print(f"{task_id}\\t{n}\\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
