"""Sync BEANS-Zero unseen-* label vocabularies into eval-task YAMLs.

For the BEANS-Zero core taxonomy subsets (unseen-{family,genus,species}-{cmn,sci,tax}),
models often emit verbose answers (casing, punctuation, extra text). Downstream
post-processing needs a reliable vocabulary of target labels to match against.

This script derives that vocabulary from the authoritative dataset rows loaded via
``esp_data`` (not from hard-coded lists) and writes it into the corresponding
``beans_next/registry/eval_task/beans_zero_unseen_*.yaml`` files as an inline
``labels:`` list.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from beans_next.datasets.esp_data import (
    _load_beans_zero_rows_via_reflection,
    require_esp_data,
)


class _YamlDumper(yaml.SafeDumper):
    """PyYAML dumper with indented sequences for readability."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:  # noqa: ANN401
        return super().increase_indent(flow, False)


def _dump_yaml(obj: dict[str, Any]) -> str:
    return yaml.dump(
        obj,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        Dumper=_YamlDumper,
        indent=2,
        width=88,
    )


def _iter_label_atoms(value: object) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return
        # Some targets are comma-separated multi-reference strings.
        for part in s.split(","):
            tok = part.strip()
            if tok:
                yield tok
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                yield item.strip()
        return
    yield str(value).strip()


def build_vocab(*, subset: str, split: str, limit: int | None) -> list[str]:
    vocab: set[str] = set()
    esp = require_esp_data()
    rows = _load_beans_zero_rows_via_reflection(esp, subset=subset, split=split)
    if limit is not None:
        rows = (r for i, r in enumerate(rows) if i < limit)
    for row in rows:
        # BEANS-Zero ground truth is stored under `output` (or `labels` in some variants).
        val = row.get("output") if "output" in row else row.get("labels")
        for tok in _iter_label_atoms(val):
            vocab.add(tok)
    return sorted(vocab)


def sync_one(*, path: Path, limit: int | None, dry_run: bool) -> tuple[str, int]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or len(doc) != 1:
        raise ValueError(f"Unexpected YAML shape in {path}")
    task_id = next(iter(doc.keys()))
    cfg = doc[task_id]
    if not isinstance(cfg, dict):
        raise ValueError(f"Unexpected task cfg type in {path}: {type(cfg)}")
    subset = cfg.get("subset")
    split = cfg.get("split")
    if not isinstance(subset, str) or not subset.strip():
        raise ValueError(f"Missing subset in {path}")
    if not isinstance(split, str) or not split.strip():
        raise ValueError(f"Missing split in {path}")

    labels = build_vocab(subset=subset.strip(), split=split.strip(), limit=limit)
    cfg["labels"] = labels

    if not dry_run:
        path.write_text(_dump_yaml(doc), encoding="utf-8")
    return task_id, len(labels)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("beans_next/registry/eval_task"),
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="beans_zero_unseen_*_{cmn,sci,tax}.yaml",
        help="Note: brace expansion is shell-level; pass a simple glob when needed.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    reg_dir: Path = args.registry_dir
    # Python's glob doesn't support brace expansion; handle the 9 files explicitly.
    patterns = [
        "beans_zero_unseen_family_cmn.yaml",
        "beans_zero_unseen_family_sci.yaml",
        "beans_zero_unseen_family_tax.yaml",
        "beans_zero_unseen_genus_cmn.yaml",
        "beans_zero_unseen_genus_sci.yaml",
        "beans_zero_unseen_genus_tax.yaml",
        "beans_zero_unseen_species_cmn.yaml",
        "beans_zero_unseen_species_sci.yaml",
        "beans_zero_unseen_species_tax.yaml",
    ]
    paths = [reg_dir / p for p in patterns]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing eval-task YAML(s): {[str(p) for p in missing]}")

    for p in paths:
        task_id, n = sync_one(path=p, limit=args.limit, dry_run=args.dry_run)
        print(f"{task_id}\t{n}\t{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
