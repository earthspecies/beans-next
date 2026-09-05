"""Drop error rows from a per-task output dir so a resume retries them.

Used once, to recover from the ``--allowed-local-media-path`` bug fixed in
``run_gemma4_eval.sbatch`` (trimmed-clip cache outside the allowed root on
the ``huggingface`` backend): every row whose audio needed trimming got a
non-null ``error`` and was still marked complete in ``checkpoint.json``, so a
plain ``--resume-from`` would never retry them. This rewrites
``predictions.jsonl``, ``processed_predictions.jsonl``, and
``scored_predictions.jsonl`` to drop rows with a non-null ``error``, and
rewrites ``checkpoint.json`` to only list the surviving sample ids, so the
next ``--resume-from`` run reprocesses exactly the dropped rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _prune_jsonl(path: Path) -> tuple[set[str], int]:
    """Drop error rows from a JSONL file in place.

    Returns
    -------
    tuple[set[str], int]
        Surviving ``sample_id`` set, and the number of rows dropped.
    """
    if not path.is_file():
        return set(), 0
    kept_ids: set[str] = set()
    kept_lines: list[str] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error") is not None:
            dropped += 1
            continue
        kept_ids.add(str(row["sample_id"]))
        kept_lines.append(line)
    path.write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8"
    )
    return kept_ids, dropped


def main() -> int:
    """Prune error rows from every affected task dir under a suite root.

    Returns
    -------
    int
        Zero always.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("task_ids", nargs="+")
    args = parser.parse_args()

    for task_id in args.task_ids:
        task_dir = args.suite_dir / task_id
        pred_ids, pred_dropped = _prune_jsonl(task_dir / "predictions.jsonl")
        proc_ids, proc_dropped = _prune_jsonl(task_dir / "processed_predictions.jsonl")
        scored_ids, scored_dropped = _prune_jsonl(task_dir / "scored_predictions.jsonl")
        surviving = pred_ids & proc_ids & scored_ids
        checkpoint_path = task_dir / "checkpoint.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "schema_version": "beans_next.checkpoint.v1",
                    "completed_sample_ids": sorted(surviving),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"{task_id}: dropped pred={pred_dropped} proc={proc_dropped} "
            f"scored={scored_dropped}, surviving={len(surviving)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
