"""Measure actual audio durations for BEANS-Next tiers 1-3 against Gemma 4's 30 s limit.

The tiers metadata parquet's ``duration_sec``/``audio_duration``/``crop_start``/
``crop_end`` fields are only populated for five multi-clip soundscape tasks;
every other task (the large majority of rows) carries no duration metadata at
all. This script instead reads each referenced WAV file's header (via
``soundfile.info``, which does not decode audio) to get an authoritative
duration, in parallel, and reports per-task counts of rows at/under vs. over
30.0 s.

This is the plan's Decision-2 precondition for BEANS-Next tiers 1-3: the
Audex launcher README's claim that every tiers clip is <= 30.0 s was written
against a different (30 s windowing) scheme, so it must be re-measured before
committing GPU hours to the full tiers run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf


def _duration_seconds(path: Path) -> float | None:
    """Return a WAV file's duration in seconds, or ``None`` if unreadable.

    Returns
    -------
    float or None
        Duration in seconds, or ``None`` if the file could not be read.
    """
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    return info.frames / info.samplerate if info.samplerate else None


def main() -> int:
    """Measure tiers 1-3 clip durations and report per-task over-30s counts.

    Returns
    -------
    int
        Zero always; this is a reporting tool, not a gate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-parquet",
        default="/gpfs/scratch/acw777/esp/beans-next/test/metadata.parquet",
    )
    parser.add_argument(
        "--audio-root",
        default="/gpfs/scratch/acw777/esp/beans-next/test",
    )
    parser.add_argument("--cap-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--output-json",
        default="/gpfs/scratch/acw777/beans-next-runs/gemma4_tiers_duration_report.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(
        args.metadata_parquet, columns=["tier", "task", "file_name"]
    )
    sub = df[df["tier"].isin([1, 2, 3])].reset_index(drop=True)
    audio_root = Path(args.audio_root)

    paths = [audio_root / rel for rel in sub["file_name"]]
    durations: list[float | None] = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_duration_seconds, path): idx
            for idx, path in enumerate(paths)
        }
        for fut in as_completed(futures):
            durations[futures[fut]] = fut.result()

    sub = sub.assign(duration_sec_measured=durations)
    unreadable = sub["duration_sec_measured"].isna().sum()

    per_task: dict[str, dict[str, float | int]] = defaultdict(dict)
    over_total = 0
    for task, group in sub.groupby("task"):
        measured = group["duration_sec_measured"].dropna()
        over = int((measured > args.cap_seconds).sum())
        over_total += over
        per_task[task] = {
            "n_rows": int(len(group)),
            "n_measured": int(len(measured)),
            "n_over_cap": over,
            "max_duration_sec": float(measured.max()) if len(measured) else None,
        }

    report = {
        "cap_seconds": args.cap_seconds,
        "total_rows": int(len(sub)),
        "unreadable_files": int(unreadable),
        "total_rows_over_cap": over_total,
        "per_task": per_task,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
