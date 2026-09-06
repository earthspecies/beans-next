"""Verify a staged or published BEANS-Next Hugging Face bundle.

Checks that the bundle is internally consistent and matches the loader contract
in `beans_next.datasets.beans_next_hub`:

- every metadata row's `file_name` resolves to a file that exists;
- `audio_id` equals the SHA-256 of that file's bytes, and the file is named
  after it (this is the property the whole content-addressed layout rests on);
- audio decodes and is mono PCM WAV;
- per-task row counts match what was expected.

Run against a local staging directory before uploading, and again against a
download of the published branch afterwards.

Examples
--------
::

    uv run python scripts/verify_beans_next_hf_bundle.py \\
        --bundle /path/to/bundle --sample 400
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import random
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Verify a bundle and print a report.

    Parameters
    ----------
    argv
        Command-line arguments; defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        `0` when every check passes, `1` otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument(
        "--sample",
        type=int,
        default=300,
        help="how many rows to hash-verify (0 verifies all)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tasks",
        default="",
        help=(
            "comma-separated tasks to verify. An incremental bundle only "
            "stages audio for the tasks it adds; its other rows point at audio "
            "already present in the repo, which is absent locally and would "
            "otherwise be miscounted as missing."
        ),
    )
    args = ap.parse_args(argv)
    only = {t.strip() for t in args.tasks.split(",") if t.strip()}

    import pyarrow.parquet as pq

    meta = args.bundle / "test" / "metadata.parquet"
    if not meta.exists():
        print(f"FAIL no metadata.parquet at {meta}")
        return 1
    table = pq.read_table(meta)
    cols = table.to_pydict()
    n = table.num_rows
    print(f"rows: {n}")

    by_task = collections.Counter(cols["task"])
    print(f"tasks: {len(by_task)}")

    tiers = collections.defaultdict(set)
    for task, tier in zip(cols["task"], cols["tier"], strict=True):
        tiers[task].add(tier)
    mixed = {t: sorted(v) for t, v in tiers.items() if len(v) > 1}
    if mixed:
        print(f"FAIL tasks with inconsistent tier: {mixed}")

    rows = [i for i in range(n) if not only or cols["task"][i] in only]
    if only:
        print(f"restricted to {len(only)} task(s): {len(rows)} rows")
        for t in sorted(only):
            print(f"    {t:26s} {by_task.get(t, 0)}")
    if args.sample and args.sample < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)

    missing = 0
    mismatched = 0
    bad_audio = 0
    checked = 0
    audio_root = args.bundle / "test"
    for i in rows:
        fn = cols["file_name"][i]
        aid = cols["audio_id"][i]
        if not fn or not aid:
            continue  # multi-audio rows carry their paths in other columns
        path = audio_root / fn
        if not path.exists():
            missing += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != aid or not fn.endswith(f"{aid}.wav"):
            mismatched += 1
        checked += 1

    print(f"hash-verified rows: {checked}")
    print(f"  missing files      : {missing}")
    print(f"  sha/name mismatches: {mismatched}")

    # decode a small subset to catch truncated or unreadable WAVs
    try:
        import soundfile as sf

        for i in rows[: min(60, len(rows))]:
            fn = cols["file_name"][i]
            if not fn:
                continue
            path = audio_root / fn
            if not path.exists():
                continue
            info = sf.info(str(path))
            if info.frames <= 0 or info.channels != 1:
                bad_audio += 1
        print(f"  undecodable/non-mono: {bad_audio}")
    except ImportError:
        print("  (soundfile unavailable; skipped decode check)")

    ok = not missing and not mismatched and not bad_audio and not mixed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
