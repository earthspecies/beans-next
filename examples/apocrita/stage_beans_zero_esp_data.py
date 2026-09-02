"""Mirror BEANS-Zero's esp_data (16 kHz) subtree to local scratch.

Downloads the per-subset ``*_test.jsonl`` metadata files and every WAV each
row references via ``audio_path_16KHz`` from the public ``esp-data-274503``
bucket, preserving the bucket's relative layout under a local output root so
that root can be pointed to directly by ``ALP_DATA_HOME`` (see
``alp_data/io/paths.py``). This removes per-job network cost across the
chained Audex evaluation.

Not gated; public bucket, plain HTTPS GET, no auth required.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/esp-data-274503/beans-zero/v0.1.0/raw"

SUBSETS = [
    "esc50",
    "cbi",
    "watkins",
    "humbugdb",
    "lifestage",
    "call-type",
    "captioning",
    "zf-indiv",
    "dcase",
    "enabirds",
    "gibbons",
    "hiceas",
    "rfcx",
    "unseen-species-cmn",
    "unseen-species-sci",
    "unseen-species-tax",
    "unseen-genus-cmn",
    "unseen-genus-sci",
    "unseen-genus-tax",
    "unseen-family-cmn",
    "unseen-family-sci",
    "unseen-family-tax",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def _fetch(url: str, dest: Path, *, retries: int) -> int:
    """Download ``url`` to ``dest`` atomically, skipping if already present.

    Returns
    -------
    int
        Number of bytes written (0 if the file already existed and was skipped).
        Re-raises whatever the last download attempt raised if all retries fail.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp-{time.time_ns()}")
    last_exc: Exception | None = None
    for attempt in range(1 + max(0, retries)):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
                data = resp.read()
            tmp.write_bytes(data)
            tmp.replace(dest)
            return len(data)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2**attempt, 10))
    tmp.unlink(missing_ok=True)
    assert last_exc is not None
    raise last_exc


def _download_jsonl_files(raw_root: Path, *, retries: int) -> dict[str, list[dict]]:
    rows_by_subset: dict[str, list[dict]] = {}
    for subset in SUBSETS:
        name = f"{subset}_test.jsonl"
        dest = raw_root / name
        _fetch(f"{BASE_URL}/{name}", dest, retries=retries)
        lines = dest.read_text().splitlines()
        rows_by_subset[subset] = [json.loads(line) for line in lines if line.strip()]
    return rows_by_subset


def main() -> int:
    """Mirror BEANS-Zero JSONL metadata plus every referenced 16 kHz WAV.

    Returns
    -------
    int
        Zero on success.
    """
    args = _parser().parse_args()
    raw_root = args.output_root / "beans-zero" / "v0.1.0" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    rows_by_subset = _download_jsonl_files(raw_root, retries=args.retries)

    audio_rel_paths: set[str] = set()
    per_subset_rows: dict[str, int] = {}
    for subset, rows in rows_by_subset.items():
        per_subset_rows[subset] = len(rows)
        for row in rows:
            rel = row.get("audio_path_16KHz")
            if isinstance(rel, str) and rel:
                audio_rel_paths.add(rel)

    downloaded_bytes = 0
    failures: list[str] = []
    skipped = 0
    downloaded = 0

    def _download_one(rel: str) -> tuple[str, int]:
        dest = raw_root / rel
        n = _fetch(f"{BASE_URL}/{rel}", dest, retries=args.retries)
        return rel, n

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_download_one, rel): rel for rel in sorted(audio_rel_paths)
        }
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                _, n = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{rel}: {exc}")
                continue
            if n == 0:
                skipped += 1
            else:
                downloaded += 1
                downloaded_bytes += n

    per_subset_bytes: Counter[str] = Counter()
    for subset, rows in rows_by_subset.items():
        for row in rows:
            rel = row.get("audio_path_16KHz")
            if isinstance(rel, str) and rel:
                p = raw_root / rel
                if p.exists():
                    per_subset_bytes[subset] += p.stat().st_size

    payload = {
        "schema_version": "beans_next.beans_zero_esp_data_stage.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "output_root": str(args.output_root.resolve()),
        "base_url": BASE_URL,
        "n_subsets": len(SUBSETS),
        "n_audio_files_referenced": len(audio_rel_paths),
        "n_audio_files_downloaded_this_run": downloaded,
        "n_audio_files_already_present": skipped,
        "n_audio_files_failed": len(failures),
        "bytes_downloaded_this_run": downloaded_bytes,
        "per_subset_row_counts": per_subset_rows,
        "per_subset_audio_bytes": dict(per_subset_bytes),
        "failures": failures,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.manifest)

    print(f"subsets={len(SUBSETS)} audio_referenced={len(audio_rel_paths)} "
          f"downloaded={downloaded} already_present={skipped} failed={len(failures)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
