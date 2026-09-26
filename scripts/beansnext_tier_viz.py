"""Summarise and plot the BEANS-Next Hub dataset per tier.

Ported from Marius Miron's `beans_pro/scripts/beansnext_tier_viz.py` in the
predecessor `beans-pro` repository. The analysis is unchanged; the data access
is rewired onto this package (`beans_next.datasets.beans_next_hub`) and the
expected-subset lists come from `TIER_*_SUBSETS` instead of a `beans_pro`
registry YAML.

Metadata-only: it streams the Hub metadata Parquet and never downloads audio.

Outputs, under `--output-dir`:

- `tier_summary.csv`, `subset_summary.csv` — row counts and field coverage;
- `tier_inventory.csv` — expected vs observed subsets per tier, which is what
  catches a subset that silently failed to publish;
- `plots/*.png` — row counts, field coverage, duration histograms, and the
  tier-4 clips-per-example histogram.

Pass `--revision` deliberately. `main` currently holds an older split
generation than the evaluation suites use, so figures built from it describe a
superseded dataset.

Example
-------
```bash
uv run python scripts/beansnext_tier_viz.py \
  --repo EarthSpeciesProject/BEANS-Next \
  --revision v2026.09-refresh \
  --output-dir docs/results/beansnext_tier_viz
```
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

from beans_next.datasets.beans_next_hub import (
    BEANS_NEXT_HUB_REPO_ID,
    TIER_1_SUBSETS,
    TIER_2_SUBSETS,
    TIER_3_SUBSETS,
    TIER_4_SUBSETS,
    iter_parquet_row_dicts,
)

EXPECTED_BY_TIER: Final[dict[int, frozenset[str]]] = {
    1: TIER_1_SUBSETS,
    2: TIER_2_SUBSETS,
    3: TIER_3_SUBSETS,
    4: TIER_4_SUBSETS,
}

#: Fields whose presence is worth tracking per subset.
TRACKED_FIELDS: Final[tuple[str, ...]] = (
    "id", "task", "tier", "messages", "audio_paths", "license",
    "metadata", "source_datasets", "output",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=BEANS_NEXT_HUB_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/results/beansnext_tier_viz")
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--tiers", default="1,2,3,4")
    parser.add_argument(
        "--no-plots", action="store_true", help="write CSVs only",
    )
    return parser.parse_args()


def metadata_url(repo: str, revision: str) -> str:
    """Build the Hub URL of the metadata Parquet.

    Returns
    -------
    str
        An `hf://` URL `iter_parquet_row_dicts` can stream.
    """
    return f"hf://datasets/{repo}@{revision}/test/metadata.parquet"


def infer_n_audios(row: Mapping[str, Any]) -> int | None:
    """Count the audio clips an example carries.

    Returns
    -------
    int or None
        Clip count, or None when the row says nothing about audio.
    """
    paths = row.get("audio_paths")
    if paths is not None and hasattr(paths, "__len__") and not isinstance(paths, str):
        return len(paths)
    messages = row.get("messages")
    iterable = messages is not None and hasattr(messages, "__iter__")
    seq = list(messages) if iterable else []
    text = " ".join(
        m.get("content", "") for m in seq
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    )
    return text.count("<AudioHere>") or None


def duration_of(row: Mapping[str, Any]) -> float | None:
    """Best-effort audio duration in seconds.

    Returns
    -------
    float or None
        Duration, or None when the row does not record one.
    """
    for key in ("duration_sec", "audio_duration", "duration_s"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = None
    if isinstance(meta, Mapping):
        for key in ("duration_sec", "duration_s", "duration"):
            value = meta.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return None


def stream_rows(url: str, tiers: set[int], max_rows: int | None) -> Iterator[dict]:
    """Yield metadata rows for the requested tiers.

    Yields
    ------
    dict
        One metadata row.
    """
    seen = 0
    for row in iter_parquet_row_dicts(url):
        tier = row.get("tier")
        try:
            tier = int(tier) if tier is not None else None
        except (TypeError, ValueError):
            tier = None
        if tier not in tiers:
            continue
        yield row
        seen += 1
        if max_rows is not None and seen >= max_rows:
            return


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """Write a small CSV without requiring pandas."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_bar(
    counts: Mapping[str, int], title: str, out_path: Path, top: int = 25
) -> None:
    """Write a bar chart of the largest counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    if not items:
        return
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.bar([k for k, _ in items], [v for _, v in items])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hist(
    values: list[float], title: str, out_path: Path, logx: bool = False
) -> None:
    """Write a histogram of the given values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [v for v in values if v and v > 0]
    if not values:
        return
    fig = plt.figure(figsize=(8.5, 4.5))
    ax = fig.add_subplot(111)
    ax.hist(values, bins=60)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Stream the metadata, write summaries, and render the plots."""
    args = parse_args()
    tiers = {int(t) for t in args.tiers.split(",") if t.strip()}
    url = metadata_url(args.repo, args.revision)
    print(f"streaming {url}")

    by_tier: Counter = Counter()
    by_subset: Counter = Counter()
    subsets_seen: dict[int, set[str]] = defaultdict(set)
    coverage: Counter = Counter()
    durations: list[float] = []
    t4_audios: list[float] = []
    total = 0

    for row in stream_rows(url, tiers, args.max_rows):
        total += 1
        tier = int(row["tier"])
        task = str(row.get("task") or row.get("dataset_name") or "?")
        by_tier[tier] += 1
        by_subset[task] += 1
        subsets_seen[tier].add(task)
        for field in TRACKED_FIELDS:
            if row.get(field) is not None:
                coverage[field] += 1
        seconds = duration_of(row)
        if seconds:
            durations.append(seconds)
        if tier == 4:
            n = infer_n_audios(row)
            if n:
                t4_audios.append(float(n))

    out = args.output_dir
    write_csv(out / "tier_summary.csv", ["tier", "rows", "subsets"],
              [[t, by_tier[t], len(subsets_seen[t])] for t in sorted(by_tier)])
    write_csv(out / "subset_summary.csv", ["subset", "rows"],
              [[s, n] for s, n in sorted(by_subset.items(), key=lambda kv: -kv[1])])

    inventory: list[list[Any]] = []
    for tier in sorted(tiers):
        expected = EXPECTED_BY_TIER.get(tier, frozenset())
        observed = subsets_seen.get(tier, set())
        for subset in sorted(expected | observed):
            inventory.append([
                tier, subset,
                "yes" if subset in expected else "no",
                "yes" if subset in observed else "no",
                by_subset.get(subset, 0),
            ])
    write_csv(out / "tier_inventory.csv",
              ["tier", "subset", "expected", "observed", "rows"], inventory)

    print(f"rows: {total}")
    for tier in sorted(by_tier):
        expected = EXPECTED_BY_TIER.get(tier, frozenset())
        missing = sorted(expected - subsets_seen.get(tier, set()))
        print(f"  tier {tier}: {by_tier[tier]} rows, {len(subsets_seen[tier])} subsets"
              + (f"  MISSING: {', '.join(missing)}" if missing else ""))

    if args.no_plots:
        return
    plots = out / "plots"
    plot_bar(by_subset, f"rows by subset ({args.revision})",
             plots / "rows_by_subset_top25.png")
    plot_bar(coverage, f"field coverage ({args.revision})",
             plots / "field_coverage.png")
    plot_hist(durations, "audio duration (s)", plots / "duration_seconds_hist.png")
    plot_hist(durations, "audio duration (s, log)",
              plots / "duration_seconds_hist_logx.png", logx=True)
    plot_hist(t4_audios, "tier-4 clips per example", plots / "tier4_n_audios_hist.png")
    print(f"wrote CSVs and plots under {out}")


if __name__ == "__main__":
    main()
