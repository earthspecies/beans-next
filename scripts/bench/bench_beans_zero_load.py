"""Benchmark dataset loading speed and full audio materialization for esp_data.

Measures two phases depending on flags:

- **row iteration** (default): time to yield `DatasetExample` objects from
  `iter_esp_data_beans_zero_examples`, which includes the GCS → local WAV download.
- **full audio** (``--full-audio``): additionally reads each WAV from disk and
  base64-encodes it, mirroring what the beans-next runner does before an HTTP POST.

Concurrency (``--workers N``) runs both phases in parallel threads to simulate
throughput under real inference load.

Example
-------
# esp_data row-iteration only (sequential):
uv run --group esp python scripts/bench/bench_beans_zero_load.py \\
  --subset esc50 --n 100 --only-esp-data

# Full materialization + base64, 4 parallel workers:
uv run --group esp python scripts/bench/bench_beans_zero_load.py \\
  --subset esc50 --n 100 --only-esp-data --full-audio --workers 4
"""

from __future__ import annotations

import argparse
import base64
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from beans_next.api.types import DatasetExample
from beans_next.datasets import dataset_name_equals
from beans_next.datasets.hf import iter_hf_dataset_examples
from beans_next.datasets.hf_streaming import iter_hf_streaming_examples


def _take(it: Iterable[DatasetExample], n: int) -> list[DatasetExample]:
    out: list[DatasetExample] = []
    for ex in it:
        out.append(ex)
        if len(out) >= n:
            break
    return out


def _materialize_example(ex: DatasetExample) -> tuple[bool, int]:
    """Read WAV bytes and base64-encode, mirroring the runner's HTTP POST path.

    Returns
    -------
    tuple[bool, int]
        ``(success, byte_count)`` — success is True when audio was read without
        error, byte_count is the number of raw WAV bytes (0 on failure).
    """
    audio_path = ex.metadata.get("audio_path") if ex.metadata else None
    if not isinstance(audio_path, str) or not audio_path:
        return False, 0
    p = Path(audio_path)
    if not p.is_file():
        return False, 0
    try:
        wav_bytes = p.read_bytes()
        base64.b64encode(wav_bytes)
        return True, len(wav_bytes)
    except Exception:
        return False, 0


@dataclass(frozen=True, slots=True)
class BenchResult:
    """A single timing result for a dataset-loading benchmark.

    Parameters
    ----------
    label
        Short identifier for the loader variant (e.g. ``"esp_data"``).
    n
        Number of examples iterated.
    seconds
        Wall-clock time spent.
    audio_ok
        Number of examples with successfully materialized audio (``--full-audio``
        only; ``-1`` when audio materialization was not measured).
    audio_bytes
        Total raw WAV bytes read across all successful examples.
    """

    label: str
    n: int
    seconds: float
    audio_ok: int = -1
    audio_bytes: int = 0

    @property
    def rows_per_sec(self) -> float:
        """Rows processed per second."""
        if self.seconds <= 0:
            return float("inf")
        return self.n / self.seconds

    @property
    def mb_per_sec(self) -> float:
        """Megabytes of audio per second (0 when audio was not measured)."""
        if self.seconds <= 0 or self.audio_bytes <= 0:
            return 0.0
        return self.audio_bytes / self.seconds / 1024 / 1024


def _run_esp_sequential(
    subset: str,
    split: str,
    n: int,
    *,
    full_audio: bool,
) -> BenchResult:
    from beans_next.datasets.esp_data import iter_esp_data_beans_zero_examples

    examples = _take(iter_esp_data_beans_zero_examples(subset=subset, split=split), n)

    if not full_audio:
        return BenchResult(label="esp_data", n=len(examples), seconds=0.0)

    audio_ok = 0
    audio_bytes = 0
    for ex in examples:
        ok, nbytes = _materialize_example(ex)
        if ok:
            audio_ok += 1
            audio_bytes += nbytes
    return BenchResult(
        label="esp_data",
        n=len(examples),
        seconds=0.0,
        audio_ok=audio_ok,
        audio_bytes=audio_bytes,
    )


def _time_load_esp(
    subset: str,
    split: str,
    n: int,
    *,
    full_audio: bool,
    workers: int,
) -> BenchResult:
    from beans_next.datasets.esp_data import (
        _WORKERS_ENV,
        _env_int,
        iter_esp_data_beans_zero_examples,
    )

    # Resolve effective workers the same way iter_esp_data_beans_zero_examples does,
    # so the label and header reflect the actual concurrency level used.
    effective_workers = workers
    if effective_workers == 1:
        effective_workers = max(1, _env_int(_WORKERS_ENV, default=1))

    t0 = time.perf_counter()

    # effective_workers > 1: iterator collects metadata rows first, then issues
    # concurrent GCS downloads in a ThreadPoolExecutor — actual network I/O overlaps.
    examples = _take(
        iter_esp_data_beans_zero_examples(
            subset=subset, split=split, workers=effective_workers
        ),
        n,
    )

    audio_ok = -1
    audio_bytes = 0
    if full_audio:
        audio_ok = 0
        for ex in examples:
            ok, nb = _materialize_example(ex)
            if ok:
                audio_ok += 1
                audio_bytes += nb

    elapsed = time.perf_counter() - t0
    return BenchResult(
        label=f"esp_data_w{effective_workers}" if effective_workers > 1 else "esp_data",
        n=len(examples),
        seconds=elapsed,
        audio_ok=audio_ok,
        audio_bytes=audio_bytes,
    )


def _time_load(label: str, fn: Callable[[], list[DatasetExample]]) -> BenchResult:
    t0 = time.perf_counter()
    rows = fn()
    elapsed = time.perf_counter() - t0
    return BenchResult(label=label, n=len(rows), seconds=elapsed)


def _print_result(r: BenchResult, *, full_audio: bool) -> None:
    parts = [
        r.label,
        f"n={r.n}",
        f"sec={r.seconds:.3f}",
        f"rows/s={r.rows_per_sec:.1f}",
    ]
    if full_audio:
        audio_ok_str = str(r.audio_ok) if r.audio_ok >= 0 else "n/a"
        parts.append(f"audio_ok={audio_ok_str}/{r.n}")
        if r.audio_bytes > 0:
            parts.append(f"MB/s={r.mb_per_sec:.2f}")
    print("\t".join(parts))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark BEANS-Zero esp_data loading: row iteration and full "
            "audio materialization (GCS download → WAV → base64)."
        )
    )
    p.add_argument("--subset", default="esc50", help="BEANS-Zero subset.")
    p.add_argument("--split", default="test", help="Split name (default: test).")
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of examples to load (default: 100).",
    )
    p.add_argument(
        "--hf-path",
        default="EarthSpeciesProject/BEANS-Zero",
        help="HuggingFace dataset id.",
    )
    p.add_argument("--hf-config", default=None, help="Optional HF config name.")
    p.add_argument(
        "--hf-streaming",
        action="store_true",
        default=False,
        help="Use HF streaming loader.",
    )
    p.add_argument(
        "--also-esp-data",
        action="store_true",
        default=False,
        help="Also benchmark esp_data loader (requires esp_data installed).",
    )
    p.add_argument(
        "--only-esp-data",
        action="store_true",
        default=False,
        help="Only run esp_data benchmark (skip HuggingFace).",
    )
    p.add_argument(
        "--only-hf",
        action="store_true",
        default=False,
        help="Only run HuggingFace benchmark (skip esp_data).",
    )
    p.add_argument(
        "--full-audio",
        action="store_true",
        default=False,
        help=(
            "After row iteration, read each WAV from disk and base64-encode it, "
            "mirroring the full runner inference-time materialization path."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker threads for esp_data loading (default: 1). "
            "Values >1 simulate concurrent inference-time download load."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    n = max(1, int(args.n))
    subset = str(args.subset).strip()
    split = str(args.split).strip()
    hf_config = None
    if args.hf_config is not None and str(args.hf_config).strip():
        hf_config = str(args.hf_config).strip()
    full_audio: bool = args.full_audio
    workers: int = max(1, int(args.workers))

    row_filter = dataset_name_equals(subset)

    if args.only_esp_data and args.only_hf:
        raise SystemExit("Choose at most one of --only-esp-data / --only-hf.")

    results: list[BenchResult] = []

    run_hf = not args.only_esp_data
    run_esp = args.also_esp_data or args.only_esp_data
    if args.only_hf:
        run_esp = False

    if run_hf:
        if args.hf_streaming:
            _take(
                iter_hf_streaming_examples(
                    str(args.hf_path),
                    split=split,
                    config_name=hf_config,
                    row_filter=row_filter,
                ),
                1,
            )
        else:
            _take(
                iter_hf_dataset_examples(
                    str(args.hf_path),
                    split=split,
                    config_name=hf_config,
                    row_filter=row_filter,
                ),
                1,
            )

        if args.hf_streaming:
            results.append(
                _time_load(
                    "hf_streaming",
                    lambda: _take(
                        iter_hf_streaming_examples(
                            str(args.hf_path),
                            split=split,
                            config_name=hf_config,
                            row_filter=row_filter,
                        ),
                        n,
                    ),
                )
            )
        else:
            results.append(
                _time_load(
                    "hf_map",
                    lambda: _take(
                        iter_hf_dataset_examples(
                            str(args.hf_path),
                            split=split,
                            config_name=hf_config,
                            row_filter=row_filter,
                        ),
                        n,
                    ),
                )
            )

    if run_esp:
        try:
            from beans_next.datasets.esp_data import iter_esp_data_beans_zero_examples
        except ImportError as exc:
            print(f"esp_data\tSKIP\t{exc}")
            return 0

        # Warmup one row (includes GCS auth + first download).
        _take(iter_esp_data_beans_zero_examples(subset=subset, split=split), 1)

        # Resolve effective workers for the header (env var may override workers=1).
        from beans_next.datasets.esp_data import _WORKERS_ENV, _env_int

        effective_workers_header = workers
        if effective_workers_header == 1:
            effective_workers_header = max(1, _env_int(_WORKERS_ENV, default=1))
        print(
            f"=== esp_data n={n} full_audio={full_audio} "
            f"workers={effective_workers_header} ==="
        )
        r = _time_load_esp(subset, split, n, full_audio=full_audio, workers=workers)
        results.append(r)

    for r in results:
        _print_result(r, full_audio=full_audio)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
