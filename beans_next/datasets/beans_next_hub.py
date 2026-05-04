"""Load BEANS-Next examples from the two-table Parquet bundle on Hugging Face Hub.

The dataset at ``EarthSpeciesProject/BEANS-Next`` ships two root-level files:

- ``beans_next_metadata.parquet``: one row per evaluation sample across all tiers
  and subsets. Columns include ``subset``, ``tier``, ``sample_id``, and either
  ``audio_id`` (tiers 1–3, single-audio) or ``audio_ids`` + ``query_audio_id``
  (tier 4, in-context multi-audio).
- ``beans_next_audio.parquet``: one row per unique audio clip, SHA-256 deduplicated.
  Columns include ``audio_id``, ``sha256``, and ``audio_bytes`` (canonical WAV).

Callers use :func:`iter_hf_beans_next_examples` with a ``subset`` name.  The
function resolves which tier the subset belongs to and routes to the appropriate
single-audio or multi-audio loader transparently.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Final

from huggingface_hub import hf_hub_url

from beans_next.api.types import DatasetExample

BEANS_NEXT_HUB_REPO_ID: Final[str] = "EarthSpeciesProject/BEANS-Next"
_METADATA_PARQUET: Final[str] = "beans_next_metadata.parquet"
_AUDIO_PARQUET: Final[str] = "beans_next_audio.parquet"

TIER_1_SUBSETS: Final[frozenset[str]] = frozenset(
    {
        "crow-description",
        "zebra-description",
        "f0-mean-seen-taxa",
        "f0-mean-heldout-taxa",
    }
)
TIER_2_SUBSETS: Final[frozenset[str]] = frozenset(
    {
        "bird-presence",
        "mammal-presence",
        "amphibian-presence",
        "alarm-call-presence",
        "flight-call-presence",
        "call-type-fixed-vocab",
    }
)
TIER_3_SUBSETS: Final[frozenset[str]] = frozenset({"insect-presence"})
TIER_4_SUBSETS: Final[frozenset[str]] = frozenset(
    {
        "gibbon-fewshot-detection-balanced",
        "giant-otter-4way",
        "dcase-fewshot-detection-balanced",
        "crow-4way",
        "zebra-4way",
        "unseen-species-4way",
    }
)

SINGLE_AUDIO_SUBSETS: Final[frozenset[str]] = (
    TIER_1_SUBSETS | TIER_2_SUBSETS | TIER_3_SUBSETS
)
ALL_SUBSETS: Final[frozenset[str]] = SINGLE_AUDIO_SUBSETS | TIER_4_SUBSETS

_SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def is_multiaudio_subset(subset: str) -> bool:
    """Return whether ``subset`` is a tier-4 (in-context multi-audio) subset.

    Parameters
    ----------
    subset
        Subset name to test.

    Returns
    -------
    bool
        ``True`` for tier-4 subsets, ``False`` for tiers 1–3.
    """
    return subset.strip() in TIER_4_SUBSETS


def _require_fsspec() -> ModuleType:
    try:
        import fsspec  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "HuggingFace Parquet loading requires `fsspec`. "
            "Install with: uv pip install fsspec"
        ) from exc
    return fsspec


def _hf_open_kwargs(url: str) -> dict[str, Any]:
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token or "huggingface.co" not in url.lower():
        return {}
    return {"headers": {"Authorization": f"Bearer {token}"}}


def _to_plain(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            out = value.item()
            if isinstance(out, (str, int, float, bool)):
                return out
        except Exception:  # noqa: BLE001
            pass
    return value


def iter_parquet_row_dicts(url: str) -> Iterator[dict[str, Any]]:
    """Yield one row dict per Parquet row, reading over HTTP(S) or a local path.

    Private Hub Parquet files require ``HF_TOKEN`` in the environment so that
    ``fsspec`` sends a ``Bearer`` header; see :func:`_hf_open_kwargs`.

    Parameters
    ----------
    url
        Parquet file URL or local path.

    Yields
    ------
    dict[str, Any]
        One decoded row per record.
    """
    import pyarrow.parquet as pq

    fsspec = _require_fsspec()
    with fsspec.open(url, "rb", **_hf_open_kwargs(url)) as fp:
        reader = pq.ParquetFile(fp)
        for batch in reader.iter_batches(batch_size=256):
            cols = batch.to_pydict()
            if not cols:
                continue
            keys = list(cols.keys())
            n = len(cols[keys[0]])
            for i in range(n):
                yield {k: _to_plain(cols[k][i]) for k in keys}


def _safe_stem(value: str) -> str:
    s = _SAFE_STEM_RE.sub("_", value.strip())[:120]
    return s or "row"


def _materialize_wav_bytes(audio_bytes: bytes, *, stem: str) -> str:
    """Write WAV bytes to a temporary file and return its path.

    Parameters
    ----------
    audio_bytes
        Raw WAV file bytes.
    stem
        Base name used for the temp file prefix.

    Returns
    -------
    str
        Absolute path to the new ``.wav`` temp file.
    """
    fd, path = tempfile.mkstemp(
        suffix=".wav",
        prefix=f"beansnext_{_safe_stem(stem)}_",
    )
    try:
        os.write(fd, audio_bytes)
    finally:
        os.close(fd)
    return path


def _audio_bytes_from_row(row: dict[str, Any]) -> bytes:
    raw = row.get("audio_bytes")
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if isinstance(raw, bytes):
        return raw
    raise TypeError(
        f"audio_bytes must be bytes or memoryview, got {type(raw).__name__}"
    )


def _parquet_url(repo_id: str, filename: str, *, revision: str = "main") -> str:
    return str(
        hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )


def _load_audio_map(
    repo_id: str,
    *,
    revision: str,
    needed_ids: set[str],
) -> dict[str, bytes]:
    """Scan ``beans_next_audio.parquet`` and return WAV bytes for requested ids.

    Stops scanning as soon as all ``needed_ids`` are found.

    Parameters
    ----------
    repo_id
        HuggingFace dataset id.
    revision
        Hub git revision.
    needed_ids
        Set of ``audio_id`` strings to retrieve.

    Returns
    -------
    dict[str, bytes]
        Mapping from ``audio_id`` to raw WAV bytes.

    Raises
    ------
    RuntimeError
        If any ``needed_ids`` are absent from the audio Parquet after a full scan.
    """
    if not needed_ids:
        return {}
    url = _parquet_url(repo_id, _AUDIO_PARQUET, revision=revision)
    out: dict[str, bytes] = {}
    for row in iter_parquet_row_dicts(url):
        aid = row.get("audio_id")
        if not isinstance(aid, str) or not aid.strip():
            continue
        key = aid.strip()
        if key not in needed_ids:
            continue
        out[key] = _audio_bytes_from_row(row)
        if len(out) == len(needed_ids):
            break
    missing = needed_ids - frozenset(out)
    if missing:
        raise RuntimeError(
            f"{_AUDIO_PARQUET} missing audio_id(s): "
            + ", ".join(sorted(missing)[:12])
            + (" …" if len(missing) > 12 else "")
        )
    return out


def _iter_single_examples(
    repo_id: str,
    *,
    subset: str,
    split: str,
    revision: str,
    task_id: str | None,
    limit: int | None,
    workers: int,
) -> Iterator[DatasetExample]:
    from beans_next.datasets.esp_data import (
        _build_dataset_example,
        _resolve_row_id,
        synthesize_esp_data_sample_id,
    )

    meta_url = _parquet_url(repo_id, _METADATA_PARQUET, revision=revision)
    meta_rows: list[dict[str, Any]] = []
    for row in iter_parquet_row_dicts(meta_url):
        if row.get("subset") != subset:
            continue
        meta_rows.append(dict(row))
        if limit is not None and len(meta_rows) >= limit:
            break

    needed: set[str] = set()
    for row in meta_rows:
        aid = row.get("audio_id")
        if isinstance(aid, str) and aid.strip():
            needed.add(aid.strip())
    audio_map = _load_audio_map(repo_id, revision=revision, needed_ids=needed)

    raw_rows: list[tuple[str, dict[str, Any], str]] = []
    for ordinal, row in enumerate(meta_rows):
        aid = row.get("audio_id")
        if not isinstance(aid, str) or not aid.strip():
            raise RuntimeError(f"metadata row missing audio_id subset={subset!r}")
        stable = _resolve_row_id(row)
        sample_id = (
            stable
            if stable is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_next", subset=subset, split=split, ordinal=ordinal
            )
        )
        wav_path = _materialize_wav_bytes(audio_map[aid.strip()], stem=sample_id)
        raw_rows.append((sample_id, row, wav_path))

    if workers > 1:

        def _build(item: tuple[str, dict[str, Any], str]) -> DatasetExample:
            sample_id, rowd, wav_path = item
            return _build_dataset_example(
                rowd,
                sample_id=sample_id,
                audio_path=wav_path,
                split=split,
                task_id=task_id,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(_build, raw_rows)
        return

    for sample_id, rowd, wav_path in raw_rows:
        yield _build_dataset_example(
            rowd,
            sample_id=sample_id,
            audio_path=wav_path,
            split=split,
            task_id=task_id,
        )


def _iter_multiaudio_examples(
    repo_id: str,
    *,
    subset: str,
    split: str,
    revision: str,
    task_id: str | None,
    limit: int | None,
    workers: int,
) -> Iterator[DatasetExample]:
    from beans_next.datasets.esp_data import (
        _build_multiaudio_dataset_example,
        _resolve_row_id,
        synthesize_esp_data_sample_id,
    )

    meta_url = _parquet_url(repo_id, _METADATA_PARQUET, revision=revision)
    meta_rows: list[dict[str, Any]] = []
    for row in iter_parquet_row_dicts(meta_url):
        if row.get("subset") != subset:
            continue
        meta_rows.append(dict(row))
        if limit is not None and len(meta_rows) >= limit:
            break

    needed: set[str] = set()
    for row in meta_rows:
        ids_raw = row.get("audio_ids")
        if isinstance(ids_raw, list):
            for x in ids_raw:
                if isinstance(x, str) and x.strip():
                    needed.add(x.strip())
        qid = row.get("query_audio_id")
        if isinstance(qid, str) and qid.strip():
            needed.add(qid.strip())
    audio_map = _load_audio_map(repo_id, revision=revision, needed_ids=needed)

    raw_rows: list[tuple[str, dict[str, Any], list[str], str]] = []
    for ordinal, row in enumerate(meta_rows):
        stable = _resolve_row_id(row)
        sample_id = (
            stable
            if stable is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_next_multiaudio",
                subset=subset,
                split=split,
                ordinal=ordinal,
            )
        )
        ids_raw = row.get("audio_ids")
        if not isinstance(ids_raw, list):
            raise RuntimeError(f"multiaudio row missing audio_ids subset={subset!r}")
        paths: list[str] = []
        for j, x in enumerate(ids_raw):
            if not isinstance(x, str) or not x.strip():
                raise RuntimeError(f"invalid audio_ids[{j}] subset={subset!r}")
            paths.append(
                _materialize_wav_bytes(
                    audio_map[x.strip()], stem=f"{sample_id}__ctx{j}"
                )
            )
        qid = row.get("query_audio_id")
        if not isinstance(qid, str) or not qid.strip():
            raise RuntimeError(
                f"multiaudio row missing query_audio_id subset={subset!r}"
            )
        q_path = _materialize_wav_bytes(
            audio_map[qid.strip()], stem=f"{sample_id}__query"
        )
        raw_rows.append((sample_id, row, paths, q_path))

    if workers > 1:

        def _build(
            item: tuple[str, dict[str, Any], list[str], str],
        ) -> DatasetExample:
            sample_id, rowd, paths, q_path = item
            return _build_multiaudio_dataset_example(
                rowd,
                sample_id=sample_id,
                audio_paths=paths,
                query_audio_path=q_path,
                split=split,
                task_id=task_id,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(_build, raw_rows)
        return

    for sample_id, rowd, paths, q_path in raw_rows:
        yield _build_multiaudio_dataset_example(
            rowd,
            sample_id=sample_id,
            audio_paths=paths,
            query_audio_path=q_path,
            split=split,
            task_id=task_id,
        )


def iter_hf_beans_next_examples(
    repo_id: str = BEANS_NEXT_HUB_REPO_ID,
    *,
    subset: str,
    split: str = "test",
    revision: str = "main",
    task_id: str | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> Iterator[DatasetExample]:
    """Yield ``DatasetExample`` rows for a BEANS-Next subset from HuggingFace Hub.

    Reads the two-table Parquet bundle (``beans_next_metadata.parquet`` +
    ``beans_next_audio.parquet``) and routes internally to the single-audio loader
    for tiers 1–3 or the multi-audio loader for tier 4 (in-context tasks), based
    on the known subset catalog.

    Parameters
    ----------
    repo_id
        HuggingFace dataset id. Defaults to ``EarthSpeciesProject/BEANS-Next``.
    subset
        Subset name, e.g. ``"crow-description"`` (tier 1) or ``"crow-4way"``
        (tier 4). The tier is resolved from the built-in catalog.
    split
        Split label stored on each ``DatasetExample`` (default ``"test"``).
    revision
        Hub git revision (default ``"main"``).
    task_id
        Optional eval-task id stored on each example.
    limit
        Optional maximum number of examples to yield.
    workers
        Parallel WAV materialization threads when ``>1``. Sequential when ``1``.

    Yields
    ------
    DatasetExample
        One example per metadata row matching ``subset``, in Parquet order.

    Raises
    ------
    KeyError
        If ``subset`` is not in the known BEANS-Next catalog.
    """
    s = subset.strip()
    if s not in ALL_SUBSETS:
        raise KeyError(
            f"Unknown BEANS-Next subset: {subset!r}. "
            f"Known subsets: {', '.join(sorted(ALL_SUBSETS))}"
        )
    fn = _iter_multiaudio_examples if is_multiaudio_subset(s) else _iter_single_examples
    yield from fn(
        repo_id,
        subset=s,
        split=split,
        revision=revision,
        task_id=task_id,
        limit=limit,
        workers=workers,
    )
