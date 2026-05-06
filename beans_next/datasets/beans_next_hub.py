"""Load BEANS-Next examples from the Hugging Face Hub dataset bundle.

Current layout (``EarthSpeciesProject/BEANS-Next``) uses:

- ``metadata.parquet``: one row per evaluation sample. Filter rows with the
  string ``task`` column (legacy tables may still expose ``subset``). ``tier``
  is an integer (1–4). Single-audio rows set ``file_name`` to a repo-relative
  path such as ``audio/<id>.wav``.
  Multi-audio rows use ``query_source_path`` + ``context_source_paths`` and/or
  ``source_audio_paths`` (legacy columns may still use ``query_audio_path`` +
  ``context_audio_paths`` + ``audio_paths``; see :func:`_multiaudio_repo_rel_paths`).
- ``audio/``: WAV (or other) files referenced by those paths (not embedded in
  Parquet).

Older revisions used ``beans_next_metadata.parquet`` + ``beans_next_audio.parquet``
(with ``audio_bytes``). That path remains supported when those files are present.

Callers use :func:`iter_hf_beans_next_examples` with a ``subset`` argument that
must match the Hub ``task`` string (e.g. ``\"crow-description\"``,
``\"crow-4way\"``).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from types import ModuleType
from typing import Any, Final

from huggingface_hub import HfApi, hf_hub_download

from beans_next.api.types import DatasetExample
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER

BEANS_NEXT_HUB_REPO_ID: Final[str] = "EarthSpeciesProject/BEANS-Next"
_METADATA_PARQUET_LEGACY: Final[str] = "beans_next_metadata.parquet"
_METADATA_PARQUET_CANONICAL: Final[str] = "metadata.parquet"
_AUDIO_PARQUET_LEGACY: Final[str] = "beans_next_audio.parquet"
_METADATA_FILE_ENV: Final[str] = "BEANS_NEXT_HF_BEANS_NEXT_METADATA_FILE"
_METADATA_FILE_ENV_COMPAT: Final[str] = "BEANS_PRO_HF_BEANS_NEXT_METADATA_FILE"

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


def _coerce_str_sequence(value: object) -> list[str] | None:
    """Coerce a Parquet-decoded value into a list of non-empty strings.

    HuggingFace-hosted Parquet sometimes encodes list columns as JSON strings
    (e.g. ``'["a", "b"]'``). Additionally, Arrow decoding can yield tuples or
    other ``Sequence`` implementations. This helper normalizes those variants.

    Returns
    -------
    list[str] | None
        Coerced strings, or ``None`` when ``value`` is not a valid string list.
    """
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        items = parsed
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
    else:
        return None

    out: list[str] = []
    for x in items:
        if not isinstance(x, str):
            return None
        sx = x.strip()
        if not sx:
            return None
        out.append(sx)
    return out


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


def _local_hub_file(repo_id: str, filename: str, *, revision: str) -> str:
    """Download (or reuse cache) a repo file and return a local filesystem path.

    Returns
    -------
    str
        Absolute path under the Hugging Face cache.
    """
    return str(
        hf_hub_download(
            repo_id,
            filename,
            repo_type="dataset",
            revision=revision,
        )
    )


@lru_cache(maxsize=64)
def _hub_dataset_files(repo_id: str, revision: str) -> frozenset[str]:
    """List repo-relative paths for a dataset revision (cached).

    Returns
    -------
    frozenset[str]
        Paths returned by ``HfApi.list_repo_files``.
    """
    paths = HfApi().list_repo_files(
        repo_id,
        repo_type="dataset",
        revision=revision,
    )
    return frozenset(paths)


def _hub_metadata_filename(repo_id: str, revision: str) -> str:
    """Resolve which metadata Parquet file exists on the Hub.

    Returns
    -------
    str
        Filename such as ``metadata.parquet`` or the legacy metadata name.

    Raises
    ------
    RuntimeError
        When no known metadata Parquet is present in the repo revision.
    """
    env = (
        os.environ.get(_METADATA_FILE_ENV, "").strip()
        or os.environ.get(_METADATA_FILE_ENV_COMPAT, "").strip()
    )
    if env:
        return env
    files = _hub_dataset_files(repo_id, revision)
    if _METADATA_PARQUET_CANONICAL in files:
        return _METADATA_PARQUET_CANONICAL
    if _METADATA_PARQUET_LEGACY in files:
        return _METADATA_PARQUET_LEGACY
    msg = (
        f"No metadata parquet found in {repo_id}@{revision!r}; expected "
        f"{_METADATA_PARQUET_CANONICAL!r} or {_METADATA_PARQUET_LEGACY!r}"
    )
    raise RuntimeError(msg)


def _hub_has_legacy_audio_parquet(repo_id: str, revision: str) -> bool:
    return _AUDIO_PARQUET_LEGACY in _hub_dataset_files(repo_id, revision)


def _hf_download_audio_path(repo_id: str, rel_path: str, *, revision: str) -> str:
    """Download (or take from cache) a repo-relative audio file.

    Returns
    -------
    str
        Absolute local path from ``hf_hub_download``.

    Raises
    ------
    ValueError
        When ``rel_path`` is empty after normalization.
    """
    rel = rel_path.strip().lstrip("/")
    if not rel:
        msg = "Hub audio path is empty"
        raise ValueError(msg)
    return str(
        hf_hub_download(
            repo_id,
            rel,
            repo_type="dataset",
            revision=revision,
        )
    )


def _row_matches_task(row: Mapping[str, Any], task: str) -> bool:
    """Match Hub row to requested subset / task string.

    Returns
    -------
    bool
        ``True`` when ``row["task"]`` or legacy ``row["subset"]`` equals ``task``.
    """
    if row.get("task") == task:
        return True
    return row.get("subset") == task


def _single_audio_rel_path(row: Mapping[str, Any]) -> str | None:
    """Return repo-relative audio path for a single-audio metadata row, if known.

    Returns
    -------
    str | None
        ``file_name`` when set, else ``audio/{audio_id}.wav`` when ``audio_id`` exists.
    """
    fn = row.get("file_name")
    if isinstance(fn, str) and fn.strip():
        return fn.strip()
    aid = row.get("audio_id")
    if isinstance(aid, str) and aid.strip():
        return f"audio/{aid.strip()}.wav"
    return None


def _placeholder_count_from_messages(row: Mapping[str, Any]) -> int | None:
    messages_raw = row.get("messages")
    if not isinstance(messages_raw, list):
        return None
    for msg in messages_raw:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"].count(AUDIO_PLACEHOLDER)
    return None


def _multiaudio_repo_rel_paths(row: Mapping[str, Any]) -> list[str] | None:
    """Pick ordered repo-relative paths for multi-audio rows.

    Prefer context paths when they align with the user-message placeholder count.
    When the last context path does not match the query path but the counts match
    (few-shot templates), replace the last slot with the query path.

    This function supports both the newer column names:

    - ``query_source_path`` (formerly ``query_audio_path``)
    - ``context_source_paths`` (formerly ``context_audio_paths``)
    - ``source_audio_paths`` (formerly ``audio_paths``)
    - ``original_source_path`` (formerly ``audio_path_original_sample_rate``)

    Returns
    -------
    list[str] | None
        Ordered repo-relative paths, or ``None`` when paths cannot be inferred.
    """
    n_ph = _placeholder_count_from_messages(row)
    q_raw = row.get("query_source_path") or row.get("query_audio_path")
    q = q_raw.strip() if isinstance(q_raw, str) and q_raw.strip() else None

    cap = _coerce_str_sequence(
        row.get("context_source_paths") or row.get("context_audio_paths")
    )
    ap = _coerce_str_sequence(
        row.get("source_audio_paths") or row.get("audio_paths")
    )

    if cap is not None and n_ph is not None and len(cap) == n_ph:
        if q is not None and cap[-1].strip() != q:
            return cap[:-1] + [q]
        return list(cap)

    if ap is not None and n_ph is not None and len(ap) == n_ph:
        if q is not None and q not in ap:
            # Last slot is the query recording; ``audio_paths`` may omit ``q`` or
            # use a mismatched tail (see tier-4 4-way tasks on the Hub).
            return ap[:-1] + [q]
        return list(ap)

    if cap is not None and cap:
        if q is not None and cap[-1].strip() != q:
            return cap[:-1] + [q]
        return list(cap)

    if ap is not None and ap:
        return list(ap)

    return None


def _prefetch_hub_files(
    repo_id: str,
    rel_paths: Sequence[str],
    *,
    revision: str,
    workers: int,
) -> dict[str, str]:
    """Download unique repo-relative paths.

    Returns
    -------
    dict[str, str]
        Maps each repo-relative path to a local cached file path.
    """
    unique = list(dict.fromkeys(rel_paths))
    if not unique:
        return {}

    def _one(rel: str) -> tuple[str, str]:
        return rel, _hf_download_audio_path(repo_id, rel, revision=revision)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(_one, unique))
    return dict(map(_one, unique))


def _load_audio_map(
    repo_id: str,
    *,
    revision: str,
    needed_ids: set[str],
) -> dict[str, bytes]:
    """Scan legacy ``beans_next_audio.parquet`` for WAV bytes by ``audio_id``.

    Returns
    -------
    dict[str, bytes]
        ``audio_id`` → ``audio_bytes`` for each requested id.

    Raises
    ------
    RuntimeError
        When any ``needed_ids`` are missing after a full scan.
    """
    if not needed_ids:
        return {}
    audio_path = _local_hub_file(
        repo_id, _AUDIO_PARQUET_LEGACY, revision=revision
    )
    out: dict[str, bytes] = {}
    for row in iter_parquet_row_dicts(audio_path):
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
            f"{_AUDIO_PARQUET_LEGACY} missing audio_id(s): "
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

    meta_name = _hub_metadata_filename(repo_id, revision)
    meta_path = _local_hub_file(repo_id, meta_name, revision=revision)
    meta_rows: list[dict[str, Any]] = []
    for row in iter_parquet_row_dicts(meta_path):
        if not _row_matches_task(row, subset):
            continue
        meta_rows.append(dict(row))
        if limit is not None and len(meta_rows) >= limit:
            break

    legacy_audio = _hub_has_legacy_audio_parquet(repo_id, revision)
    rels: list[str | None] = [_single_audio_rel_path(r) for r in meta_rows]
    legacy_ids: set[str] = set()
    for row, rel in zip(meta_rows, rels, strict=True):
        if rel is None:
            aid = row.get("audio_id")
            if isinstance(aid, str) and aid.strip():
                legacy_ids.add(aid.strip())

    audio_map: dict[str, bytes] = {}
    if legacy_ids:
        if not legacy_audio:
            keys_preview = ", ".join(sorted(legacy_ids)[:8])
            raise RuntimeError(
                "Single-audio Hub rows lack file_name/audio paths but "
                f"{_AUDIO_PARQUET_LEGACY!r} is not in the repo. audio_id(s): "
                f"{keys_preview}"
            )
        audio_map = _load_audio_map(repo_id, revision=revision, needed_ids=legacy_ids)

    to_fetch = [r for r in rels if r is not None]
    path_by_rel = _prefetch_hub_files(
        repo_id, to_fetch, revision=revision, workers=max(1, workers)
    )

    raw_rows: list[tuple[str, dict[str, Any], str]] = []
    for ordinal, row in enumerate(meta_rows):
        rel = rels[ordinal]
        stable = _resolve_row_id(row)
        sample_id = (
            stable
            if stable is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_next", subset=subset, split=split, ordinal=ordinal
            )
        )
        if rel is not None:
            wav_path = path_by_rel[rel]
        else:
            aid = row.get("audio_id")
            if not isinstance(aid, str) or not aid.strip():
                raise RuntimeError(
                    f"metadata row missing file_name/audio_id subset={subset!r}"
                )
            key = aid.strip()
            if key not in audio_map:
                raise RuntimeError(
                    f"metadata row missing resolvable audio subset={subset!r} "
                    f"audio_id={key!r}"
                )
            wav_path = _materialize_wav_bytes(audio_map[key], stem=sample_id)
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

    meta_name = _hub_metadata_filename(repo_id, revision)
    meta_path = _local_hub_file(repo_id, meta_name, revision=revision)
    meta_rows: list[dict[str, Any]] = []
    for row in iter_parquet_row_dicts(meta_path):
        if not _row_matches_task(row, subset):
            continue
        meta_rows.append(dict(row))
        if limit is not None and len(meta_rows) >= limit:
            break

    legacy_audio = _hub_has_legacy_audio_parquet(repo_id, revision)

    raw_plan: list[
        tuple[str, dict[str, Any], list[str] | None, list[str] | None]
    ] = []
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
        rels = _multiaudio_repo_rel_paths(row)
        ids = _coerce_str_sequence(row.get("audio_ids"))
        raw_plan.append((sample_id, row, rels, ids))

    needed_legacy: set[str] = set()
    for _sid, row, rels, ids in raw_plan:
        if rels is not None:
            continue
        if ids is not None:
            needed_legacy.update(ids)
        qid = row.get("query_audio_id")
        if isinstance(qid, str) and qid.strip():
            needed_legacy.add(qid.strip())

    audio_map: dict[str, bytes] = {}
    if needed_legacy:
        if not legacy_audio:
            raise RuntimeError(
                "multiaudio row missing repo-relative audio paths and "
                f"{_AUDIO_PARQUET_LEGACY!r} is not available for subset={subset!r}"
            )
        audio_map = _load_audio_map(
            repo_id, revision=revision, needed_ids=needed_legacy
        )

    all_rels: list[str] = []
    for _sid, _row, rels, _ids in raw_plan:
        if rels is not None:
            all_rels.extend(rels)
    path_by_rel = _prefetch_hub_files(
        repo_id, all_rels, revision=revision, workers=max(1, workers)
    )

    raw_rows: list[tuple[str, dict[str, Any], list[str], str]] = []
    for sample_id, row, rels, ids in raw_plan:
        if rels is not None:
            local_paths = [path_by_rel[r] for r in rels]
            q_path = local_paths[-1]
            raw_rows.append((sample_id, row, local_paths, q_path))
            continue

        if ids is None:
            keys = ", ".join(sorted(row.keys()))
            raise RuntimeError(
                "multiaudio row missing repo-relative audio lists and audio_ids "
                f"subset={subset!r} keys=[{keys}]"
            )
        paths: list[str] = []
        for j, x in enumerate(ids):
            paths.append(
                _materialize_wav_bytes(
                    audio_map[x], stem=f"{sample_id}__ctx{j}"
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

    Reads ``metadata.parquet`` (or the legacy metadata filename), resolves
    repo-relative audio paths under ``audio/``, and routes to the single-audio
    loader for tiers 1–3 or the multi-audio loader for tier 4, based on the
    built-in subset catalog.

    Parameters
    ----------
    repo_id
        HuggingFace dataset id. Defaults to ``EarthSpeciesProject/BEANS-Next``.
    subset
        Hub ``task`` string, e.g. ``"crow-description"`` (tier 1) or
        ``"crow-4way"`` (tier 4).
    split
        Split label stored on each ``DatasetExample`` (default ``"test"``).
    revision
        Hub git revision (default ``"main"``).
    task_id
        Optional eval-task id stored on each example.
    limit
        Optional maximum number of examples to yield.
    workers
        Parallel download / WAV materialization threads when ``>1``.

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
