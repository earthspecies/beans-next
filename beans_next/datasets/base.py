"""Shared helpers for HuggingFace dataset loaders.

Sample identifiers
------------------
Rows from ``EarthSpeciesProject/BEANS-Zero`` expose a stable string ``id``
(UUID). When a row provides a non-empty string under the configured id key
(default ``"id"``), that exact string is used as ``DatasetExample.sample_id``.

When no stable id is present, loaders synthesize::

    beanspro:hf:<64-char lower-case hex sha256>

The digest is ``hashlib.sha256`` over UTF-8 segments joined by ``\\0`` in
order: ``path_or_id``, ``split`` (empty string if missing), ``revision`` (empty
if missing), and ``ordinal`` (decimal string). ``ordinal`` is the zero-based
row index for map-style datasets, or the zero-based count of rows **yielded**
after optional filtering for streaming iterators. This scheme is deterministic,
process-stable, and does **not** use Python's built-in :func:`hash` (which is
salted per interpreter).

Raises
------
ImportError
    If optional ``datasets`` is required but not installed, or if audio
    materialization is required but ``soundfile`` is not installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from functools import lru_cache
from types import ModuleType
from typing import Any

from beans_next.api.types import DatasetExample

_SAMPLE_ID_PREFIX = "beanspro:hf:"
_AUDIO_CACHE_ENV = "BEANS_NEXT_HF_AUDIO_CACHE_DIR"
_AUDIO_CACHE_ENV_COMPAT = "BEANS_PRO_HF_AUDIO_CACHE_DIR"
_DEFAULT_HF_AUDIO_SAMPLE_RATE_ENV = "BEANS_NEXT_HF_DEFAULT_SAMPLE_RATE_HZ"
_DEFAULT_HF_AUDIO_SAMPLE_RATE_ENV_COMPAT = "BEANS_PRO_HF_DEFAULT_SAMPLE_RATE_HZ"
_DEFAULT_HF_AUDIO_SAMPLE_RATE_HZ = 32000


@lru_cache(maxsize=1)
def _audio_cache_dir() -> str:
    """Return (and lazily create) the audio materialization cache directory.

    Returns
    -------
    str
        Absolute path to the cache directory.
    """
    root = os.environ.get(_AUDIO_CACHE_ENV) or os.environ.get(_AUDIO_CACHE_ENV_COMPAT)
    if root is None or not root.strip():
        root = tempfile.mkdtemp(prefix="beans-next-hf-audio-")
    os.makedirs(root, exist_ok=True)
    return root


@lru_cache(maxsize=1)
def require_datasets() -> ModuleType:
    """Import and return the ``datasets`` package (cached).

    Returns
    -------
    module
        The installed ``datasets`` module.

    Raises
    ------
    ImportError
        If ``datasets`` is not installed.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        msg = (
            "HuggingFace dataset loaders require the optional `datasets` "
            "dependency. Install it in your environment (for example "
            "`uv pip install datasets`)."
        )
        raise ImportError(msg) from exc
    return datasets


def dataset_name_equals(component: str) -> Callable[[Mapping[str, Any]], bool]:
    """Build a row predicate that keeps BEANS-Zero ``dataset_name`` rows.

    Parameters
    ----------
    component
        Value compared to ``row[\"dataset_name\"]`` (for example ``\"esc50\"``).

    Returns
    -------
    Callable[[Mapping[str, Any]], bool]
        Predicate suitable for ``row_filter`` parameters on loaders.

    Raises
    ------
    ValueError
        If ``component`` is empty after stripping.
    """
    name = component.strip()
    if not name:
        msg = "component must be a non-empty string"
        raise ValueError(msg)

    def _pred(row: Mapping[str, Any]) -> bool:
        v = row.get("dataset_name")
        return isinstance(v, str) and v == name

    return _pred


def synthesize_hf_sample_id(
    *,
    path_or_id: str,
    split: str | None,
    revision: str | None,
    ordinal: int,
) -> str:
    """Build the fallback ``beanspro:hf:…`` sample id for a Hub row.

    Parameters
    ----------
    path_or_id
        Dataset path or repo id passed to ``datasets.load_dataset``.
    split
        Split name, if any.
    revision
        Hub git revision / branch / tag, if any.
    ordinal
        Non-negative stable ordering token (map index or streaming yield
        counter).

    Returns
    -------
    str
        Deterministic synthetic id.

    Raises
    ------
    ValueError
        If ``ordinal`` is negative.
    """
    if ordinal < 0:
        msg = "ordinal must be non-negative"
        raise ValueError(msg)
    parts = (
        path_or_id,
        split if split is not None else "",
        revision if revision is not None else "",
        str(int(ordinal)),
    )
    joined = "\0".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"{_SAMPLE_ID_PREFIX}{digest}"


def resolve_hf_sample_id(
    row: Mapping[str, Any],
    *,
    path_or_id: str,
    split: str | None,
    revision: str | None,
    ordinal: int,
    id_field: str = "id",
) -> str:
    """Pick ``sample_id`` for a Hub row using the BEANS-Next scheme.

    Parameters
    ----------
    row
        One decoded example mapping.
    path_or_id, split, revision, ordinal
        Forwarded to :func:`synthesize_hf_sample_id` when no explicit id exists.
    id_field
        Row key inspected for a stable string identifier.

    Returns
    -------
    str
        Either the trimmed row id or a synthetic digest id.
    """
    raw = row.get(id_field)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            return stripped
    return synthesize_hf_sample_id(
        path_or_id=path_or_id,
        split=split,
        revision=revision,
        ordinal=ordinal,
    )


def _audio_ref(audio_val: object) -> dict[str, Any] | None:
    if isinstance(audio_val, dict):
        audio_dict = audio_val
        ref: dict[str, Any] = {}
        path = audio_dict.get("path")
        if path is not None:
            ref["path"] = path
        sr = audio_dict.get("sampling_rate")
        if sr is not None:
            ref["sampling_rate"] = sr
        arr = audio_dict.get("array")
        if arr is not None:
            try:
                ref["num_samples"] = len(arr)
            except TypeError:
                ref["num_samples_unknown"] = True
        return ref or None

    # Some streaming dataset pipelines yield audio already decoded as a list of
    # floats (no dict wrapper, no sample rate). Preserve only a lightweight hint.
    if isinstance(audio_val, list):
        try:
            n = len(audio_val)
        except TypeError:
            n = None
        return {"num_samples": n} if n is not None else {"num_samples_unknown": True}

    return None


def _ensure_audio_path_from_array(
    audio_val: object,
    *,
    sample_id: str,
    sample_rate: int | None = None,
) -> str | None:
    """Materialize a temp WAV for decoded HF audio arrays.

    HuggingFace `datasets` can yield decoded audio rows where `audio.path` is
    missing/empty but an in-memory `audio.array` and `audio.sampling_rate` are
    present. BEANS-Next prompt templates may request `payload_type: file_path`,
    so we materialize a local WAV file in that case and return its path.

    Parameters
    ----------
    audio_val
        The value stored under the ``audio`` key of a Hub row. May be a
        ``dict`` (standard HF Audio feature), a ``list`` or ``numpy.ndarray``
        of float samples (streaming/numpy-format variants), or any other object
        (returns ``None``).
    sample_id
        Stable row identifier; used to derive a deterministic cache filename.
    sample_rate
        Explicit sample rate in Hz to use when ``audio_val`` is a bare array
        (``list`` or ``numpy.ndarray``) that carries no embedded rate.
        Falls back to the ``BEANS_NEXT_HF_DEFAULT_SAMPLE_RATE_HZ`` env var (compat:
        ``BEANS_PRO_HF_DEFAULT_SAMPLE_RATE_HZ``) and
        then to the built-in default (32 kHz).

    Returns
    -------
    str | None
        Path to a cached WAV file when audio can be materialized, otherwise
        ``None``.

    Raises
    ------
    ImportError
        If ``soundfile`` is not installed but audio materialization is required.
    """
    # Typical `datasets` audio feature shape:
    # {"array": np.ndarray, "sampling_rate": int, "path": str}
    if isinstance(audio_val, dict):
        path = audio_val.get("path")
        if isinstance(path, str) and path.strip():
            return path
        array = audio_val.get("array")
        sr = audio_val.get("sampling_rate")
        if array is None or not isinstance(sr, int):
            return None
    # Some streaming paths or set_format("numpy") yield audio already as a
    # list or numpy array of float samples with no sample rate embedded.
    # For BEANS-Zero, 32kHz is the canonical rate, so we use that as a default.
    elif isinstance(audio_val, list):
        array = audio_val
        if sample_rate is not None:
            sr = sample_rate
        else:
            sr_raw = os.environ.get(
                _DEFAULT_HF_AUDIO_SAMPLE_RATE_ENV
            ) or os.environ.get(_DEFAULT_HF_AUDIO_SAMPLE_RATE_ENV_COMPAT)
            if sr_raw is not None and sr_raw.strip():
                try:
                    sr = int(sr_raw)
                except ValueError:
                    sr = _DEFAULT_HF_AUDIO_SAMPLE_RATE_HZ
            else:
                sr = _DEFAULT_HF_AUDIO_SAMPLE_RATE_HZ
    else:
        # Handle numpy arrays (returned by set_format("numpy")).
        try:
            import numpy as np

            if isinstance(audio_val, np.ndarray):
                array = audio_val
                if sample_rate is not None:
                    sr = sample_rate
                else:
                    sr_raw = os.environ.get(
                        _DEFAULT_HF_AUDIO_SAMPLE_RATE_ENV
                    ) or os.environ.get(_DEFAULT_HF_AUDIO_SAMPLE_RATE_ENV_COMPAT)
                    if sr_raw is not None and sr_raw.strip():
                        try:
                            sr = int(sr_raw)
                        except ValueError:
                            sr = _DEFAULT_HF_AUDIO_SAMPLE_RATE_HZ
                    else:
                        sr = _DEFAULT_HF_AUDIO_SAMPLE_RATE_HZ
            else:
                return None
        except ImportError:
            return None

    # Avoid using raw sample ids as filenames (can contain `/` or other separators).
    safe_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    out_path = os.path.join(_audio_cache_dir(), f"{safe_id}.wav")
    if os.path.exists(out_path):
        return out_path

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - core deps should include soundfile
        msg = (
            "Audio materialization for HuggingFace rows requires the `soundfile` "
            "dependency, but it is not installed."
        )
        raise ImportError(msg) from exc

    sf.write(out_path, array, sr)
    return out_path


def ensure_audio_paths_from_sequence(
    audio_val: object,
    *,
    sample_id: str,
) -> list[str] | None:
    """Materialize a list of local WAV paths for a multi-audio Hub row.

    Parameters
    ----------
    audio_val
        Value stored under a multi-audio key (typically ``row["audio"]``) when
        the Hub dataset uses ``Sequence(Audio)``. Expected to be a list of
        per-audio elements shaped like HuggingFace ``Audio`` decodes, e.g.
        ``{"array": ..., "sampling_rate": ..., "path": ...}``. Elements may also
        be bare arrays (streaming variants), in which case they are written to a
        cache WAV path.
    sample_id
        Stable row identifier used to derive deterministic cache file names.

    Returns
    -------
    list[str] | None
        One local file path per audio slot, or ``None`` when ``audio_val`` is
        missing/unrecognized.

    Raises
    ------
    ImportError
        If audio materialization is required but the optional ``soundfile``
        dependency is missing.
    ValueError
        If an element is present but cannot be converted into a local path.
    """
    if audio_val is None:
        return None
    if not isinstance(audio_val, list):
        return None

    out: list[str] = []
    for i, element in enumerate(audio_val):
        path = _ensure_audio_path_from_array(element, sample_id=f"{sample_id}__audio{i}")
        if path is None:
            msg = (
                "Unable to materialize audio slot from HuggingFace row. "
                f"sample_id={sample_id!r} index={i}"
            )
            raise ValueError(msg)
        out.append(path)
    return out


def hf_row_metadata(row: Mapping[str, Any], *, sample_id: str) -> dict[str, Any]:
    """Copy lightweight, JSON-friendly fields from a Hub row.

    Omits raw audio arrays. Numeric / string scalars for common BEANS-Zero keys
    are preserved when present.

    When the Hub row includes decoded ``audio`` with a string ``path``, the
    flattened key ``audio_path`` is also set so prompt specs that declare
    ``metadata_key: audio_path`` resolve without dotted lookups.

    Parameters
    ----------
    row
        One example mapping from ``datasets``.

    Returns
    -------
    dict[str, Any]
        Metadata suitable for ``DatasetExample.metadata``.
    """
    meta: dict[str, Any] = {}
    audio_val = row.get("audio")
    audio_ref = _audio_ref(audio_val)
    if audio_ref is not None:
        meta["audio"] = audio_ref
        path_val = audio_ref.get("path")
        if isinstance(path_val, str) and path_val.strip():
            meta["audio_path"] = path_val
    if "audio_path" not in meta:
        direct_audio_path = row.get("audio_path")
        if isinstance(direct_audio_path, str) and direct_audio_path.strip():
            meta["audio_path"] = direct_audio_path
    if "audio_path" not in meta:
        # Parse sample_rate from the BEANS-Zero metadata JSON column so that
        # bare numpy/list arrays (set_format("numpy") path) get the correct rate.
        row_sample_rate: int | None = None
        row_meta_raw = row.get("metadata")
        if isinstance(row_meta_raw, str):
            try:
                row_meta_parsed = json.loads(row_meta_raw)
                sr_val = row_meta_parsed.get("sample_rate")
                if isinstance(sr_val, int):
                    row_sample_rate = sr_val
            except (json.JSONDecodeError, AttributeError):
                pass
        materialized = _ensure_audio_path_from_array(
            audio_val, sample_id=sample_id, sample_rate=row_sample_rate
        )
        if materialized is not None:
            meta["audio_path"] = materialized
    row_meta = row.get("metadata")
    if isinstance(row_meta, str):
        meta["hf_metadata"] = row_meta
    for key in (
        "instruction",
        "instruction_text",
        "file_name",
        "source_dataset",
        "dataset_name",
        "task",
        "license",
        "created_at",
    ):
        val = row.get(key)
        if isinstance(val, str | int | float | bool):
            meta[key] = val
    return meta


def hf_row_to_dataset_example(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    task_id: str | None,
    split: str | None,
    label_field: str = "output",
) -> DatasetExample:
    """Normalize a Hub row into a :class:`~beans_next.api.types.DatasetExample`.

    Parameters
    ----------
    row
        Raw row mapping.
    sample_id
        Resolved stable id (see :func:`resolve_hf_sample_id`).
    task_id
        Optional eval-task registry id for downstream prompt/metric wiring.
    split
        Dataset split name, if known.
    label_field
        Row key used for ground-truth ``labels`` (BEANS-Zero uses ``output``).

    Returns
    -------
    DatasetExample
        Validated pipeline row.
    """
    labels: str | list[str] | dict[str, Any] | None
    raw_labels = row.get(label_field)
    if raw_labels is None:
        labels = None
    elif isinstance(raw_labels, str | list | dict):
        labels = raw_labels
    else:
        labels = str(raw_labels)
    return DatasetExample(
        sample_id=sample_id,
        task_id=task_id,
        split=split,
        labels=labels,
        metadata=hf_row_metadata(row, sample_id=sample_id),
    )


def iter_filtered_indices(
    length: int,
    row_getter: Callable[[int], Mapping[str, Any]],
    row_filter: Callable[[Mapping[str, Any]], bool] | None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield ``(index, row)`` pairs optionally filtered by ``row_filter``.

    Parameters
    ----------
    length
        Number of rows in the map-style dataset.
    row_getter
        Callable returning the row mapping for a dataset index.
    row_filter
        Optional predicate; when ``None`` all rows are retained.

    Yields
    ------
    tuple[int, Mapping[str, Any]]
        Index and row for each retained example.
    """
    for idx in range(length):
        row = row_getter(idx)
        if row_filter is None or row_filter(row):
            yield idx, row
