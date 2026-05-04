"""Optional `esp_data`-backed dataset loader.

This module provides an **optional** fast-path for ESP-internal (and soon public)
dataset access via `esp_data`. It is intentionally a guarded import so that
BEANS-Next remains installable without any private dependencies.

The integration is config-driven: callers select `data_source="esp_data"` and
BEANS-Next will attempt to import and use `esp_data`. When `esp_data` is not
installed, this module raises `ImportError` with a clear, actionable message.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import signal
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from types import ModuleType

from beans_next.api.types import DatasetExample

_SAMPLE_ID_PREFIX = "beanspro:esp_data:"
_AUDIO_CACHE_ENV = "BEANS_PRO_ESP_AUDIO_CACHE_DIR"
_DIAGNOSTICS_ENV = "BEANS_PRO_ESP_DATA_DIAGNOSTICS"
_ROW_TIMEOUT_S_ENV = "BEANS_PRO_ESP_DATA_ROW_TIMEOUT_S"
_AUDIO_TIMEOUT_S_ENV = "BEANS_PRO_ESP_DATA_AUDIO_TIMEOUT_S"
_AUDIO_WRITE_RETRIES_ENV = "BEANS_PRO_ESP_DATA_AUDIO_WRITE_RETRIES"
_LOG_EVERY_N_ENV = "BEANS_PRO_ESP_DATA_LOG_EVERY_N"
# Controls parallel GCS download threads.  Set to e.g. 8 on nodes with ample CPUs.
_WORKERS_ENV = "BEANS_PRO_ESP_DATA_WORKERS"
# Sentinel key injected into rows when we bypass _process (no GCS audio download yet).
_DATA_ROOT_KEY = "_beans_next_data_root"

_LOG = logging.getLogger(__name__)


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class _AlarmTimeout:
    """Best-effort wall-clock timeout using `SIGALRM` (Unix main thread only).

    This is intentionally scoped to the `esp_data` integration to bound stalls
    caused by remote filesystem access during row/audio materialization.
    """

    def __init__(self, seconds: float | None, *, label: str) -> None:
        self._seconds = seconds
        self._label = label
        self._enabled = False
        self._old_handler: object | None = None

    def __enter__(self) -> None:
        if self._seconds is None or self._seconds <= 0:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        if not hasattr(signal, "SIGALRM"):
            return

        def _handler(_signum: int, _frame: object) -> None:  # pragma: no cover
            raise TimeoutError(self._label)

        self._enabled = True
        self._old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(self._seconds))

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object | None,
    ) -> None:
        if not self._enabled:
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if self._old_handler is not None:
            signal.signal(signal.SIGALRM, self._old_handler)  # type: ignore[arg-type]


def _row_get(row: Mapping[str, object], key: str, *, timeout_s: float | None) -> object:
    with _AlarmTimeout(timeout_s, label=f"timeout reading row[{key!r}]"):
        return row.get(key)


def _write_wav_with_retries(
    path: str,
    *,
    arr: object,
    sample_rate: int,
    retries: int,
    timeout_s: float | None,
) -> None:
    # Import inside so callers can degrade gracefully if soundfile isn't installed.
    import soundfile as sf  # type: ignore[import-not-found]

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with _AlarmTimeout(timeout_s, label="timeout writing WAV via soundfile"):
                sf.write(path, arr, int(sample_rate), format="WAV")
            return
        except (OSError, RuntimeError, TimeoutError) as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(min(0.25 * (2**attempt), 2.0))
    if last_exc is not None:  # pragma: no cover
        raise last_exc


def _download_gcs_to_wav(
    gcs_path: str,
    *,
    sample_id: str,
    timeout_s: float | None,
    diagnostics: bool = False,
) -> str | None:
    """Download a GCS audio path to a local WAV file using a daemon thread.

    Uses a daemon thread rather than `SIGALRM` so the timeout fires even when
    the GCS I/O stall occurs inside a C extension (e.g. urllib3 inside gcsfs),
    where `SIGALRM` cannot interrupt reliably.

    Parameters
    ----------
    gcs_path : str
        Absolute GCS URL, e.g. ``gs://bucket/path/to/audio.wav``.
    sample_id : str
        Sample identifier used to name the local cache file.
    timeout_s : float | None
        Wall-clock timeout for the download in seconds. ``None`` means wait
        indefinitely (not recommended for GCS paths).
    diagnostics : bool
        When ``True``, emit a WARNING log on timeout or download failure.

    Returns
    -------
    str | None
        Path to the materialized WAV file, or ``None`` on failure or timeout.
    """
    result: dict[str, object] = {"path": None, "error": None}

    def _do() -> None:
        try:
            esp = require_esp_data()
            read_audio_fn = getattr(getattr(esp, "io", None), "read_audio", None)
            if read_audio_fn is None:
                result["error"] = "esp_data.io.read_audio not available"
                return
            import numpy as np

            audio, sr = read_audio_fn(gcs_path)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            path = os.path.join(_audio_cache_dir(), f"{sample_id}.wav")
            _write_wav_with_retries(
                path, arr=audio, sample_rate=int(sr), retries=0, timeout_s=None
            )
            result["path"] = path
        except Exception as exc:
            result["error"] = str(exc)

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        if diagnostics:
            _LOG.warning(
                "esp_data GCS download timed out after %.1fs: %s sample_id=%s",
                timeout_s,
                gcs_path,
                sample_id,
            )
        return None

    err = result["error"]
    if err:
        if diagnostics:
            _LOG.warning(
                "esp_data GCS download failed %s sample_id=%s error=%s",
                gcs_path,
                sample_id,
                err,
            )
        return None

    return result["path"]  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _audio_cache_dir() -> str:
    root = os.environ.get(_AUDIO_CACHE_ENV)
    if root is None or not root.strip():
        root = tempfile.mkdtemp(prefix="beans-next-esp-audio-")
    os.makedirs(root, exist_ok=True)
    return root


def _materialize_wav_from_row_audio(
    row: Mapping[str, object],
    *,
    sample_id: str,
) -> str | None:
    """Write `row['audio']` to a temp WAV and return its path.

    Returns
    -------
    str | None
        Path to the materialized WAV file if the row contains a decodable audio
        payload and `soundfile` is available; otherwise ``None``.

    Raises
    ------
    RuntimeError
        If row/audio materialization hits a best-effort timeout. This is intended
        to turn silent I/O stalls (e.g. remote filesystem reads) into actionable,
        bounded failures.
    """
    diagnostics = os.environ.get(_DIAGNOSTICS_ENV, "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )
    row_timeout_s = float(_env_int(_ROW_TIMEOUT_S_ENV, default=0)) or None
    audio_timeout_s = float(_env_int(_AUDIO_TIMEOUT_S_ENV, default=0)) or None
    write_retries = max(0, _env_int(_AUDIO_WRITE_RETRIES_ENV, default=2))

    try:
        audio_val = _row_get(row, "audio", timeout_s=row_timeout_s)
    except TimeoutError as exc:
        raise RuntimeError(
            "esp_data row materialization timed out while reading `audio`. "
            f"sample_id={sample_id!r}. "
            f"Set `{_ROW_TIMEOUT_S_ENV}`/`{_AUDIO_TIMEOUT_S_ENV}` to tune timeouts, "
            f"or switch to HuggingFace loading (`data_source: hf`)."
        ) from exc
    if audio_val is None:
        return None

    try:
        sample_rate = _row_get(row, "sample_rate", timeout_s=row_timeout_s)
    except TimeoutError as exc:
        raise RuntimeError(
            "esp_data row materialization timed out while reading `sample_rate`. "
            f"sample_id={sample_id!r}. "
            f"Set `{_ROW_TIMEOUT_S_ENV}`/`{_AUDIO_TIMEOUT_S_ENV}` to tune timeouts, "
            f"or switch to HuggingFace loading (`data_source: hf`)."
        ) from exc
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        return None

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None

    if isinstance(audio_val, np.ndarray):
        arr = audio_val
    elif isinstance(audio_val, list):
        try:
            arr = np.asarray(audio_val, dtype=np.float32)
        except Exception:
            return None
    else:
        return None

    cache_dir = _audio_cache_dir()
    path = os.path.join(cache_dir, f"{sample_id}.wav")

    if importlib.util.find_spec("soundfile") is None:
        return None

    try:
        _write_wav_with_retries(
            path,
            arr=arr,
            sample_rate=int(sample_rate),
            retries=write_retries,
            timeout_s=audio_timeout_s,
        )
    except Exception as exc:
        if diagnostics:
            _LOG.warning(
                "esp_data WAV materialization failed sample_id=%s path=%s: %s",
                sample_id,
                path,
                exc,
            )
        return None
    return path


def require_esp_data() -> ModuleType:
    """Import and return the `esp_data` package.

    Returns
    -------
    types.ModuleType
        The imported `esp_data` module.

    Raises
    ------
    ImportError
        If `esp_data` is not installed in the current environment.
    """
    try:
        import esp_data  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        msg = (
            "esp_data dataset loading was requested, but `esp_data` is not installed. "
            "Install it in your environment (internal: `uv sync` in an env that "
            "includes esp_data; public: `uv pip install esp-data` once released), "
            "or switch to HuggingFace loading by setting `data_source: hf`."
        )
        raise ImportError(msg) from exc
    return esp_data


def synthesize_esp_data_sample_id(
    *,
    dataset: str,
    subset: str,
    split: str,
    ordinal: int,
) -> str:
    """Build a deterministic `sample_id` for esp_data rows missing a stable id.

    Parameters
    ----------
    dataset
        Dataset family identifier (for this module: typically `"beans_zero"`).
    subset
        Subset name (e.g. `"esc50"`).
    split
        Split name (e.g. `"test"`).
    ordinal
        Zero-based ordinal in the yielded stream.

    Returns
    -------
    str
        Stable synthetic sample id.
    """
    parts = (dataset, subset, split, str(int(ordinal)))
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{_SAMPLE_ID_PREFIX}{digest}"


def _resolve_row_id(row: Mapping[str, object]) -> str | None:
    raw = row.get("id") or row.get("sample_id") or row.get("uuid")
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            return stripped
    return None


def _audio_path_from_row(row: Mapping[str, object]) -> str | None:
    direct = row.get("audio_path")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    # esp_data's BEANS-Zero rows commonly expose normalized paths by sample rate.
    # Prefer a consistent resampled path if present.
    for key in (
        "audio_path_32KHz",
        "audio_path_16KHz",
        "audio_path_original_sample_rate",
    ):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    audio = row.get("audio")
    if isinstance(audio, Mapping):
        path = audio.get("path")
        if isinstance(path, str) and path.strip():
            return path.strip()
    return None


def _labels_from_row(
    row: Mapping[str, object],
) -> str | list[str] | dict[str, object] | None:
    val = row.get("output") if "output" in row else row.get("labels")
    if val is None:
        return None
    if isinstance(val, str | list | dict):
        return val
    return str(val)


def _load_beans_zero_rows_via_reflection(
    esp_data: ModuleType,
    *,
    subset: str,
    split: str,
) -> Iterator[Mapping[str, object]]:
    """Load rows from esp_data using best-effort reflection.

    `esp_data` APIs have evolved across internal versions. This function tries a
    small set of common call patterns and raises a clear error if none work.

    Yields
    ------
    Mapping[str, object]
        Raw dataset rows (mapping-like).

    Raises
    ------
    RuntimeError
        When no known esp_data entrypoint matches the installed version.
    """
    attempts: list[str] = []

    # Newer esp_data exposes a `BeansZero` dataset class. In that API, the "subset"
    # we want (e.g. "esc50") can be selected either:
    # - as a dedicated dataset split (e.g. split="esc50"), or
    # - by filtering the global split="test" by row["dataset_name"] == subset.
    beans_zero_cls = getattr(esp_data, "BeansZero", None)
    if callable(beans_zero_cls):
        esp_split = split
        try:
            ds = beans_zero_cls(split=esp_split)
        except TypeError:
            attempts.append(f"esp_data.BeansZero(split={esp_split!r})")
        else:
            try:
                # Bypass _process() which downloads audio from GCS for every row.
                # Instead, access the underlying polars DataFrame directly (metadata
                # only, no audio I/O) and attach data_root so callers can construct
                # GCS paths and download with their own bounded timeout.
                data_root = str(getattr(ds, "data_root", ""))
                backend_df = getattr(getattr(ds, "_data", None), "_df", None)
                if backend_df is not None and hasattr(backend_df, "iter_rows"):
                    for raw in backend_df.iter_rows(named=True):  # type: ignore[union-attr]
                        if split == "test" and raw.get("dataset_name") != subset:
                            continue
                        out: dict[str, object] = dict(raw)
                        if data_root:
                            out[_DATA_ROOT_KEY] = data_root
                        yield out
                    return
                # Fallback when the private _df attribute is unavailable:
                # iterate via __iter__ which calls _process → GCS download per row.
                if split == "test":
                    for row in ds:  # type: ignore[misc]
                        if row.get("dataset_name") == subset:
                            yield row
                else:
                    yield from ds  # type: ignore[misc]
                return
            except TypeError:
                attempts.append(f"iter(esp_data.BeansZero(split={esp_split!r}))")

    fn = getattr(esp_data, "load_dataset", None)
    if callable(fn):
        for kwargs in (
            {"name": "beans_zero", "subset": subset, "split": split},
            {"dataset": "beans_zero", "subset": subset, "split": split},
            {"path": "beans_zero", "subset": subset, "split": split},
        ):
            try:
                obj = fn(**kwargs)
            except TypeError:
                attempts.append(f"esp_data.load_dataset({kwargs!r})")
            else:
                if isinstance(obj, Iterator):
                    yield from obj
                    return
                if isinstance(obj, list):
                    yield from obj
                    return
                if hasattr(obj, "__iter__"):
                    yield from obj  # type: ignore[misc]
                    return

    fn2 = getattr(esp_data, "get_dataset", None)
    if callable(fn2):
        for kwargs in ({"name": "beans_zero"}, {"dataset": "beans_zero"}):
            try:
                ds = fn2(**kwargs)
            except TypeError:
                attempts.append(f"esp_data.get_dataset({kwargs!r})")
                continue
            getter = getattr(ds, "iter", None) or getattr(ds, "iterate", None)
            if callable(getter):
                try:
                    yield from getter(subset=subset, split=split)
                    return
                except TypeError:
                    attempts.append(
                        f"{type(ds).__name__}.iter(subset={subset!r}, split={split!r})"
                    )

    msg = (
        "Unable to load BEANS-Zero rows via `esp_data`. "
        "BEANS-Next attempted known esp_data call patterns but none matched this "
        "installed esp_data version.\n"
        f"Attempts: {attempts or ['<no callable entrypoints found>']}\n"
        "Fix: update this loader to match your esp_data API (or pin esp_data), "
        "or use HuggingFace loading (`data_source: hf`)."
    )
    raise RuntimeError(msg)


def _load_beans_next_rows_via_reflection(
    esp_data: ModuleType,
    *,
    split: str,
) -> Iterator[Mapping[str, object]]:
    """Load BeansPro rows from esp_data by using the `BeansPro` dataset class.

    This mirrors the BEANS-Zero fast-path: iterate the polars backend (metadata-only)
    and inject `_DATA_ROOT_KEY` so audio resolution uses the bounded GCS download
    path in `_resolve_audio_for_row`.

    Parameters
    ----------
    esp_data
        Imported `esp_data` module.
    split
        BeansPro split name (e.g. `"crow-description"`, `"alarm-call-presence"`).

    Yields
    ------
    Mapping[str, object]
        Raw dataset rows (mapping-like).

    Raises
    ------
    RuntimeError
        If `esp_data.BeansPro` is unavailable or the installed API does not match
        expected access patterns.
    """
    beans_next_cls = getattr(esp_data, "BeansPro", None)
    if not callable(beans_next_cls):
        yield from _load_beans_next_rows_from_gcs_jsonl(split=split)
        return

    try:
        ds = beans_next_cls(split=split)
    except TypeError as exc:
        raise RuntimeError(
            f"Unable to construct `esp_data.BeansPro(split={split!r})` (API mismatch). "
            "Fix: update this loader to match your esp_data version, or switch to "
            "HuggingFace loading (`data_source: hf`)."
        ) from exc

    data_root = str(getattr(ds, "data_root", ""))
    backend_df = getattr(getattr(ds, "_data", None), "_df", None)
    if backend_df is not None and hasattr(backend_df, "iter_rows"):
        for raw in backend_df.iter_rows(named=True):  # type: ignore[union-attr]
            out: dict[str, object] = dict(raw)
            if data_root:
                out[_DATA_ROOT_KEY] = data_root
            yield out
        return

    # Fallback when the private _df attribute is unavailable:
    # iterate via __iter__ which calls _process → potential GCS download per row.
    if hasattr(ds, "__iter__"):
        yield from ds  # type: ignore[misc]
        return

    raise RuntimeError(
        "Unable to iterate BeansPro rows from `esp_data.BeansPro`. "
        "Fix: update this loader to match your esp_data version, or switch to "
        "HuggingFace loading (`data_source: hf`)."
    )


def _load_beans_next_rows_from_gcs_jsonl(
    *, split: str
) -> Iterator[Mapping[str, object]]:
    """Load BeansPro rows by streaming public GCS JSONL metadata.

    This is a compatibility fallback for environments where the installed
    `esp_data` package does not yet ship the `BeansPro` dataset class.

    Parameters
    ----------
    split
        BeansPro split name (e.g. `"crow-description"`, `"alarm-call-presence"`).

    Yields
    ------
    Mapping[str, object]
        JSON-decoded row dicts. Each row includes a `_DATA_ROOT_KEY` entry so
        `_resolve_audio_for_row` can materialize audio via bounded GCS download.

    Raises
    ------
    RuntimeError
        If the split is unknown or the JSONL cannot be read/parsed.
    """
    import json

    # Import inside to keep this module optional and avoid hard deps when
    # `data_source != esp_data`.
    try:
        import fsspec  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "BeansPro fallback loading requires `fsspec` (and `gcsfs` for `gs://`). "
            "Install them or use a different data source."
        ) from exc

    split_to_jsonl_and_root: dict[str, tuple[str, str]] = {
        # Acoustic description MCQ (audio copied under beans-next ingestion root)
        "crow-description": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/"
            "carrion_crow_descriptions/test.jsonl",
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/",
        ),
        "zebra-description": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/test.jsonl",
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/",
        ),
        # Mean F0 (audio lives under the shared f0-prediction audio root)
        "f0-mean-seen-taxa": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/f0_mean_seen_taxa/test.jsonl",
            "gs://esp-data-ingestion/f0-prediction/audio/",
        ),
        "f0-mean-heldout-taxa": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/f0_mean_heldout_taxa/test.jsonl",
            "gs://esp-data-ingestion/f0-prediction/audio/",
        ),
        # Presence / call-type — audio paths (audio_32k/...) are relative to the
        # xeno-canto mirror; do NOT set root to .../raw/audio_32k/ or the
        # segment would be doubled.
        "bird-presence": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/bird_presence/test.jsonl",
            "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        ),
        "mammal-presence": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/mammal_presence/test.jsonl",
            "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        ),
        "insect-presence": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/insect_presence/test.jsonl",
            "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        ),
        "amphibian-presence": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/amphibian_presence/test.jsonl",
            "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        ),
        "alarm-call-presence": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/alarm_call_presence/test.jsonl",
            "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        ),
        "flight-call-presence": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/flight_call_presence/test.jsonl",
            "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        ),
        "call-type-fixed-vocab": (
            "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/call_type_fixed_vocab/test.jsonl",
            "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        ),
    }

    cfg = split_to_jsonl_and_root.get(split)
    if cfg is None:
        known = ", ".join(sorted(split_to_jsonl_and_root))
        raise RuntimeError(
            f"Unknown BeansPro split {split!r} for fallback JSONL loading. "
            f"Known splits: {known}. "
            "Fix: add the split's JSONL path + audio data root mapping."
        )
    jsonl_path, data_root = cfg

    try:
        with fsspec.open(jsonl_path, "rt") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                if not isinstance(row, dict):
                    continue
                # Inject data root so downstream audio resolution uses the bounded,
                # stall-resistant GCS download path.
                row[_DATA_ROOT_KEY] = data_root
                yield row
    except Exception as exc:
        raise RuntimeError(
            f"Failed to stream BeansPro split JSONL from GCS: split={split!r} "
            f"path={jsonl_path!r}. Error: {exc}"
        ) from exc


def _resolve_audio_for_row(
    row: Mapping[str, object],
    *,
    sample_id: str,
    subset: str,
    split: str,
    diagnostics: bool,
) -> str | None:
    """Resolve a local WAV path for a single metadata row.

    Two paths depending on how the row was produced:

    - ``_DATA_ROOT_KEY`` present: ``_load_beans_zero_rows_via_reflection`` bypassed
      ``_process``; GCS audio not downloaded yet.  Constructs the absolute GCS URL
      and downloads via a daemon thread (immune to C-level I/O stalls).
    - ``_DATA_ROOT_KEY`` absent: ``_process`` already ran and decoded audio into
      ``row["audio"]`` as a numpy array; materializes a WAV from that array.

    Parameters
    ----------
    row
        Raw metadata row from the dataset loader.
    sample_id
        Stable sample identifier used to name the cache file.
    subset
        BEANS-Zero subset (used in error messages).
    split
        Dataset split (used in error messages).
    diagnostics
        When ``True``, emit WARNING logs on timeout or download failure.

    Returns
    -------
    str | None
        Absolute path to the local WAV file, or ``None`` when audio is
        unavailable or download failed.

    Raises
    ------
    RuntimeError
        If ``_process``-path audio materialization hits a best-effort timeout.
    """
    data_root = row.get(_DATA_ROOT_KEY)
    if isinstance(data_root, str) and data_root:
        gcs_rel: str | None = None
        for _key in (
            "audio_path_16KHz",
            "audio_path_32KHz",
            "audio_path_original_sample_rate",
        ):
            _val = row.get(_key)
            if isinstance(_val, str) and _val.strip():
                gcs_rel = _val.strip()
                break
        if not gcs_rel:
            return None
        gcs_abs = data_root.rstrip("/") + "/" + gcs_rel.lstrip("/")
        dl_timeout_raw = _env_int(_AUDIO_TIMEOUT_S_ENV, default=60)
        dl_timeout: float | None = float(dl_timeout_raw) if dl_timeout_raw > 0 else None
        return _download_gcs_to_wav(
            gcs_abs,
            sample_id=sample_id,
            timeout_s=dl_timeout,
            diagnostics=diagnostics,
        )

    audio_path = _audio_path_from_row(row)
    if isinstance(audio_path, str) and audio_path.strip():
        if not os.path.isabs(audio_path):
            if diagnostics:
                _LOG.info(
                    "esp_data non-absolute audio_path ignored "
                    "sample_id=%s audio_path=%s",
                    sample_id,
                    audio_path,
                )
            audio_path = None
    if audio_path is None:
        try:
            audio_path = _materialize_wav_from_row_audio(row, sample_id=sample_id)
        except TimeoutError as exc:
            msg = (
                "esp_data audio materialization timed out. "
                f"subset={subset!r} split={split!r} sample_id={sample_id!r}. "
                f"Set `{_ROW_TIMEOUT_S_ENV}`/`{_AUDIO_TIMEOUT_S_ENV}` "
                "to tune timeouts, or switch to HuggingFace loading "
                "(`data_source: hf`)."
            )
            raise RuntimeError(msg) from exc
    return audio_path


def _build_dataset_example(
    row: Mapping[str, object],
    *,
    sample_id: str,
    audio_path: str | None,
    split: str,
    task_id: str | None,
) -> DatasetExample:
    """Assemble a `DatasetExample` from a metadata row and a resolved audio path.

    Parameters
    ----------
    row
        Raw metadata row from the dataset loader.
    sample_id
        Stable sample identifier.
    audio_path
        Absolute local WAV path, or ``None`` when audio is unavailable.
    split
        Dataset split stored on the example.
    task_id
        Optional eval-task id stored on the example.

    Returns
    -------
    DatasetExample
        Fully assembled example ready for the runner.
    """
    meta: dict[str, object] = {}
    if isinstance(audio_path, str) and audio_path.strip():
        meta["audio_path"] = audio_path
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
    instruction = row.get("instruction")
    instruction_text = row.get("instruction_text")
    if isinstance(instruction, str) and instruction.strip():
        meta.setdefault("instruction", instruction.strip())
    elif isinstance(instruction_text, str) and instruction_text.strip():
        meta.setdefault("instruction", instruction_text.strip())
    return DatasetExample(
        sample_id=sample_id,
        task_id=task_id,
        split=split,
        labels=_labels_from_row(row),
        metadata=meta,
    )


def iter_esp_data_beans_zero_examples(
    *,
    subset: str,
    split: str,
    task_id: str | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> Iterator[DatasetExample]:
    """Yield `DatasetExample` rows for a BEANS-Zero subset via `esp_data`.

    Parameters
    ----------
    subset
        BEANS-Zero subset id (e.g. `"esc50"`, `"enabirds"`, `"captioning"`).
    split
        Split name (typically `"test"` for BEANS-Zero).
    task_id
        Optional eval-task id stored on each yielded example.
    limit
        Optional maximum number of examples to yield.
    workers
        Number of parallel threads for GCS audio downloads.  ``1`` (default)
        downloads sequentially; values ``>1`` collect all metadata rows first
        then issue concurrent GCS downloads, which significantly reduces
        wall-clock time when network latency dominates.

    Yields
    ------
    DatasetExample
        One normalized example at a time, in dataset order.
    """
    esp_mod = require_esp_data()
    diagnostics = os.environ.get(_DIAGNOSTICS_ENV, "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )
    log_every_n = max(1, _env_int(_LOG_EVERY_N_ENV, default=50))
    # Allow env-var override so Slurm jobs can set BEANS_PRO_ESP_DATA_WORKERS=8
    # without changing CLI args.
    if workers == 1:
        workers = max(1, _env_int(_WORKERS_ENV, default=1))

    if workers > 1:
        yield from _iter_esp_data_concurrent(
            esp_mod,
            subset=subset,
            split=split,
            task_id=task_id,
            limit=limit,
            workers=workers,
            diagnostics=diagnostics,
        )
        return

    kept = 0
    ordinal = 0
    last_heartbeat = time.monotonic()
    for raw in _load_beans_zero_rows_via_reflection(
        esp_mod, subset=subset, split=split
    ):
        if diagnostics and (kept == 0 or kept % log_every_n == 0):
            now = time.monotonic()
            dt = now - last_heartbeat
            last_heartbeat = now
            _LOG.info(
                "esp_data iter heartbeat subset=%s split=%s "
                "kept=%s ordinal=%s dt=%.2fs",
                subset,
                split,
                kept,
                ordinal,
                dt,
            )
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        stable_id = _resolve_row_id(row)
        sample_id = (
            stable_id
            if stable_id is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_zero", subset=subset, split=split, ordinal=ordinal
            )
        )
        audio_path = _resolve_audio_for_row(
            row,
            sample_id=sample_id,
            subset=subset,
            split=split,
            diagnostics=diagnostics,
        )
        yield _build_dataset_example(
            row,
            sample_id=sample_id,
            audio_path=audio_path,
            split=split,
            task_id=task_id,
        )
        kept += 1
        ordinal += 1
        if limit is not None and kept >= limit:
            return


def _iter_esp_data_concurrent(
    esp_mod: object,
    *,
    subset: str,
    split: str,
    task_id: str | None,
    limit: int | None,
    workers: int,
    diagnostics: bool,
) -> Iterator[DatasetExample]:
    """Concurrent GCS-download path for `iter_esp_data_beans_zero_examples`.

    Collects all metadata rows first (polars iteration, no I/O), then issues
    GCS downloads concurrently via a ``ThreadPoolExecutor``.  Results are yielded
    in original dataset order.

    Parameters
    ----------
    esp_mod
        The imported ``esp_data`` module.
    subset
        BEANS-Zero subset id.
    split
        Dataset split.
    task_id
        Optional eval-task id.
    limit
        Optional cap on the number of examples.
    workers
        Number of parallel download threads.
    diagnostics
        When ``True``, emit WARNING logs on timeout or download failure.

    Yields
    ------
    DatasetExample
        Examples in dataset order.
    """

    # Phase 1: collect all metadata rows (polars only, no network I/O).
    raw_rows: list[tuple[int, str, Mapping[str, object]]] = []
    for ordinal, raw in enumerate(
        _load_beans_zero_rows_via_reflection(
            esp_mod,  # type: ignore[arg-type]
            subset=subset,
            split=split,
        )
    ):
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        stable_id = _resolve_row_id(row)
        sample_id = (
            stable_id
            if stable_id is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_zero", subset=subset, split=split, ordinal=ordinal
            )
        )
        raw_rows.append((ordinal, sample_id, row))
        if limit is not None and len(raw_rows) >= limit:
            break

    # Phase 2: concurrent GCS downloads; executor.map preserves input order.
    def _dl(item: tuple[int, str, Mapping[str, object]]) -> DatasetExample:
        _, sample_id, row = item
        audio_path = _resolve_audio_for_row(
            row,
            sample_id=sample_id,
            subset=subset,
            split=split,
            diagnostics=diagnostics,
        )
        return _build_dataset_example(
            row,
            sample_id=sample_id,
            audio_path=audio_path,
            split=split,
            task_id=task_id,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_dl, raw_rows)


def iter_esp_data_beans_next_examples(
    *,
    subset: str,
    split: str,
    task_id: str | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> Iterator[DatasetExample]:
    """Yield `DatasetExample` rows for a BeansPro split via `esp_data`.

    Parameters
    ----------
    subset
        BeansPro subset id. For BeansPro, this is the split name (e.g.
        `"crow-description"`, `"alarm-call-presence"`).
    split
        BeansPro split name passed to `esp_data.BeansPro(split=...)`.
    task_id
        Optional eval-task id stored on each yielded example.
    limit
        Optional maximum number of examples to yield.
    workers
        Number of parallel threads for GCS audio downloads.  ``1`` (default)
        downloads sequentially; values ``>1`` collect all metadata rows first
        then issue concurrent GCS downloads.

    Yields
    ------
    DatasetExample
        One normalized example at a time, in dataset order.
    """
    esp_mod = require_esp_data()
    diagnostics = os.environ.get(_DIAGNOSTICS_ENV, "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )
    log_every_n = max(1, _env_int(_LOG_EVERY_N_ENV, default=50))
    if workers == 1:
        workers = max(1, _env_int(_WORKERS_ENV, default=1))

    if workers > 1:
        yield from _iter_esp_data_beans_next_concurrent(
            esp_mod,
            subset=subset,
            split=split,
            task_id=task_id,
            limit=limit,
            workers=workers,
            diagnostics=diagnostics,
        )
        return

    kept = 0
    ordinal = 0
    last_heartbeat = time.monotonic()
    for raw in _load_beans_next_rows_via_reflection(esp_mod, split=split):
        if diagnostics and (kept == 0 or kept % log_every_n == 0):
            now = time.monotonic()
            dt = now - last_heartbeat
            last_heartbeat = now
            _LOG.info(
                "esp_data iter heartbeat dataset=beans_next subset=%s split=%s "
                "kept=%s ordinal=%s dt=%.2fs",
                subset,
                split,
                kept,
                ordinal,
                dt,
            )
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        stable_id = _resolve_row_id(row)
        sample_id = (
            stable_id
            if stable_id is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_next", subset=subset, split=split, ordinal=ordinal
            )
        )
        audio_path = _resolve_audio_for_row(
            row,
            sample_id=sample_id,
            subset=subset,
            split=split,
            diagnostics=diagnostics,
        )
        yield _build_dataset_example(
            row,
            sample_id=sample_id,
            audio_path=audio_path,
            split=split,
            task_id=task_id,
        )
        kept += 1
        ordinal += 1
        if limit is not None and kept >= limit:
            return


def _iter_esp_data_beans_next_concurrent(
    esp_mod: object,
    *,
    subset: str,
    split: str,
    task_id: str | None,
    limit: int | None,
    workers: int,
    diagnostics: bool,
) -> Iterator[DatasetExample]:
    """Concurrent GCS-download path for `iter_esp_data_beans_next_examples`.

    Yields
    ------
    DatasetExample
        Examples in dataset order.
    """
    raw_rows: list[tuple[int, str, Mapping[str, object]]] = []
    for ordinal, raw in enumerate(
        _load_beans_next_rows_via_reflection(
            esp_mod,  # type: ignore[arg-type]
            split=split,
        )
    ):
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        stable_id = _resolve_row_id(row)
        sample_id = (
            stable_id
            if stable_id is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_next", subset=subset, split=split, ordinal=ordinal
            )
        )
        raw_rows.append((ordinal, sample_id, row))
        if limit is not None and len(raw_rows) >= limit:
            break

    def _dl(item: tuple[int, str, Mapping[str, object]]) -> DatasetExample:
        _, sample_id, row = item
        audio_path = _resolve_audio_for_row(
            row,
            sample_id=sample_id,
            subset=subset,
            split=split,
            diagnostics=diagnostics,
        )
        return _build_dataset_example(
            row,
            sample_id=sample_id,
            audio_path=audio_path,
            split=split,
            task_id=task_id,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_dl, raw_rows)


# ---------------------------------------------------------------------------
# BirdSet (esp_data.BirdSet) loader
# ---------------------------------------------------------------------------

_BIRDSET_GCS_ROOT = "gs://esp-ml-datasets/birdset/v0.1.0/raw"


def _birdset_labels(row: Mapping[str, object]) -> list[str] | None:
    """Extract the BirdSet ground-truth label list from a metadata row.

    BirdSet recordings are multi-label: a single 5s clip may contain several
    focal species. ``canonical_name_multispecies`` is the authoritative
    ground-truth set — when populated, every species present in the clip is
    listed there. The other taxonomy fields (``species``,
    ``scientific_name_unified_original``) are single-species fallbacks used
    only when the multispecies field is missing on a given subset.

    Resolution order (first non-empty source wins, returned as a deduped list):
    1) ``canonical_name_multispecies`` (list/JSON-list of scientific names)
    2) ``species`` (single scientific name)
    3) ``scientific_name_unified_original`` (single scientific name)
    4) ``species_common`` (common-name fallback when no scientific names
       are present anywhere)

    Parameters
    ----------
    row
        Raw BirdSet metadata row.

    Returns
    -------
    list[str] or None
        Deduplicated list of label strings drawn from the first populated
        source, or ``None`` when no usable taxonomy information is present.
    """

    def _from_value(value: object) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _add(text: object) -> None:
            if not isinstance(text, str):
                return
            s = text.strip()
            if not s or s in seen:
                return
            seen.add(s)
            out.append(s)

        if value is None:
            return out
        if isinstance(value, list):
            for item in value:
                _add(item)
            return out
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return out
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, list):
                    for item in decoded:
                        _add(item)
                    return out
            _add(stripped)
        return out

    for key in (
        "canonical_name_multispecies",
        "species",
        "scientific_name_unified_original",
        "species_common",
    ):
        labels = _from_value(row.get(key))
        if labels:
            return labels

    return None


def _normalize_birdset_row(
    raw: dict[str, object], *, data_root: str
) -> dict[str, object]:
    """Normalise a BirdSet CSV row for the shared audio-resolution pipeline.

    BirdSet uses ``16khz_path`` / ``32khz_path`` / ``audio_path`` column names
    that differ from the BEANS-Zero / BEANS-Next convention.  This function adds
    aliased keys so ``_resolve_audio_for_row`` works without modification.

    It also injects ``output`` from BirdSet taxonomy fields so
    ``_labels_from_row`` picks up the multi-species ground-truth label
    list automatically. ``output`` is written as ``list[str]`` (the union
    of every species in the row) so downstream metric routing reaches the
    multi-reference any-of branch.

    Returns
    -------
    dict[str, object]
        Normalised copy of the row with aliased audio-path keys and
        ``output`` set to a list of label strings when extractable.
    """
    out: dict[str, object] = dict(raw)
    if data_root:
        out[_DATA_ROOT_KEY] = data_root
    for src, dst in (
        ("16khz_path", "audio_path_16KHz"),
        ("32khz_path", "audio_path_32KHz"),
        ("audio_path", "audio_path_original_sample_rate"),
    ):
        val = out.get(src)
        if isinstance(val, str) and val.strip():
            out.setdefault(dst, val)
    labels = _birdset_labels(out)
    if labels:
        existing = out.get("output")
        if existing is None:
            out["output"] = labels
        elif isinstance(existing, str) and not existing.strip():
            out["output"] = labels
        elif isinstance(existing, list) and not existing:
            out["output"] = labels
    return out


def _load_birdset_rows_via_reflection(
    esp_data: ModuleType,
    *,
    split: str,
) -> Iterator[Mapping[str, object]]:
    """Load BirdSet rows from esp_data using the ``BirdSet`` dataset class.

    Uses the polars backend fast-path (metadata-only, no audio I/O) when
    available, falling back to full iteration via ``__iter__``.

    Parameters
    ----------
    esp_data
        Imported ``esp_data`` module.
    split
        BirdSet split name, e.g. ``"HSN-test_5s"``.

    Yields
    ------
    Mapping[str, object]
        Normalised dataset rows ready for ``_resolve_audio_for_row``.

    Raises
    ------
    RuntimeError
        If ``esp_data.BirdSet`` is unavailable or the API does not match.
    """
    birdset_cls = getattr(esp_data, "BirdSet", None)
    if not callable(birdset_cls):
        raise RuntimeError(
            "esp_data.BirdSet is unavailable in the installed esp_data version. "
            "Fix: update esp_data, or use HuggingFace loading (`data_source: hf`)."
        )
    try:
        ds = birdset_cls(split=split)
    except TypeError as exc:
        raise RuntimeError(
            f"Unable to construct `esp_data.BirdSet(split={split!r})` (API mismatch). "
            "Fix: update this loader to match your esp_data version, or switch to "
            "HuggingFace loading (`data_source: hf`)."
        ) from exc

    data_root = str(getattr(ds, "data_root", f"{_BIRDSET_GCS_ROOT}/"))
    backend_df = getattr(getattr(ds, "_data", None), "_df", None)
    if backend_df is not None and hasattr(backend_df, "iter_rows"):
        for raw in backend_df.iter_rows(named=True):  # type: ignore[union-attr]
            yield _normalize_birdset_row(dict(raw), data_root=data_root)
        return

    if hasattr(ds, "__iter__"):
        for row in ds:  # type: ignore[misc]
            yield _normalize_birdset_row(dict(row), data_root=data_root)
        return

    raise RuntimeError(
        "Unable to iterate BirdSet rows from `esp_data.BirdSet`. "
        "Fix: update this loader to match your esp_data version, or switch to "
        "HuggingFace loading (`data_source: hf`)."
    )


def iter_esp_data_birdset_examples(
    *,
    subset: str,
    split: str,
    task_id: str | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> Iterator[DatasetExample]:
    """Yield ``DatasetExample`` rows for a BirdSet subset via ``esp_data``.

    Parameters
    ----------
    subset
        BirdSet subset id (e.g. ``"HSN-test_5s"``).  Used for sample-id
        synthesis and logging; must match the ``split`` argument.
    split
        BirdSet split name passed to ``esp_data.BirdSet(split=...)``.
    task_id
        Optional eval-task id stored on each yielded example.
    limit
        Optional maximum number of examples to yield.
    workers
        Number of parallel threads for GCS audio downloads.

    Yields
    ------
    DatasetExample
        One normalised example at a time, in dataset order.
    """
    esp_mod = require_esp_data()
    diagnostics = os.environ.get(_DIAGNOSTICS_ENV, "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )
    log_every_n = max(1, _env_int(_LOG_EVERY_N_ENV, default=50))
    if workers == 1:
        workers = max(1, _env_int(_WORKERS_ENV, default=1))

    if workers > 1:
        yield from _iter_esp_data_birdset_concurrent(
            esp_mod,
            subset=subset,
            split=split,
            task_id=task_id,
            limit=limit,
            workers=workers,
            diagnostics=diagnostics,
        )
        return

    kept = 0
    ordinal = 0
    last_heartbeat = time.monotonic()
    for raw in _load_birdset_rows_via_reflection(esp_mod, split=split):
        if diagnostics and (kept == 0 or kept % log_every_n == 0):
            now = time.monotonic()
            dt = now - last_heartbeat
            last_heartbeat = now
            _LOG.info(
                "esp_data iter heartbeat dataset=birdset subset=%s split=%s "
                "kept=%s ordinal=%s dt=%.2fs",
                subset,
                split,
                kept,
                ordinal,
                dt,
            )
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        sample_id = synthesize_esp_data_sample_id(
            dataset="birdset", subset=subset, split=split, ordinal=ordinal
        )
        audio_path = _resolve_audio_for_row(
            row,
            sample_id=sample_id,
            subset=subset,
            split=split,
            diagnostics=diagnostics,
        )
        yield _build_dataset_example(
            row,
            sample_id=sample_id,
            audio_path=audio_path,
            split=split,
            task_id=task_id,
        )
        kept += 1
        ordinal += 1
        if limit is not None and kept >= limit:
            return


def _iter_esp_data_birdset_concurrent(
    esp_mod: object,
    *,
    subset: str,
    split: str,
    task_id: str | None,
    limit: int | None,
    workers: int,
    diagnostics: bool,
) -> Iterator[DatasetExample]:
    """Concurrent GCS-download path for ``iter_esp_data_birdset_examples``.

    Yields
    ------
    DatasetExample
        Examples in dataset order.
    """
    raw_rows: list[tuple[int, str, Mapping[str, object]]] = []
    for ordinal, raw in enumerate(
        _load_birdset_rows_via_reflection(
            esp_mod,  # type: ignore[arg-type]
            split=split,
        )
    ):
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        sample_id = synthesize_esp_data_sample_id(
            dataset="birdset", subset=subset, split=split, ordinal=ordinal
        )
        raw_rows.append((ordinal, sample_id, row))
        if limit is not None and len(raw_rows) >= limit:
            break

    def _dl(item: tuple[int, str, Mapping[str, object]]) -> DatasetExample:
        _, sample_id, row = item
        audio_path = _resolve_audio_for_row(
            row,
            sample_id=sample_id,
            subset=subset,
            split=split,
            diagnostics=diagnostics,
        )
        return _build_dataset_example(
            row,
            sample_id=sample_id,
            audio_path=audio_path,
            split=split,
            task_id=task_id,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_dl, raw_rows)


# ---------------------------------------------------------------------------
# BeansProMultiAudio (esp_data.BeansProMultiAudio) loader
# ---------------------------------------------------------------------------

_MULTIAUDIO_GCS_BASE = "gs://esp-data-ingestion/beans-pro/v0.1.0/raw"
_MULTIAUDIO_GIBBON_BASE = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw"

_MULTIAUDIO_SPLIT_JSONL_AND_ROOT: dict[str, tuple[str, str]] = {
    "gibbon-fewshot-detection": (
        f"{_MULTIAUDIO_GCS_BASE}/gibbon_fewshot_detection/test.jsonl",
        f"{_MULTIAUDIO_GIBBON_BASE}/",
    ),
    "gibbon-fewshot-detection-balanced": (
        f"{_MULTIAUDIO_GCS_BASE}/gibbon_fewshot_detection_balanced/test.jsonl",
        f"{_MULTIAUDIO_GIBBON_BASE}/",
    ),
    "giant-otter-4way": (
        f"{_MULTIAUDIO_GCS_BASE}/giant_otter_4way/test.jsonl",
        f"{_MULTIAUDIO_GCS_BASE}/",
    ),
    "dcase-fewshot-detection-balanced": (
        f"{_MULTIAUDIO_GCS_BASE}/dcase_fewshot_detection_balanced/test.jsonl",
        f"{_MULTIAUDIO_GIBBON_BASE}/",
    ),
    "crow-4way": (
        f"{_MULTIAUDIO_GCS_BASE}/crow_4way/test.jsonl",
        f"{_MULTIAUDIO_GCS_BASE}/carrion_crow_descriptions/",
    ),
    "zebra-4way": (
        f"{_MULTIAUDIO_GCS_BASE}/zebra_4way/test.jsonl",
        f"{_MULTIAUDIO_GCS_BASE}/zebra_descriptions/",
    ),
    # unseen-species audio paths contain xeno-canto/... or inaturalist/...
    # relative to the esp-data-ingestion bucket root.
    "unseen-species-4way": (
        f"{_MULTIAUDIO_GCS_BASE}/unseen_species_4way/test.jsonl",
        "gs://esp-data-ingestion/",
    ),
    "unseen-species-4way-hard": (
        f"{_MULTIAUDIO_GCS_BASE}/unseen_species_4way_hard/test.jsonl",
        "gs://esp-data-ingestion/",
    ),
    "unseen-genus-4way": (
        f"{_MULTIAUDIO_GCS_BASE}/unseen_genus_4way/test.jsonl",
        f"{_MULTIAUDIO_GCS_BASE}/",
    ),
    "unseen-genus-4way-hard": (
        f"{_MULTIAUDIO_GCS_BASE}/unseen_genus_4way_hard/test.jsonl",
        f"{_MULTIAUDIO_GCS_BASE}/",
    ),
    "unseen-family-4way": (
        f"{_MULTIAUDIO_GCS_BASE}/unseen_family_4way/test.jsonl",
        f"{_MULTIAUDIO_GCS_BASE}/",
    ),
    "unseen-family-4way-hard": (
        f"{_MULTIAUDIO_GCS_BASE}/unseen_family_4way_hard/test.jsonl",
        f"{_MULTIAUDIO_GCS_BASE}/",
    ),
}

_AUDIO_PLACEHOLDER_TAG = "<Audio><AudioHere></Audio>"


def _load_beans_next_multiaudio_rows_via_reflection(
    esp_data: ModuleType,
    *,
    split: str,
) -> Iterator[Mapping[str, object]]:
    """Load BeansProMultiAudio rows from esp_data, bypassing ``_process()``.

    Mirrors the BeansPro fast-path: iterates the polars backend for metadata-only
    rows and injects ``_DATA_ROOT_KEY`` so audio resolution uses bounded GCS
    downloads rather than per-row ``_process()`` calls.

    Parameters
    ----------
    esp_data
        Imported ``esp_data`` module.
    split
        BeansProMultiAudio split name (e.g. ``"crow-4way"``).

    Yields
    ------
    Mapping[str, object]
        Raw dataset rows with ``_DATA_ROOT_KEY`` injected.

    Raises
    ------
    RuntimeError
        If ``esp_data.BeansProMultiAudio`` is unavailable or the API does not
        match expected access patterns.
    """
    cls = getattr(esp_data, "BeansProMultiAudio", None)
    if not callable(cls):
        yield from _load_beans_next_multiaudio_rows_from_gcs_jsonl(split=split)
        return

    try:
        ds = cls(split=split)
    except TypeError as exc:
        raise RuntimeError(
            f"Unable to construct `esp_data.BeansProMultiAudio(split={split!r})`. "
            "Fix: update this loader or switch to HuggingFace loading."
        ) from exc

    data_root = str(getattr(ds, "data_root", ""))
    backend_df = getattr(getattr(ds, "_data", None), "_df", None)
    if backend_df is not None and hasattr(backend_df, "iter_rows"):
        for raw in backend_df.iter_rows(named=True):  # type: ignore[union-attr]
            out: dict[str, object] = dict(raw)
            if data_root:
                out[_DATA_ROOT_KEY] = data_root
            yield out
        return

    if hasattr(ds, "__iter__"):
        yield from ds  # type: ignore[misc]
        return

    raise RuntimeError(
        "Unable to iterate BeansProMultiAudio rows from `esp_data.BeansProMultiAudio`. "
        "Fix: update this loader to match your esp_data version."
    )


def _load_beans_next_multiaudio_rows_from_gcs_jsonl(
    *, split: str
) -> Iterator[Mapping[str, object]]:
    """Load BeansProMultiAudio rows by streaming GCS JSONL metadata.

    Compatibility fallback for environments where the installed ``esp_data``
    package does not yet ship ``BeansProMultiAudio``.

    Parameters
    ----------
    split
        BeansProMultiAudio split name.

    Yields
    ------
    Mapping[str, object]
        JSON-decoded row dicts with ``_DATA_ROOT_KEY`` injected.

    Raises
    ------
    RuntimeError
        If the split is unknown or the JSONL cannot be read/parsed.
    """
    import json

    try:
        import fsspec  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "BeansProMultiAudio fallback loading requires `fsspec` (and `gcsfs` for "
            "`gs://`). Install them or use a different data source."
        ) from exc

    cfg = _MULTIAUDIO_SPLIT_JSONL_AND_ROOT.get(split)
    if cfg is None:
        known = ", ".join(sorted(_MULTIAUDIO_SPLIT_JSONL_AND_ROOT))
        raise RuntimeError(
            f"Unknown BeansProMultiAudio split {split!r}. Known: {known}."
        )
    jsonl_path, data_root = cfg

    try:
        with fsspec.open(jsonl_path, "rt") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                if not isinstance(row, dict):
                    continue
                row[_DATA_ROOT_KEY] = data_root
                yield row
    except Exception as exc:
        raise RuntimeError(
            f"Failed to stream BeansProMultiAudio JSONL: split={split!r} "
            f"path={jsonl_path!r}. Error: {exc}"
        ) from exc


def _resolve_audio_paths_for_row(
    row: Mapping[str, object],
    *,
    sample_id: str,
    diagnostics: bool,
) -> list[str]:
    """Resolve local WAV paths for all ``audio_paths`` entries in a multi-audio row.

    Downloads each path from GCS using a bounded daemon-thread timeout (same
    mechanism as ``_resolve_audio_for_row``).

    Parameters
    ----------
    row
        Raw metadata row containing ``audio_paths`` (list of relative paths) and
        optionally ``_DATA_ROOT_KEY`` for the GCS root prefix.
    sample_id
        Stable sample identifier; used to name per-audio cache files.
    diagnostics
        When ``True``, emit WARNING logs on timeout or download failure.

    Returns
    -------
    list[str]
        Resolved local WAV paths (one per entry in ``audio_paths``).  Entries that
        fail to download are skipped (warning logged when ``diagnostics=True``).
    """
    data_root = row.get(_DATA_ROOT_KEY)
    audio_paths_raw = row.get("audio_paths")
    if not isinstance(audio_paths_raw, list) or not audio_paths_raw:
        return []

    dl_timeout_raw = _env_int(_AUDIO_TIMEOUT_S_ENV, default=60)
    dl_timeout: float | None = float(dl_timeout_raw) if dl_timeout_raw > 0 else None

    resolved: list[str] = []
    for i, rel_path in enumerate(audio_paths_raw):
        if not isinstance(rel_path, str) or not rel_path.strip():
            if diagnostics:
                _LOG.warning(
                    "BeansProMultiAudio audio_paths[%d] is not a string; "
                    "skipping sample_id=%s",
                    i,
                    sample_id,
                )
            continue
        rel = rel_path.strip()
        if isinstance(data_root, str) and data_root:
            gcs_abs = data_root.rstrip("/") + "/" + rel.lstrip("/")
            path = _download_gcs_to_wav(
                gcs_abs,
                sample_id=f"{sample_id}__audio{i}",
                timeout_s=dl_timeout,
                diagnostics=diagnostics,
            )
            if path is None:
                if diagnostics:
                    _LOG.warning(
                        "BeansProMultiAudio GCS download failed for audio[%d] "
                        "sample_id=%s path=%s",
                        i,
                        sample_id,
                        gcs_abs,
                    )
                continue
            resolved.append(path)
        elif os.path.isabs(rel):
            resolved.append(rel)
        else:
            if diagnostics:
                _LOG.warning(
                    "BeansProMultiAudio audio_paths[%d] is relative and no data_root "
                    "is set; skipping sample_id=%s path=%s",
                    i,
                    sample_id,
                    rel,
                )
    return resolved


def _strip_audio_placeholders_except_last(conversation: str) -> str:
    """Replace all but the last ``<Audio><AudioHere></Audio>`` with ``[audio]``.

    Used to build a single-audio reformulation of multi-audio prompts for
    launchers that support only one audio input (e.g. NatureLM v1.1).

    Parameters
    ----------
    conversation
        User message text containing one or more ``<Audio><AudioHere></Audio>``
        placeholders.

    Returns
    -------
    str
        Modified conversation with all but the last placeholder replaced by
        ``[audio]``.
    """
    tag = _AUDIO_PLACEHOLDER_TAG
    idx = conversation.rfind(tag)
    if idx == -1:
        return conversation
    prefix = conversation[:idx].replace(tag, "[audio]")
    return prefix + conversation[idx:]


def _build_multiaudio_dataset_example(
    row: Mapping[str, object],
    *,
    sample_id: str,
    audio_paths: list[str],
    query_audio_path: str | None,
    split: str,
    task_id: str | None,
) -> DatasetExample:
    """Assemble a ``DatasetExample`` from a BeansProMultiAudio row.

    Parameters
    ----------
    row
        Raw metadata row from the dataset loader.
    sample_id
        Stable sample identifier.
    audio_paths
        Resolved local WAV paths for all ``audio_paths`` entries.
    query_audio_path
        Resolved local WAV path for the query audio
        (``audio_path_original_sample_rate``), or ``None`` when unavailable.
        Stored as ``metadata["audio_path"]`` for single-audio prompt specs.
    split
        Dataset split stored on the example.
    task_id
        Optional eval-task id stored on the example.

    Returns
    -------
    DatasetExample
        Fully assembled example ready for the runner.
    """
    meta: dict[str, object] = {}

    if audio_paths:
        meta["audio_paths"] = audio_paths
        meta["n_audios"] = len(audio_paths)

    effective_query = query_audio_path or (audio_paths[-1] if audio_paths else None)
    if effective_query:
        meta["audio_path"] = effective_query

    conversation = ""
    labels: str | None = None
    messages_raw = row.get("messages")
    if isinstance(messages_raw, list):
        for msg in messages_raw:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and isinstance(content, str):
                conversation = content
            elif role == "assistant" and isinstance(content, str):
                labels = content.strip() or None

    if conversation:
        meta["conversation"] = conversation
        meta["conversation_query_only"] = _strip_audio_placeholders_except_last(
            conversation
        )

    for key in (
        "task",
        "dataset_name",
        "source_dataset",
        "license",
        "template_path",
    ):
        val = row.get(key)
        if isinstance(val, str | int | float | bool):
            meta[key] = val

    return DatasetExample(
        sample_id=sample_id,
        task_id=task_id,
        split=split,
        labels=labels,
        metadata=meta,
    )


def iter_esp_data_beans_next_multiaudio_examples(
    *,
    split: str,
    task_id: str | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> Iterator[DatasetExample]:
    """Yield ``DatasetExample`` rows for a BeansProMultiAudio split via ``esp_data``.

    Each example contains:

    - ``metadata["audio_paths"]``: list of resolved local WAV paths (all N audios).
    - ``metadata["audio_path"]``: resolved path for the query audio (single-audio
      fallback for NatureLM v1.1).
    - ``metadata["conversation"]``: full user message with N ``<AudioHere>`` tags.
    - ``metadata["conversation_query_only"]``: reformulated message with only the
      last ``<AudioHere>`` tag (for single-audio launchers).
    - ``labels``: gold answer from the assistant message (e.g. ``"A"``, ``"None"``).

    Parameters
    ----------
    split
        BeansProMultiAudio split name (e.g. ``"crow-4way"``).
    task_id
        Optional eval-task id stored on each yielded example.
    limit
        Optional maximum number of examples to yield.
    workers
        Number of parallel threads for GCS audio downloads.  ``1`` downloads
        sequentially; values ``>1`` collect all metadata rows first then issue
        concurrent GCS downloads, significantly reducing wall-clock time.

    Yields
    ------
    DatasetExample
        One normalized example at a time, in dataset order.
    """
    esp_mod = require_esp_data()
    diagnostics = os.environ.get(_DIAGNOSTICS_ENV, "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )
    log_every_n = max(1, _env_int(_LOG_EVERY_N_ENV, default=50))
    if workers == 1:
        workers = max(1, _env_int(_WORKERS_ENV, default=1))

    if workers > 1:
        yield from _iter_esp_data_beans_next_multiaudio_concurrent(
            esp_mod,
            split=split,
            task_id=task_id,
            limit=limit,
            workers=workers,
            diagnostics=diagnostics,
        )
        return

    kept = 0
    ordinal = 0
    last_heartbeat = time.monotonic()
    for raw in _load_beans_next_multiaudio_rows_via_reflection(esp_mod, split=split):
        if diagnostics and (kept == 0 or kept % log_every_n == 0):
            now = time.monotonic()
            dt = now - last_heartbeat
            last_heartbeat = now
            _LOG.info(
                "esp_data iter heartbeat dataset=beans_next_multiaudio split=%s "
                "kept=%s ordinal=%s dt=%.2fs",
                split,
                kept,
                ordinal,
                dt,
            )
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        stable_id = _resolve_row_id(row)
        sample_id = (
            stable_id
            if stable_id is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_next_multiaudio",
                subset=split,
                split=split,
                ordinal=ordinal,
            )
        )
        audio_paths = _resolve_audio_paths_for_row(
            row, sample_id=sample_id, diagnostics=diagnostics
        )
        query_audio_path = _resolve_audio_for_row(
            row,
            sample_id=f"{sample_id}__query",
            subset=split,
            split=split,
            diagnostics=diagnostics,
        )
        yield _build_multiaudio_dataset_example(
            row,
            sample_id=sample_id,
            audio_paths=audio_paths,
            query_audio_path=query_audio_path,
            split=split,
            task_id=task_id,
        )
        kept += 1
        ordinal += 1
        if limit is not None and kept >= limit:
            return


def _iter_esp_data_beans_next_multiaudio_concurrent(
    esp_mod: object,
    *,
    split: str,
    task_id: str | None,
    limit: int | None,
    workers: int,
    diagnostics: bool,
) -> Iterator[DatasetExample]:
    """Concurrent GCS-download path for ``iter_esp_data_beans_next_multiaudio_examples``.

    Collects all metadata rows first (polars iteration, no I/O), then issues
    GCS downloads concurrently via a ``ThreadPoolExecutor``.

    Yields
    ------
    DatasetExample
        Examples in dataset order.
    """
    raw_rows: list[tuple[int, str, Mapping[str, object]]] = []
    for ordinal, raw in enumerate(
        _load_beans_next_multiaudio_rows_via_reflection(
            esp_mod,  # type: ignore[arg-type]
            split=split,
        )
    ):
        row: Mapping[str, object] = dict(raw) if not isinstance(raw, Mapping) else raw
        stable_id = _resolve_row_id(row)
        sample_id = (
            stable_id
            if stable_id is not None
            else synthesize_esp_data_sample_id(
                dataset="beans_next_multiaudio",
                subset=split,
                split=split,
                ordinal=ordinal,
            )
        )
        raw_rows.append((ordinal, sample_id, row))
        if limit is not None and len(raw_rows) >= limit:
            break

    def _dl(item: tuple[int, str, Mapping[str, object]]) -> DatasetExample:
        _, sample_id, row = item
        audio_paths = _resolve_audio_paths_for_row(
            row, sample_id=sample_id, diagnostics=diagnostics
        )
        query_audio_path = _resolve_audio_for_row(
            row,
            sample_id=f"{sample_id}__query",
            subset=split,
            split=split,
            diagnostics=diagnostics,
        )
        return _build_multiaudio_dataset_example(
            row,
            sample_id=sample_id,
            audio_paths=audio_paths,
            query_audio_path=query_audio_path,
            split=split,
            task_id=task_id,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_dl, raw_rows)
