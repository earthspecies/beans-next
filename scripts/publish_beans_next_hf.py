"""Build a BEANS-Next Hugging Face bundle from BEANS-Next eval manifests.

The published layout (``EarthSpeciesProject/BEANS-Next``) is:

- ``test/metadata.parquet`` -- one row per evaluation sample.
- ``test/audio/<xx>/<sha256>.wav`` -- content-addressed audio, where ``<sha256>``
  is the SHA-256 of the **published WAV bytes** and ``<xx>`` is its first two
  hex characters.

Conventions were recovered from the existing release rather than assumed:

- ``audio_id`` is the SHA-256 of the final WAV file, and the file name repeats it.
- A ``.wav`` source is copied **verbatim**; its SHA is the SHA of the source
  bytes. (Verified against a published row: identical SHA and byte length.)
- A non-WAV source is decoded to mono PCM_16 WAV at its **native** sample rate;
  there is no fixed target rate in the existing release.
- The parquet ``task`` column carries the manifest's ``dataset_name``, not the
  manifest's own ``task`` field (which holds a task-family string such as
  ``call_type_presence_binary``).
- ``audio_path_original_sample_rate`` is provenance only -- loading goes via
  ``file_name`` -- so the manifest value is preserved verbatim even though the
  existing release is inconsistent about whether it is absolute or relative.

This script only ever writes to a local staging directory. Uploading is a
separate, explicit step (``huggingface-cli upload-large-folder``), so a build
can be inspected before anything becomes public.

Examples
--------
Stage a bundle merging new manifests over an existing metadata table::

    uv run python scripts/publish_beans_next_hf.py \\
        --plan plan.json \\
        --base-metadata hf_metadata.parquet \\
        --out-dir /scratch/$USER/beans_next_hf

`plan.json` is a list of ``{"manifest": path, "task": str, "tier": int}``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_AUDIO_KEY = "audio_path_original_sample_rate"
_DEFAULT_BUCKET = "esp-data-ingestion"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _split_uri(uri: str) -> tuple[str, str]:
    """Split a manifest audio reference into ``(bucket, object)``.

    Parameters
    ----------
    uri
        Either a ``gs://bucket/object`` URI or a path relative to the default
        ingestion bucket.

    Returns
    -------
    tuple of (str, str)
        Bucket name and object name.
    """
    if uri.startswith("gs://"):
        bucket, _, obj = uri[len("gs://") :].partition("/")
        return bucket, obj
    return _DEFAULT_BUCKET, uri


def _access_token() -> str:
    """Return a GCS access token from application default credentials.

    Returns
    -------
    str
        OAuth access token.

    Raises
    ------
    RuntimeError
        If no token could be obtained.
    """
    try:
        out = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        msg = "could not obtain a GCS access token via gcloud ADC"
        raise RuntimeError(msg) from exc
    token = out.stdout.strip()
    if not token:
        msg = "gcloud returned an empty access token"
        raise RuntimeError(msg)
    return token


def _download(bucket: str, obj: str, token: str) -> bytes | None:
    """Fetch one GCS object.

    Parameters
    ----------
    bucket
        Bucket name.
    obj
        Object name.
    token
        OAuth access token.

    Returns
    -------
    bytes or None
        Object bytes, or `None` when the object does not exist.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    quoted = urllib.parse.quote(obj, safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quoted}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 403):
                return None
            if attempt == 3:
                return None
        except Exception:  # noqa: BLE001 - transient network errors are retried
            if attempt == 3:
                return None
    return None


def _to_published_wav(raw: bytes, source_name: str) -> bytes:
    """Return the WAV bytes to publish for one source file.

    A `.wav` source is returned unchanged so its content hash matches the
    source exactly. Any other container is decoded to mono PCM_16 WAV at its
    native sample rate.

    Parameters
    ----------
    raw
        Source file bytes.
    source_name
        Source file name, used only for its extension.

    Returns
    -------
    bytes
        WAV bytes to publish.

    Raises
    ------
    ValueError
        If the source could not be decoded, propagated from the ffmpeg
        fallback.
    """  # noqa: DOC502 - ValueError reaches callers via _ffmpeg_to_wav
    if source_name.lower().endswith(".wav"):
        return raw
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - fall back to ffmpeg below
        return _ffmpeg_to_wav(raw, source_name)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    buf = io.BytesIO()
    sf.write(buf, np.asarray(mono, dtype="float32"), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _ffmpeg_to_wav(raw: bytes, source_name: str) -> bytes:
    """Decode a source libsndfile cannot read, using a bundled ffmpeg.

    Some files in the corpus are mislabelled containers -- AAC streams carrying
    a `.mp3` extension -- and others are 320 kbps MP3 or ID3v2.4-tagged files
    that libsndfile rejects. ffmpeg decodes all of them. This matters beyond
    tidiness: for at least one of these files libsndfile does not always fail
    loudly, it returns a fraction of a second of audio from a 585 KB file, so
    falling back is safer than trusting a short read.

    Parameters
    ----------
    raw
        Source file bytes.
    source_name
        Source file name, used for error messages and the temp suffix.

    Returns
    -------
    bytes
        Mono PCM_16 WAV bytes at the source's native sample rate.

    Raises
    ------
    ValueError
        If ffmpeg is unavailable or could not decode the source.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - reported as ValueError
        msg = f"could not decode {source_name!r} (no ffmpeg available)"
        raise ValueError(msg) from exc

    suffix = Path(source_name).suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="bn-hf-dec-") as tmpdir:
        src = Path(tmpdir) / f"in{suffix}"
        dst = Path(tmpdir) / "out.wav"
        src.write_bytes(raw)
        proc = subprocess.run(
            [exe, "-v", "error", "-i", str(src), "-ac", "1",
             "-c:a", "pcm_s16le", "-f", "wav", str(dst), "-y"],
            capture_output=True,
        )
        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
            detail = proc.stderr.decode("utf-8", "replace").strip()[:120]
            msg = f"could not decode {source_name!r} via ffmpeg: {detail}"
            raise ValueError(msg)
        return dst.read_bytes()


def _stage_one(
    uri: str, out_dir: Path, token: str
) -> tuple[str, str | None, str | None]:
    """Download, convert and store one audio file.

    Parameters
    ----------
    uri
        Manifest audio reference.
    out_dir
        Bundle root; audio is written under ``<out_dir>/test/audio``.
    token
        OAuth access token.

    Returns
    -------
    tuple of (str, str or None, str or None)
        The input `uri`, its content SHA-256 (or `None` on failure), and an
        error string (or `None` on success).
    """
    bucket, obj = _split_uri(uri)
    raw = _download(bucket, obj, token)
    if raw is None:
        return uri, None, "missing"
    try:
        wav = _to_published_wav(raw, obj)
    except ValueError as exc:
        return uri, None, str(exc)
    sha = hashlib.sha256(wav).hexdigest()
    dest = out_dir / "test" / "audio" / sha[:2] / f"{sha}.wav"
    if not dest.exists():
        # Distinct source files can decode to identical bytes and therefore
        # share a hash, so several workers may stage the same destination at
        # once. Give each attempt its own temp name: with a shared one the
        # first `os.replace` moves the file out from under the others, which
        # then die with FileNotFoundError and take the whole pool down.
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{sha}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(wav)
            os.replace(tmp, dest)  # atomic; content is identical either way
        finally:
            if tmp.exists():
                tmp.unlink()
    return uri, sha, None


def stage_audio(
    uris: Iterable[str], out_dir: Path, workers: int = 32
) -> tuple[dict[str, str], dict[str, str]]:
    """Stage every unique audio file and return its content hash.

    Parameters
    ----------
    uris
        Manifest audio references (duplicates are collapsed).
    out_dir
        Bundle root.
    workers
        Number of concurrent downloads.

    Returns
    -------
    tuple of (dict, dict)
        Mapping from URI to SHA-256 for successes, and URI to error for
        failures.
    """
    token = _access_token()
    todo = sorted(set(uris))
    sha_by_uri: dict[str, str] = {}
    errors: dict[str, str] = {}

    # Reuse a previous run's map so rebuilding the metadata does not re-download
    # tens of gigabytes. Entries are only trusted while their file is still on
    # disk, so a pruned bundle re-stages rather than emitting dangling rows.
    cache_path = out_dir / "staging_sha_map.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cached = {}
        for uri, sha in cached.items():
            if (out_dir / "test" / "audio" / sha[:2] / f"{sha}.wav").exists():
                sha_by_uri[uri] = sha
        if sha_by_uri:
            _log(f"reusing {len(sha_by_uri)} cached hashes from {cache_path.name}")
        todo = [u for u in todo if u not in sha_by_uri]

    _log(f"staging {len(todo)} audio files with {workers} workers")
    done = 0

    def _safe(uri: str) -> tuple[str, str | None, str | None]:
        """Stage one file, converting any unexpected error into a result row.

        A raised exception inside `ThreadPoolExecutor.map` aborts the whole
        run, discarding work already done, so failures are reported per file.

        Returns
        -------
        tuple of (str, str or None, str or None)
            As :func:`_stage_one`, with unexpected errors in the third slot.
        """
        try:
            return _stage_one(uri, out_dir, token)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            return uri, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for uri, sha, err in pool.map(_safe, todo):
            done += 1
            if sha is not None:
                sha_by_uri[uri] = sha
            else:
                errors[uri] = err or "unknown"
            if done % 500 == 0:
                _log(f"  {done}/{len(todo)} staged ({len(errors)} failed)")
    _log(f"staged {len(sha_by_uri)}; {len(errors)} failed")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(sha_by_uri))
    return sha_by_uri, errors


_LICENSE_MAP: dict[str, str] = {
    "https://creativecommons.org/licenses/by-nc-sa/4.0/": "CC-BY-NC-SA-4.0",
    "https://creativecommons.org/licenses/by-nc/4.0/": "CC-BY-NC-4.0",
    "https://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
    "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc0-1.0": "CC0-1.0",
}


def normalize_license(value: object) -> object:
    """Map a licence string onto the release's SPDX-style short form.

    Manifests carry a mix of bare URLs and lower-case identifiers; the
    published table predominantly uses upper-case SPDX-like codes such as
    `CC-BY-NC-4.0`. Unrecognised values are passed through untouched rather
    than guessed at, since licence metadata should never be invented.

    Parameters
    ----------
    value
        Raw licence value from a manifest row.

    Returns
    -------
    object
        The normalised licence string, or `value` unchanged.
    """
    if not isinstance(value, str):
        return value
    key = value.strip()
    mapped = _LICENSE_MAP.get(key) or _LICENSE_MAP.get(key.lower())
    if mapped:
        return mapped
    if key.rstrip("/") + "/" in _LICENSE_MAP:
        return _LICENSE_MAP[key.rstrip("/") + "/"]
    return value


def rows_from_manifest(
    manifest: Path, task: str, tier: int, sha_by_uri: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Convert one manifest into published metadata rows.

    Rows whose audio failed to stage are dropped rather than published with a
    dangling reference.

    Parameters
    ----------
    manifest
        Path to a JSONL manifest.
    task
        Published `task` value (also used for `dataset_name`).
    tier
        Published tier (1-4).
    sha_by_uri
        Mapping from manifest audio reference to content SHA-256.

    Returns
    -------
    list of dict
        Published rows.
    """
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        src = json.loads(line)
        uri = src.get(_AUDIO_KEY)
        sha = sha_by_uri.get(uri) if isinstance(uri, str) else None
        if sha is None:
            continue
        meta = src.get("metadata")
        rows.append(
            {
                "audio_id": sha,
                "file_name": f"audio/{sha[:2]}/{sha}.wav",
                _AUDIO_KEY: uri,
                "task": task,
                "dataset_name": task,
                "tier": tier,
                "id": src.get("id"),
                "sample_id": src.get("id"),
                "instruction": src.get("instruction"),
                "instruction_text": src.get("instruction_text"),
                "output": src.get("output"),
                "label": src.get("output"),
                "license": normalize_license(src.get("license")),
                "source_dataset": src.get("source_dataset"),
                "metadata": meta
                if isinstance(meta, str) or meta is None
                else json.dumps(meta),
            }
        )
    return rows


def build_metadata(
    base_parquet: Path, new_rows: list[dict[str, Any]], replaced: set[str], out: Path
) -> tuple[int, int]:
    """Merge new rows into the existing metadata table.

    Parameters
    ----------
    base_parquet
        Existing published ``metadata.parquet``.
    new_rows
        Rows produced by :func:`rows_from_manifest`.
    replaced
        Task names whose existing rows are dropped before appending.
    out
        Destination parquet path.

    Returns
    -------
    tuple of (int, int)
        Number of rows kept from the base table and number appended.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = pq.read_table(base_parquet)
    schema = base.schema
    mask = [t not in replaced for t in base.column("task").to_pylist()]
    kept = base.filter(pa.array(mask))

    columns = {}
    for field in schema:
        values = [r.get(field.name) for r in new_rows]
        columns[field.name] = pa.array(values, type=field.type)
    added = pa.Table.from_pydict(columns, schema=schema)

    merged = pa.concat_tables([kept, added])
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, out)
    return kept.num_rows, added.num_rows


def main(argv: list[str] | None = None) -> int:
    """Run the bundle builder.

    Parameters
    ----------
    argv
        Command-line arguments; defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--base-metadata", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument(
        "--audio-only", action="store_true", help="stage audio, skip the parquet"
    )
    args = ap.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    manifests = [(Path(p["manifest"]), p["task"], int(p["tier"])) for p in plan]
    _log(f"plan: {len(manifests)} manifests")

    uris: list[str] = []
    for path, _, _ in manifests:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            v = json.loads(line).get(_AUDIO_KEY)
            if isinstance(v, str) and v:
                uris.append(v)

    sha_by_uri, errors = stage_audio(uris, args.out_dir, workers=args.workers)
    if errors:
        err_path = args.out_dir / "staging_errors.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(errors, indent=2))
        _log(f"wrote {len(errors)} staging errors to {err_path}")
    if args.audio_only:
        return 0

    new_rows: list[dict[str, Any]] = []
    replaced: set[str] = set()
    for path, task, tier in manifests:
        rows = rows_from_manifest(path, task, tier, sha_by_uri)
        _log(f"  {task:28s} tier={tier}  rows={len(rows)}")
        new_rows.extend(rows)
        replaced.add(task)

    out_parquet = args.out_dir / "test" / "metadata.parquet"
    kept, added = build_metadata(
        args.base_metadata, new_rows, replaced, out_parquet
    )
    _log(f"metadata: kept {kept} existing rows, appended {added} -> {out_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
