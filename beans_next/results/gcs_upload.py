"""GCS upload helper for benchmark run artifacts."""

# ruff: noqa: DOC501, DOC502

from __future__ import annotations

import logging
from pathlib import Path

__all__ = ["upload_run_artifacts"]

_LOG = logging.getLogger(__name__)

_GCS_URI_PREFIX = "gs://"

_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "predictions.jsonl",
    "processed_predictions.jsonl",
    "scored_predictions.jsonl",
    "summary.json",
    "model_identity.json",
    "checkpoint.json",
    "judge_outputs.jsonl",
    "judge_scored_predictions.jsonl",
    "judge_summary.json",
)


def upload_run_artifacts(output_dir: Path, gcs_prefix: str) -> list[str]:
    """Upload benchmark run artifacts from ``output_dir`` to GCS.

    Uploads whichever of the standard artifact files exist in ``output_dir``.
    Silently skips absent files.

    Parameters
    ----------
    output_dir
        Local directory containing the benchmark artifacts.
    gcs_prefix
        GCS destination prefix of the form ``gs://<bucket>/<path>``.
        A trailing slash is stripped automatically.

    Returns
    -------
    list[str]
        GCS URIs of every file successfully uploaded, in iteration order.

    Raises
    ------
    ImportError
        When ``google-cloud-storage`` is not installed.
    ValueError
        When ``gcs_prefix`` is not a valid ``gs://`` URI.

    Notes
    -----
    Standard artifact files attempted: ``predictions.jsonl``,
    ``processed_predictions.jsonl``, ``scored_predictions.jsonl``,
    ``summary.json``, ``model_identity.json``, ``checkpoint.json``,
    ``judge_outputs.jsonl``, ``judge_scored_predictions.jsonl``,
    ``judge_summary.json``.

    This function only looks at **files in the single directory** ``output_dir``.
    It does **not** walk nested ``suite/<suite_id>/<task_id>/`` trees. For full
    suite/benchmark runs, use the same approach as ``examples/slurm/run_inference.sh``:
    ``gsutil -m rsync -r <local_run_dir>/ <gs://.../run_prefix/>`` (after copying
    out of node-local scratch to a stable path if needed). Rescoring from GCS
    requires the per-task ``predictions.jsonl`` / ``processed_predictions.jsonl``,
    not only a top-level ``summary.json``.
    """
    try:
        from google.cloud import storage as gcs  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for GCS uploads. "
            "Install it with: pip install google-cloud-storage"
        ) from exc

    if not gcs_prefix.startswith(_GCS_URI_PREFIX):
        raise ValueError(f"gcs_prefix must start with 'gs://'; got {gcs_prefix!r}")

    path_part = gcs_prefix[len(_GCS_URI_PREFIX) :].rstrip("/")
    if "/" in path_part:
        bucket_name, blob_prefix = path_part.split("/", 1)
    else:
        bucket_name = path_part
        blob_prefix = ""

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    uploaded: list[str] = []

    for filename in _ARTIFACT_FILENAMES:
        local_path = output_dir / filename
        if not local_path.is_file():
            continue
        blob_name = f"{blob_prefix}/{filename}" if blob_prefix else filename
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        uri = f"gs://{bucket_name}/{blob_name}"
        _LOG.info("Uploaded %s → %s", local_path, uri)
        uploaded.append(uri)

    return uploaded
