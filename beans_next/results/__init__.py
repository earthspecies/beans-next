"""Benchmark result artifacts and storage helpers."""

from beans_next.results.store import BenchmarkArtifactWriter, dumps_canonical

try:
    from beans_next.results.gcs_upload import upload_run_artifacts
except ModuleNotFoundError:  # pragma: no cover
    def upload_run_artifacts(*args: object, **kwargs: object) -> list[str]:
        """Upload benchmark run artifacts to GCS.

        This helper requires the optional ``beans_next.results.gcs_upload`` module.
        If you're running from a minimal install, install the GCS extra:
        ``pip install 'beans-next[gcs]'``.
        """

        raise ImportError(
            "GCS upload support is unavailable (missing beans_next.results.gcs_upload)."
        )

__all__ = [
    "BenchmarkArtifactWriter",
    "dumps_canonical",
    "upload_run_artifacts",
]
