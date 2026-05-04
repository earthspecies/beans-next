"""Two-layer benchmark cache: inference (wire) + scoring (metrics).

Both layers use separate SQLite files under a shared directory. Keys are
content hashes so the cache remains portable across machines when inputs match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from beans_next.api.http_schemas import (
    PredictionsV1RequestItem,
    PredictionsV1ResponseItem,
)
from beans_next.cache.sqlite_kv import SqliteJsonStore

__all__ = [
    "TwoLayerRunCache",
    "inference_cache_key",
    "scoring_cache_key",
]


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inference_cache_key(
    *,
    predict_url: str,
    model_revision: str | None,
    item: PredictionsV1RequestItem,
) -> str:
    """Stable primary key for one ``predictions_v1`` request item.

    Parameters
    ----------
    predict_url
        Effective launcher ``POST /predict`` URL (disambiguates servers).
    model_revision
        Launcher ``/info`` ``model_revision`` field; ``None`` when absent.

        Notes
        -----
        This value is accepted for forward compatibility with callers, but it
        is **not** currently incorporated into the cache key. The I6-A cache key
        design namespaces inference by ``predict_url`` and the canonical wire
        request item only.
    item
        One batched wire request row.

    Returns
    -------
    str
        Hex digest prefixed for future schema evolution.

    Warnings
    --------
    ``model_revision`` is accepted for forward compatibility but is **not**
    included in the key. If the model checkpoint changes at the same URL,
    cached inference entries will still be returned. Clear or relocate the
    cache directory whenever the launcher model changes.
    """
    canonical = json.dumps(
        item.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = predict_url + "\x00" + canonical
    return f"inf:v1:{_sha256_hex(payload)}"


def scoring_cache_key(
    *,
    predict_url: str,
    sample_id: str,
    task_id: str | None,
    labels: str | list[str] | dict[str, Any] | None,
    raw_predictions: list[str],
    processed_prediction: str,
    postprocess_fingerprint: str,
    postprocess_version: str | None,
    scorer_versions: dict[str, str] | None,
) -> str:
    """Stable primary key for metric outputs of one scored sample.

    Parameters
    ----------
    predict_url
        Launcher URL (scores may depend on which model produced ``raw_predictions``).
    sample_id
        Dataset sample identifier.
    task_id
        Optional eval-task id.
    labels
        Ground-truth labels as stored on the dataset example (JSON-serialized).
    raw_predictions
        Raw launcher output strings (primary scoring input).
    processed_prediction
        Post-processed text used by ``score_sample``.
    postprocess_fingerprint
        Canonical fingerprint of parser/cleaner configuration.
    postprocess_version
        Optional version tag from :class:`~beans_next.runner.runner.RunnerConfig`.
    scorer_versions
        Optional reproducibility map from runner config.

    Returns
    -------
    str
        Hex digest prefixed for future schema evolution.
    """
    payload = {
        "predict_url": predict_url,
        "sample_id": sample_id,
        "task_id": task_id,
        "labels": labels,
        "raw_predictions": list(raw_predictions),
        "processed_prediction": processed_prediction,
        "postprocess_fingerprint": postprocess_fingerprint,
        "postprocess_version": postprocess_version,
        "scorer_versions": scorer_versions or {},
    }
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return f"score:v1:{_sha256_hex(canonical)}"


class TwoLayerRunCache:
    """Open inference + scoring SQLite caches for one benchmark process.

    Notes
    -----
    Inference entries are only written when the wire response has no
    ``error`` field, so transient launcher failures can be retried without
    poisoning the cache.

    Scoring entries are only read/written when the runner uses the default
    ``beans_next.metrics.score_sample`` path (no custom ``scorer`` callback).
    """

    def __init__(
        self,
        *,
        inference_store: SqliteJsonStore,
        scoring_store: SqliteJsonStore,
        predict_url: str,
        model_revision: str | None = None,
    ) -> None:
        self._infer = inference_store
        self._score = scoring_store
        self._predict_url = predict_url
        self._model_revision = model_revision

    @classmethod
    def open(
        cls,
        cache_dir: Path,
        *,
        predict_url: str,
        model_revision: str | None = None,
    ) -> TwoLayerRunCache:
        """Create stores under ``cache_dir`` (mkdirs when missing).

        Parameters
        ----------
        cache_dir
            Directory that will hold ``inference.sqlite`` and ``scoring.sqlite``.
        predict_url
            Same URL passed to :class:`~beans_next.models.http.HttpClient`.
        model_revision
            Launcher ``/info`` ``model_revision`` field. Accepted for forward
            compatibility with callers; it is not currently incorporated into
            inference cache keys.

        Returns
        -------
        TwoLayerRunCache
            Ready-to-use cache handle.
        """
        resolved = cache_dir.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        inf = SqliteJsonStore(resolved / "inference.sqlite")
        sc = SqliteJsonStore(resolved / "scoring.sqlite")
        return cls(
            inference_store=inf,
            scoring_store=sc,
            predict_url=predict_url,
            model_revision=model_revision,
        )

    @property
    def predict_url(self) -> str:
        """Launcher URL bound into cache keys."""
        return self._predict_url

    @property
    def model_revision(self) -> str | None:
        """Launcher model revision bound into inference cache keys."""
        return self._model_revision

    def get_inference_item(
        self,
        item: PredictionsV1RequestItem,
    ) -> PredictionsV1ResponseItem | None:
        """Return a cached wire response row, if present and valid.

        Returns
        -------
        PredictionsV1ResponseItem or None
            A validated ``predictions_v1`` response row, or ``None`` on miss /
            corruption.
        """
        key = inference_cache_key(
            predict_url=self._predict_url,
            model_revision=self._model_revision,
            item=item,
        )
        raw = self._infer.get(key)
        if raw is None:
            return None
        try:
            return PredictionsV1ResponseItem.model_validate_json(raw)
        except Exception:
            return None

    def put_inference_item(
        self,
        item: PredictionsV1RequestItem,
        response_item: PredictionsV1ResponseItem,
    ) -> None:
        """Persist one successful inference response."""
        if response_item.error is not None:
            return
        key = inference_cache_key(
            predict_url=self._predict_url,
            model_revision=self._model_revision,
            item=item,
        )
        self._infer.put(
            key,
            response_item.model_dump_json(),
        )

    def get_scores(self, key: str) -> dict[str, float] | None:
        """Return cached metric floats for a scoring key, if any.

        Returns
        -------
        dict[str, float] or None
            Parsed metric map, or ``None`` when missing or invalid JSON.
        """
        raw = self._score.get(key)
        if raw is None:
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        out: dict[str, float] = {}
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                out[k] = float(v)
        return out

    def put_scores(self, key: str, scores: dict[str, float]) -> None:
        """Persist metric outputs for a scoring key."""
        self._score.put(
            key,
            json.dumps(scores, sort_keys=True, separators=(",", ":")),
        )

    def close(self) -> None:
        """Close both SQLite connections."""
        self._infer.close()
        self._score.close()
