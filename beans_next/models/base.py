"""Protocols for model-side call shapes (HTTP adapter implements this)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from beans_next.api.http_schemas import PredictionsV1Request, PredictionsV1Response


@runtime_checkable
class InferenceModel(Protocol):
    """Abstract batch inference over the `predictions_v1` wire schema.

    The shipped implementation in BEANS-Next is `beans_next.models.http.HttpClient`
    (not defined in this increment). Callers and tests may use this protocol to
    type-check code that consumes launchers without importing a concrete client.

    Notes
    -----
    Matching of responses to requests is by `sample_id` only (DESIGN §4.6).

    """

    def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
        """Run inference for a full batch envelope.

        Parameters
        ----------
        request
            `predictions_v1` request body (one HTTP round-trip per call).

        Returns
        -------
        PredictionsV1Response
            `predictions_v1` response body with one item per request sample.

        """
        ...


__all__ = ["InferenceModel"]
