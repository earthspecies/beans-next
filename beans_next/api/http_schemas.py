"""Pydantic models for the ``predictions_v1`` HTTP request/response contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Wire constants ---

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"

AudioPayloadType = Literal["base64_wav", "file_path", "file_url"]


class HttpChatMessage(BaseModel):
    """One chat message on the wire (``role`` + ``content``)."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class HttpAudioInput(BaseModel):
    """Single audio slot aligned with the i-th ``<AudioHere>`` placeholder."""

    model_config = ConfigDict(extra="forbid")

    payload_type: AudioPayloadType
    data: str = Field(
        ...,
        description="Base64 WAV payload, filesystem path, or URL string, depending on "
        "`payload_type`.",
    )
    sample_rate: int = Field(
        ...,
        description="Nominal sample rate in Hz for this slot.",
    )


class HttpGenerationConfig(BaseModel):
    """Decoding / sampling parameters for the launcher.

    The contract documents ``max_tokens`` and ``temperature`` (DESIGN §4.1).
    Additional keys may appear on the wire; they are preserved but not validated
    beyond JSON object shape when using :meth:`model_validate` on a dict with
    extra keys — this model uses ``extra='allow'`` so forward-compatible payloads
    round-trip.
    """

    model_config = ConfigDict(extra="allow")

    max_tokens: int | None = None
    temperature: float | None = None
    max_length_seconds: int | None = None


class PredictionsV1RequestItem(BaseModel):
    """One batched sample in a ``predictions_v1`` request."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    messages: list[HttpChatMessage]
    audio_inputs: list[HttpAudioInput]
    generation_config: HttpGenerationConfig


class PredictionsV1Request(BaseModel):
    """Envelope for ``POST /predict`` body (``predictions_v1``)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    requests: list[PredictionsV1RequestItem]


class HttpUsage(BaseModel):
    """Optional token (or similar) usage block on a per-sample response."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class PredictionsV1ResponseItem(BaseModel):
    """One batched sample in a ``predictions_v1`` response."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    predictions: list[str]
    finish_reason: str | None = None
    usage: HttpUsage | None = None
    latency_sec: float | None = None
    error: str | None = None


class PredictionsV1Response(BaseModel):
    """Envelope returned by ``POST /predict`` (``predictions_v1``)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    responses: list[PredictionsV1ResponseItem]


__all__ = [
    "AudioPayloadType",
    "HttpAudioInput",
    "HttpChatMessage",
    "HttpGenerationConfig",
    "HttpUsage",
    "PREDICTIONS_V1",
    "PredictionsV1Request",
    "PredictionsV1RequestItem",
    "PredictionsV1Response",
    "PredictionsV1ResponseItem",
]
