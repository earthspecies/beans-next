"""BEANS-Next Tier-2 reference launcher: `hf_transformers` (``predictions_v1`` contract).

This is a minimal, contract-first FastAPI app. It defaults to **stub mode** so it
can pass launcher conformance on CPU-only machines without importing heavy ML
dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"

LAUNCHER_NAME: str = "beans-next-tier2-hf-transformers"
DEFAULT_MODEL_NAME: str = "hf_transformers/unspecified"

# Tier-2 reference launchers should be conservative by default.
MAX_BATCH_SIZE: int = int(os.environ.get("HF_TRANSFORMERS_MAX_BATCH_SIZE", "8"))

_AUDIO_PAYLOAD_TYPES: list[str] = ["base64_wav", "file_path", "file_url"]
_SUPPORTS_BATCHING: bool = True
_SCHEMA_VERSIONS: list[str] = [PREDICTIONS_V1]


def _stub_enabled() -> bool:
    return os.environ.get("HF_TRANSFORMERS_STUB", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


class HttpChatMessage(BaseModel):
    """A single chat message in a `predictions_v1` request item.

    Attributes
    ----------
    role
        Message role (e.g. `system`, `user`, `assistant`).
    content
        Message content.
    """

    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class HttpAudioInput(BaseModel):
    """Audio payload metadata in a `predictions_v1` request item.

    Attributes
    ----------
    payload_type
        Audio transport type (e.g. `base64_wav`, `file_path`, `file_url`).
    data
        Audio payload (encoded or a locator, depending on `payload_type`).
    sample_rate
        Declared sample rate (Hz).
    """

    model_config = ConfigDict(extra="forbid")
    payload_type: Literal["base64_wav", "file_path", "file_url"]
    data: str
    sample_rate: int


class HttpGenerationConfig(BaseModel):
    """Generation parameters for a request item.

    Notes
    -----
    Extra keys are allowed so clients can send forward-compatible settings.
    """

    model_config = ConfigDict(extra="allow")
    max_tokens: int | None = None
    temperature: float | None = None


class PredictionsV1RequestItem(BaseModel):
    """One element of the `requests` list in a `predictions_v1` envelope.

    Attributes
    ----------
    sample_id
        Stable identifier for request/response matching.
    messages
        Chat messages; audio placeholders (if any) live in message content.
    audio_inputs
        Audio inputs aligned with placeholders.
    generation_config
        Generation-time settings for the model.
    """

    model_config = ConfigDict(extra="forbid")
    sample_id: str
    messages: list[HttpChatMessage]
    audio_inputs: list[HttpAudioInput]
    generation_config: HttpGenerationConfig


class PredictionsV1ResponseItem(BaseModel):
    """One element of the `responses` list in a `predictions_v1` envelope.

    Attributes
    ----------
    sample_id
        Identifier matching the corresponding request item.
    predictions
        Model outputs (n-best). Stub mode returns a single deterministic string.
    error
        Optional per-sample error string for partial failures.
    """

    model_config = ConfigDict(extra="forbid")
    sample_id: str
    predictions: list[str]
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_sec: float | None = None
    error: str | None = None


class PredictionsV1Response(BaseModel):
    """Top-level `predictions_v1` response envelope.

    Attributes
    ----------
    schema_version
        Wire schema version (must be `predictions_v1`).
    responses
        One response item per request item.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    responses: list[PredictionsV1ResponseItem]


class InfoResponse(BaseModel):
    """Capability document returned by `GET /info`.

    Attributes
    ----------
    name
        Launcher name.
    model
        Model identifier string.
    model_revision
        Model revision string (used for reproducibility).
    audio_payload_types
        Supported audio payload types.
    max_batch_size
        Maximum allowed `requests` batch size.
    supports_batching
        Whether batching is supported.
    schema_versions
        Supported schema versions (must include `predictions_v1`).
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    model: str
    model_revision: str
    audio_payload_types: list[str]
    max_batch_size: int
    supports_batching: bool
    schema_versions: list[str]
    load_status: str | None = None
    loading_stage: str | None = None
    loading_elapsed_sec: float | None = None
    last_error: str | None = None


def _deterministic_prediction(sample_id: str, item: PredictionsV1RequestItem) -> str:
    messages_payload = [{"role": m.role, "content": m.content} for m in item.messages]
    audio_meta = [
        {
            "payload_type": a.payload_type,
            "sample_rate": a.sample_rate,
            "data_len": len(a.data),
        }
        for a in item.audio_inputs
    ]
    gen = item.generation_config.model_dump(mode="json", exclude_none=True)
    key = json.dumps(
        {
            "launcher": LAUNCHER_NAME,
            "sample_id": sample_id,
            "messages": messages_payload,
            "audio_meta": audio_meta,
            "gen": gen,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"stub:{digest}"


def _item_response_or_error(
    sample_id: str, raw: dict[str, Any]
) -> PredictionsV1ResponseItem:
    try:
        item = PredictionsV1RequestItem.model_validate(raw)
    except ValidationError as exc:
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error=f"invalid request item: {exc}",
        )

    if len(item.messages) == 0:
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error="missing or empty messages",
        )

    if _stub_enabled():
        pred = _deterministic_prediction(sample_id, item)
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[pred],
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            latency_sec=0.0,
            error=None,
        )

    # Non-stub mode: reference starter only (kept intentionally minimal).
    # Users should extend this to load a specific HF Transformers model.
    return PredictionsV1ResponseItem(
        sample_id=sample_id,
        predictions=[],
        error=(
            "HF_TRANSFORMERS_STUB is disabled, but real inference is not "
            "implemented in this Tier-2 starter. Set HF_TRANSFORMERS_STUB=1 "
            "or extend serve.py."
        ),
    )


app = FastAPI(title="BEANS-Next Tier-2 launcher: hf_transformers", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    load_status = "stub" if _stub_enabled() else "failed"
    return {"status": "ok", "load_status": load_status}


@app.get("/info")
def info() -> InfoResponse:
    model_name = os.environ.get("HF_TRANSFORMERS_MODEL", DEFAULT_MODEL_NAME)
    if _stub_enabled():
        model_revision = "stub"
        load_status = "stub"
        last_error = None
    else:
        model_revision = os.environ.get("HF_TRANSFORMERS_MODEL_REVISION", "unknown")
        load_status = "failed"
        last_error = (
            "HF_TRANSFORMERS_STUB is disabled, but real inference is not implemented "
            "in this Tier-2 starter. Set HF_TRANSFORMERS_STUB=1 or extend serve.py."
        )
    return InfoResponse(
        name=LAUNCHER_NAME,
        model=model_name,
        model_revision=model_revision,
        audio_payload_types=list(_AUDIO_PAYLOAD_TYPES),
        max_batch_size=MAX_BATCH_SIZE,
        supports_batching=_SUPPORTS_BATCHING,
        schema_versions=list(_SCHEMA_VERSIONS),
        load_status=load_status,
        last_error=last_error,
    )


@app.post("/predict")
def predict(body: dict[str, Any]) -> Response:
    if body.get("schema_version") != PREDICTIONS_V1:
        raise HTTPException(
            status_code=400,
            detail=f"schema_version must be {PREDICTIONS_V1!r}",
        )

    raw_requests = body.get("requests")
    if not isinstance(raw_requests, list):
        raise HTTPException(
            status_code=400,
            detail="body.requests must be a JSON array",
        )

    if len(raw_requests) > MAX_BATCH_SIZE:
        msg = f"batch size {len(raw_requests)} exceeds max_batch_size={MAX_BATCH_SIZE}"
        return Response(
            content=json.dumps({"detail": msg}),
            status_code=413,
            media_type="application/json",
        )

    seen: set[str] = set()
    for idx, raw in enumerate(raw_requests):
        if not isinstance(raw, dict):
            raise HTTPException(
                status_code=400,
                detail=f"requests[{idx}] must be a JSON object",
            )
        sid = raw.get("sample_id")
        if not isinstance(sid, str) or not sid:
            raise HTTPException(
                status_code=400,
                detail=f"requests[{idx}].sample_id must be a non-empty string",
            )
        if sid in seen:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate sample_id in batch: {sid!r}",
            )
        seen.add(sid)

    responses: list[PredictionsV1ResponseItem] = []
    for raw in raw_requests:
        assert isinstance(raw, dict)
        sample_id = raw["sample_id"]
        assert isinstance(sample_id, str)
        responses.append(_item_response_or_error(sample_id, raw))

    responses.sort(key=lambda r: r.sample_id, reverse=True)
    envelope = PredictionsV1Response(responses=responses)
    return Response(
        content=envelope.model_dump_json(),
        media_type="application/json",
        status_code=200,
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("HF_TRANSFORMERS_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
