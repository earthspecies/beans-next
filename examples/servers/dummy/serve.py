"""Minimal BEANS-Next dummy HTTP launcher (``predictions_v1`` contract).

Standalone copy of the wire shapes from ``beans_next.api.http_schemas``; this
module does not import ``beans_next`` so the launcher keeps its own venv.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

# --- Contract constants (aligned with beans_next.api.http_schemas) ---

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"
MAX_BATCH_SIZE: int = 32

LAUNCHER_NAME: str = "beans-next-dummy"
MODEL_NAME: str = "dummy/no-weights"
MODEL_REVISION: str = "0"

AUDIO_PAYLOAD_TYPES: list[str] = ["base64_wav", "file_path", "file_url"]
SUPPORTS_BATCHING: bool = True
SCHEMA_VERSIONS: list[str] = [PREDICTIONS_V1]


class HttpChatMessage(BaseModel):
    """A single chat message in a ``predictions_v1`` request item."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class HttpAudioInput(BaseModel):
    """Audio payload metadata in a ``predictions_v1`` request item."""

    model_config = ConfigDict(extra="forbid")

    payload_type: Literal["base64_wav", "file_path", "file_url"]
    data: str
    sample_rate: int


class HttpGenerationConfig(BaseModel):
    """Generation parameters for a request item.

    Notes
    -----
    This model allows extra keys so launchers can ignore unknown generation
    parameters while remaining schema-compatible.
    """

    model_config = ConfigDict(extra="allow")

    max_tokens: int | None = None
    temperature: float | None = None


class PredictionsV1RequestItem(BaseModel):
    """One element of the ``requests`` list in a ``predictions_v1`` envelope."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    messages: list[HttpChatMessage]
    audio_inputs: list[HttpAudioInput]
    generation_config: HttpGenerationConfig


class HttpUsage(BaseModel):
    """Optional token usage metadata in a response item."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class PredictionsV1ResponseItem(BaseModel):
    """One element of the ``responses`` list in a ``predictions_v1`` envelope."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    predictions: list[str]
    finish_reason: str | None = None
    usage: HttpUsage | None = None
    latency_sec: float | None = None
    error: str | None = None


class PredictionsV1Response(BaseModel):
    """Top-level ``predictions_v1`` response envelope."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    responses: list[PredictionsV1ResponseItem]


class InfoResponse(BaseModel):
    """``GET /info`` capability document (DESIGN §4.3)."""

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
    """Return a stable label derived from request fields (no task registry).

    Parameters
    ----------
    sample_id
        Request `sample_id`.
    item
        Parsed request item.

    Returns
    -------
    str
        Deterministic dummy prediction string.
    """
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
            "sample_id": sample_id,
            "messages": messages_payload,
            "audio_meta": audio_meta,
            "gen": gen,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"dummy:{digest}"


def _item_response_or_error(
    sample_id: str,
    raw: dict[str, Any],
) -> PredictionsV1ResponseItem:
    """Build one response row, converting validation issues to `error`.

    Parameters
    ----------
    sample_id
        Request `sample_id`.
    raw
        Raw request item JSON object.

    Returns
    -------
    PredictionsV1ResponseItem
        Response row for this sample, with `error` set on per-item failure.
    """
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

    label = _deterministic_prediction(sample_id, item)
    return PredictionsV1ResponseItem(
        sample_id=sample_id,
        predictions=[label],
        finish_reason="stop",
        usage=HttpUsage(prompt_tokens=1, completion_tokens=1),
        latency_sec=0.0,
        error=None,
    )


app = FastAPI(title="BEANS-Next dummy launcher", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness probe.

    Returns
    -------
    dict[str, str]
        JSON object with a simple status value.
    """
    return {"status": "ok", "load_status": "ready"}


@app.get("/info")
def info() -> InfoResponse:
    """Server capability discovery (DESIGN §4.3).

    Returns
    -------
    InfoResponse
        Capability document advertised by this launcher.
    """
    return InfoResponse(
        name=LAUNCHER_NAME,
        model=MODEL_NAME,
        model_revision=MODEL_REVISION,
        audio_payload_types=list(AUDIO_PAYLOAD_TYPES),
        max_batch_size=MAX_BATCH_SIZE,
        supports_batching=SUPPORTS_BATCHING,
        schema_versions=list(SCHEMA_VERSIONS),
        load_status="ready",
    )


@app.post("/predict")
def predict(body: dict[str, Any]) -> Response:
    """`predictions_v1` batch inference; responses keyed by `sample_id`.

    Parameters
    ----------
    body
        Raw JSON request body.

    Returns
    -------
    Response
        JSON response envelope with one item per request.

    Raises
    ------
    HTTPException
        For batch-scope contract violations (bad schema, bad shape, duplicates).
    """
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

    # Order must not be relied upon by clients; sort descending by sample_id.
    responses.sort(key=lambda r: r.sample_id, reverse=True)

    envelope = PredictionsV1Response(responses=responses)
    return Response(
        content=envelope.model_dump_json(),
        media_type="application/json",
        status_code=200,
    )


def main() -> None:
    """Run the dummy launcher via Uvicorn."""
    import uvicorn

    host = os.environ.get("DUMMY_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
