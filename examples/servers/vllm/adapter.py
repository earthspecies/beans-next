"""BEANS-Next vLLM adapter sidecar HTTP launcher (``predictions_v1`` contract).

This launcher is part of the BEANS-Next serving kit. It intentionally keeps an
isolated environment and does not import ``beans_next``.

Modes
-----
Stub mode (``VLLM_ADAPTER_STUB=1``):
    Returns deterministic fake predictions. Used for contract conformance
    testing and CPU-only validation.

Proxy mode (``VLLM_ADAPTER_STUB=0``):
    Proxies each request item to an OpenAI-compatible upstream (typically
    ``vllm serve`` exposing ``POST /v1/chat/completions``) and translates the
    response into a ``predictions_v1`` response item.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"


def _get_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int_env(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    return int(val)


def _get_float_env(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    return float(val)


def _get_optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _canonicalize_base64_wav(
    data_b64: str,
    *,
    clip_seconds: float | None,
) -> str:
    """Best-effort: re-encode to canonical PCM16 WAV (optionally clipped).

    This mirrors the behavior of the OpenAI-compatible proxy launcher:
    optionally clip to the first N seconds to bound request size and reduce
    downstream token pressure for omni models.

    If decoding fails for any reason, returns the original base64 string unchanged.

    Returns
    -------
    str
        Base64-encoded WAV data.

        When decoding + re-encoding succeeds, this is a canonical PCM16 WAV (optionally
        clipped). Otherwise, the input `data_b64` is returned unchanged.
    """

    try:
        import numpy as np
        import soundfile as sf
    except Exception:  # noqa: BLE001
        return data_b64

    try:
        raw = base64.b64decode(data_b64)
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if not isinstance(sr, int) or sr <= 0:
            return data_b64

        if clip_seconds is not None and clip_seconds > 0:
            n = int(sr * clip_seconds)
            if n > 0:
                audio = audio[:n]

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio[:, 0]

        out = io.BytesIO()
        sf.write(out, audio, sr, format="WAV", subtype="PCM_16")
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return data_b64


class HttpChatMessage(BaseModel):
    """Chat message element in a `predictions_v1` request.

    Attributes
    ----------
    role:
        Chat role (e.g. `system`, `user`, `assistant`).
    content:
        Message content as plain text.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class HttpAudioInput(BaseModel):
    """Audio input element in a `predictions_v1` request.

    Attributes
    ----------
    payload_type:
        Audio transport type (`base64_wav`, `file_path`, or `file_url`).
    data:
        Payload data (base64 bytes or path/URL string depending on `payload_type`).
    sample_rate:
        Declared audio sample rate (Hz).
    """

    model_config = ConfigDict(extra="forbid")

    payload_type: Literal["base64_wav", "file_path", "file_url"]
    data: str
    sample_rate: int


class HttpGenerationConfig(BaseModel):
    """Generation parameters for a request item.

    Notes
    -----
    This model allows extra keys so the launcher can ignore unknown generation
    parameters while remaining schema-compatible.
    """

    model_config = ConfigDict(extra="allow")

    max_tokens: int | None = None
    temperature: float | None = None


class PredictionsV1RequestItem(BaseModel):
    """One item in a `predictions_v1` batch request.

    Attributes
    ----------
    sample_id:
        Unique identifier for request/response matching.
    messages:
        Chat-style messages. Audio placeholders are expected in message content.
    audio_inputs:
        Audio payloads aligned to `<AudioHere>` placeholders.
    generation_config:
        Generation parameters (best-effort proxy mapping in proxy mode).
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    messages: list[HttpChatMessage]
    audio_inputs: list[HttpAudioInput]
    generation_config: HttpGenerationConfig


class HttpUsage(BaseModel):
    """Token usage accounting returned by a launcher (optional).

    Notes
    -----
    This launcher does not attempt exact token accounting in proxy mode. Stub
    mode returns a small placeholder usage object to satisfy contract tooling.
    """

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class PredictionsV1ResponseItem(BaseModel):
    """One item in a `predictions_v1` batch response.

    Attributes
    ----------
    sample_id:
        Identifier of the request item this response corresponds to.
    predictions:
        List of model predictions (typically length 1).
    error:
        Optional per-item error message; when set, the HTTP call still returns
        status 200 to represent a partial failure.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    predictions: list[str]
    finish_reason: str | None = None
    usage: HttpUsage | None = None
    latency_sec: float | None = None
    error: str | None = None


class PredictionsV1Response(BaseModel):
    """Envelope for a `predictions_v1` batch response."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    responses: list[PredictionsV1ResponseItem]


class InfoResponse(BaseModel):
    """Capability discovery response for `GET /info`."""

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


def _deterministic_stub_prediction(
    sample_id: str,
    item: PredictionsV1RequestItem,
) -> str:
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
    return f"vllm_adapter_stub:{digest}"


def _extract_chat_completion_text(resp_json: dict[str, Any]) -> str:
    choices = resp_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("upstream response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("upstream choices[0] is not an object")

    msg = first.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content

    text = first.get("text")
    if isinstance(text, str):
        return text

    raise ValueError("upstream response missing assistant content text")


def _http_error_to_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text
        return f"upstream HTTP {status}: {body[:500]}"
    if isinstance(exc, httpx.TimeoutException):
        return "upstream timeout"
    return f"upstream error: {exc}"


@dataclass(frozen=True)
class _AdapterConfig:
    """Runtime configuration for the adapter sidecar."""

    stub: bool
    max_batch_size: int
    supports_batching: bool
    schema_versions: list[str]
    audio_payload_types: list[str]

    launcher_name: str
    model_id: str
    model_revision: str

    upstream_base_url: str
    upstream_timeout_sec: float
    upstream_retries: int
    audio_content_format: str
    output_modalities: list[str]
    extra_body: dict[str, Any]
    audio_only: bool


def _load_config() -> _AdapterConfig:
    stub = _get_bool_env("VLLM_ADAPTER_STUB", True)

    max_batch = _get_int_env("VLLM_MAX_BATCH_SIZE", 32)
    supports_batching = True
    schema_versions = [PREDICTIONS_V1]
    audio_payload_types = ["base64_wav", "file_path", "file_url"]

    launcher_name = "beans-next-vllm-adapter"
    model_id = os.environ.get("VLLM_MODEL_ID", "vllm/unknown")
    model_revision = os.environ.get("VLLM_MODEL_REVISION", "unknown")

    upstream_base_url = os.environ.get("VLLM_UPSTREAM_BASE_URL", "").strip()
    upstream_timeout_sec = _get_float_env("VLLM_UPSTREAM_TIMEOUT_SEC", 30.0)
    upstream_retries = _get_int_env("VLLM_UPSTREAM_RETRIES", 1)
    audio_content_format = os.environ.get(
        "VLLM_AUDIO_CONTENT_FORMAT", "input_audio"
    ).strip()
    output_modalities = [
        item.strip()
        for item in os.environ.get("VLLM_OUTPUT_MODALITIES", "").split(",")
        if item.strip()
    ]
    extra_body_raw = os.environ.get("VLLM_EXTRA_BODY_JSON", "").strip()
    extra_body: dict[str, Any] = {}
    if extra_body_raw:
        loaded = json.loads(extra_body_raw)
        if not isinstance(loaded, dict):
            raise ValueError("VLLM_EXTRA_BODY_JSON must decode to a JSON object")
        extra_body = loaded
    audio_only = _get_bool_env("VLLM_AUDIO_ONLY", False)

    if max_batch < 1:
        raise ValueError("VLLM_MAX_BATCH_SIZE must be >= 1")
    if audio_content_format not in {"input_audio", "audio_url_data"}:
        raise ValueError(
            "VLLM_AUDIO_CONTENT_FORMAT must be 'input_audio' or 'audio_url_data'"
        )

    return _AdapterConfig(
        stub=stub,
        max_batch_size=max_batch,
        supports_batching=supports_batching,
        schema_versions=schema_versions,
        audio_payload_types=audio_payload_types,
        launcher_name=launcher_name,
        model_id=model_id,
        model_revision=model_revision,
        upstream_base_url=upstream_base_url,
        upstream_timeout_sec=upstream_timeout_sec,
        upstream_retries=upstream_retries,
        audio_content_format=audio_content_format,
        output_modalities=output_modalities,
        extra_body=extra_body,
        audio_only=audio_only,
    )


CFG = _load_config()

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

HTTP = httpx.Client(timeout=httpx.Timeout(CFG.upstream_timeout_sec))

app = FastAPI(title="BEANS-Next vLLM adapter sidecar", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    mode = "stub" if CFG.stub else "proxy"
    load_status = "stub" if CFG.stub else "ready"
    return {"status": "ok", "mode": mode, "load_status": load_status}


@app.get("/info")
def info() -> InfoResponse:
    return InfoResponse(
        name=CFG.launcher_name,
        model="stub" if CFG.stub else CFG.model_id,
        model_revision="stub" if CFG.stub else CFG.model_revision,
        audio_payload_types=list(CFG.audio_payload_types),
        max_batch_size=CFG.max_batch_size,
        supports_batching=CFG.supports_batching,
        schema_versions=list(CFG.schema_versions),
        load_status="stub" if CFG.stub else "ready",
    )


def _call_upstream_chat_completion(
    *,
    messages: list[dict[str, Any]],
    audio_inputs: list[HttpAudioInput],
    generation_config: HttpGenerationConfig,
) -> str:
    if not CFG.upstream_base_url:
        raise RuntimeError(
            "VLLM_UPSTREAM_BASE_URL is required when VLLM_ADAPTER_STUB=0"
        )

    # Best-effort OpenAI-compatible multimodal format: attach audio items to the last
    # user message.
    openai_messages: list[dict[str, Any]] = [
        {"role": m.get("role", ""), "content": m.get("content", "")} for m in messages
    ]

    clip_seconds = _get_optional_float_env("VLLM_ADAPTER_MAX_AUDIO_SECONDS")
    canonicalize = _get_bool_env("VLLM_ADAPTER_CANONICALIZE_WAV", default=False)

    audio_items: list[dict[str, Any]] = []
    for a in audio_inputs:
        if a.payload_type != "base64_wav":
            raise ValueError(
                "unsupported audio_inputs payload_type for proxy mode: "
                f"{a.payload_type!r}"
            )
        data_b64 = a.data
        if canonicalize or (clip_seconds is not None and clip_seconds > 0):
            data_b64 = _canonicalize_base64_wav(data_b64, clip_seconds=clip_seconds)
        if CFG.audio_content_format == "audio_url_data":
            audio_items.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{data_b64}"},
                }
            )
        else:
            audio_items.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": data_b64, "format": "wav"},
                }
            )

    if CFG.audio_only and audio_items:
        openai_messages = [{"role": "user", "content": audio_items}]
    elif audio_items:
        last_user = next(
            (
                i
                for i in range(len(openai_messages) - 1, -1, -1)
                if openai_messages[i].get("role") == "user"
            ),
            None,
        )
        if last_user is None:
            openai_messages.append({"role": "user", "content": []})
            last_user = len(openai_messages) - 1

        existing = openai_messages[last_user].get("content")
        if isinstance(existing, list):
            content_list: list[dict[str, Any]] = [*existing]
        elif isinstance(existing, str):
            content_list = [{"type": "text", "text": existing}]
        else:
            content_list = []
        # Qwen3-Omni's evaluation guidance expects audio before the text task.
        content_list = [*audio_items, *content_list]
        openai_messages[last_user]["content"] = content_list

    body: dict[str, Any] = {"model": CFG.model_id, "messages": openai_messages}
    gen = generation_config.model_dump(mode="json", exclude_none=True)
    if "max_tokens" in gen:
        body["max_tokens"] = gen["max_tokens"]
    if "temperature" in gen:
        body["temperature"] = gen["temperature"]
    if CFG.output_modalities:
        body["modalities"] = list(CFG.output_modalities)
    if CFG.extra_body:
        body.update(CFG.extra_body)

    url = f"{CFG.upstream_base_url.rstrip('/')}{_CHAT_COMPLETIONS_PATH}"

    last_exc: Exception | None = None
    for _ in range(1 + max(0, CFG.upstream_retries)):
        try:
            resp = HTTP.post(url, json=body)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise ValueError("upstream response is not a JSON object")
            return _extract_chat_completion_text(payload)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:
                continue
            raise

    assert last_exc is not None
    raise last_exc


def _item_response_or_error(
    sample_id: str,
    raw: dict[str, Any],
) -> PredictionsV1ResponseItem:
    try:
        item = PredictionsV1RequestItem.model_validate(raw)
    except ValidationError as exc:
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[""],
            error=f"invalid request item: {exc}",
        )

    if len(item.messages) == 0:
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[""],
            error="missing or empty messages",
        )

    t0 = time.time()

    if CFG.stub:
        pred = _deterministic_stub_prediction(sample_id, item)
        latency = time.time() - t0
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[pred],
            finish_reason="stop",
            usage=HttpUsage(prompt_tokens=1, completion_tokens=1),
            latency_sec=latency,
            error=None,
        )

    try:
        messages = [{"role": m.role, "content": m.content} for m in item.messages]
        text = _call_upstream_chat_completion(
            messages=messages,
            audio_inputs=item.audio_inputs,
            generation_config=item.generation_config,
        )
        latency = time.time() - t0
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[text],
            finish_reason="stop",
            usage=None,
            latency_sec=latency,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        latency = time.time() - t0
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[""],
            finish_reason=None,
            usage=None,
            latency_sec=latency,
            error=_http_error_to_message(exc),
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

    if len(raw_requests) > CFG.max_batch_size:
        msg = (
            f"batch size {len(raw_requests)} exceeds "
            f"max_batch_size={CFG.max_batch_size}"
        )
        return Response(
            content=json.dumps({"detail": msg}),
            status_code=413,
            media_type="application/json",
        )

    seen: set[str] = set()
    for idx, raw in enumerate(raw_requests):
        if not isinstance(raw, dict):
            raise HTTPException(
                status_code=400, detail=f"requests[{idx}] must be a JSON object"
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

    responses.sort(key=lambda r: r.sample_id)
    envelope = PredictionsV1Response(responses=responses)
    return Response(
        content=envelope.model_dump_json(),
        media_type="application/json",
        status_code=200,
    )


def main() -> None:
    """Run the launcher via Uvicorn."""
    import uvicorn

    host = os.environ.get("VLLM_ADAPTER_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
