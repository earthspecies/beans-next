"""BEANS-Next NVIDIA Audex adapter sidecar HTTP launcher (``predictions_v1`` contract).

This launcher is part of the BEANS-Next serving kit. It intentionally keeps an
isolated environment and does not import ``beans_next``.

Modes
-----
Stub mode (``AUDEX_ADAPTER_STUB=1``):
    Returns deterministic fake predictions. Used for contract conformance
    testing and CPU-only validation.

Proxy mode (``AUDEX_ADAPTER_STUB=0``):
    Proxies each request item to a local OpenAI-compatible upstream (``vllm
    serve`` running the vendor ``audex_30b_a3b_vllm`` plugin) and translates
    the response into a ``predictions_v1`` response item.

Audex-specific behavior (on top of the shared vLLM-adapter pattern used by
``examples/servers/vllm/adapter.py``):

- ``file_path`` audio inputs are sent upstream as
  ``{"type": "audio_url", "audio_url": {"url": "file:///..."}}`` instead of
  being base64-encoded, so the runner's ``--preserve-file-paths`` mode avoids
  re-encoding ~1.3 MB per request. ``base64_wav`` payloads fall back to a
  ``data:audio/wav;base64,...`` URI.
- ``/predict`` batches are proxied concurrently via a thread pool (the vLLM
  adapter's stub-derived proxy path is sequential, which is far too slow for
  51k rows against a 30B model).
- Returned text has any ``<think>...</think>`` span and stray Audex
  audio-codec/placeholder token ids stripped before being handed to the
  shared BEANS-Next postprocess/scoring pipeline, so scoring stays
  byte-identical across models.
- ``AUDEX_EXTRA_BODY_JSON`` (mirroring ``VLLM_EXTRA_BODY_JSON``) is merged
  into the upstream request body. This is how
  ``chat_template_kwargs={"enable_thinking": false}``, ``temperature``,
  ``top_p``, and ``logit_bias`` suppression of the audio-codec token ids
  (29/30/31) are set, so the greedy-vs-NVIDIA-sampling pilot arms are just
  two different values of this env var with no code change.
- When ``item.generation_config.max_length_seconds`` is set and the source
  clip is longer, the audio is head-truncated to that many seconds before
  being sent upstream (never zero-padded — Audex handles variable-length
  audio natively). Clips already at or under the cap pass through untouched.
  Truncated clips are cached under ``AUDEX_TRIMMED_CACHE_DIR`` keyed by
  ``(resolved source path, cap)`` so a resumed/chained run reuses them
  instead of re-truncating.

License: NVIDIA's Audex checkpoints are released under the NVIDIA OneWay
Noncommercial License. See ``examples/servers/audex/README.md``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"

# Audex audio-codec / placeholder token ids that must never leak into scored
# text output. Confirmed from the vendor client, which suppresses these via
# `allowed_token_ids` (a full-vocab allow-list); we suppress the same ids via
# `logit_bias` (far cheaper to send per request) and strip any that leak
# through as a belt-and-braces fallback.
AUDEX_CODEC_TOKEN_IDS: tuple[int, ...] = (29, 30, 31)

_THINK_SPAN_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODEC_TOKEN_RE = re.compile(
    r"<\|?(?:audio_codec|codec|audio_bos|audio_eos|audio_pad)[^>]*\|?>",
    re.IGNORECASE,
)


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


def _strip_think_and_codec_tokens(text: str) -> str:
    """Remove ``<think>...</think>`` spans and stray codec/placeholder tokens.

    Returns
    -------
    str
        The cleaned text, stripped of leading/trailing whitespace so the
        shared postprocess pipeline sees the same shape it sees from other
        launchers.
    """

    cleaned = _THINK_SPAN_RE.sub("", text)
    cleaned = _CODEC_TOKEN_RE.sub("", cleaned)
    return cleaned.strip()


class HttpChatMessage(BaseModel):
    """Chat message element in a `predictions_v1` request."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class HttpAudioInput(BaseModel):
    """Audio input element in a `predictions_v1` request."""

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
    max_length_seconds: int | None = None


class PredictionsV1RequestItem(BaseModel):
    """One item in a `predictions_v1` batch request."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    messages: list[HttpChatMessage]
    audio_inputs: list[HttpAudioInput]
    generation_config: HttpGenerationConfig


class HttpUsage(BaseModel):
    """Token usage accounting returned by a launcher (optional)."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class PredictionsV1ResponseItem(BaseModel):
    """One item in a `predictions_v1` batch response."""

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
    return f"audex_adapter_stub:{digest}"


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
    """Runtime configuration for the Audex adapter sidecar."""

    stub: bool
    max_batch_size: int
    max_concurrency: int
    supports_batching: bool
    schema_versions: list[str]
    audio_payload_types: list[str]

    launcher_name: str
    model_id: str
    model_revision: str

    upstream_base_url: str
    upstream_timeout_sec: float
    upstream_retries: int
    extra_body: dict[str, Any]


def _load_config() -> _AdapterConfig:
    stub = _get_bool_env("AUDEX_ADAPTER_STUB", True)

    max_batch = _get_int_env("AUDEX_MAX_BATCH_SIZE", 32)
    max_concurrency = _get_int_env("AUDEX_MAX_CONCURRENCY", 8)
    supports_batching = True
    schema_versions = [PREDICTIONS_V1]
    audio_payload_types = ["base64_wav", "file_path", "file_url"]

    launcher_name = "beans-next-audex-adapter"
    model_id = os.environ.get("AUDEX_MODEL_ID", "nvidia/Nemotron-Labs-Audex-30B-A3B")
    model_revision = os.environ.get("AUDEX_MODEL_REVISION", "unknown")

    upstream_base_url = os.environ.get("AUDEX_UPSTREAM_BASE_URL", "").strip()
    upstream_timeout_sec = _get_float_env("AUDEX_UPSTREAM_TIMEOUT_SEC", 120.0)
    upstream_retries = _get_int_env("AUDEX_UPSTREAM_RETRIES", 1)

    extra_body_raw = os.environ.get("AUDEX_EXTRA_BODY_JSON", "").strip()
    extra_body: dict[str, Any] = {}
    if extra_body_raw:
        loaded = json.loads(extra_body_raw)
        if not isinstance(loaded, dict):
            raise ValueError("AUDEX_EXTRA_BODY_JSON must decode to a JSON object")
        extra_body = loaded

    if max_batch < 1:
        raise ValueError("AUDEX_MAX_BATCH_SIZE must be >= 1")
    if max_concurrency < 1:
        raise ValueError("AUDEX_MAX_CONCURRENCY must be >= 1")

    return _AdapterConfig(
        stub=stub,
        max_batch_size=max_batch,
        max_concurrency=max_concurrency,
        supports_batching=supports_batching,
        schema_versions=schema_versions,
        audio_payload_types=audio_payload_types,
        launcher_name=launcher_name,
        model_id=model_id,
        model_revision=model_revision,
        upstream_base_url=upstream_base_url,
        upstream_timeout_sec=upstream_timeout_sec,
        upstream_retries=upstream_retries,
        extra_body=extra_body,
    )


CFG = _load_config()

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

HTTP = httpx.Client(timeout=httpx.Timeout(CFG.upstream_timeout_sec))

app = FastAPI(title="BEANS-Next Audex adapter sidecar", version="0.1.0")


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


def _trimmed_cache_dir() -> Path:
    configured = os.environ.get("AUDEX_TRIMMED_CACHE_DIR", "").strip()
    default_dir = Path(tempfile.gettempdir()) / "audex-trimmed-cache"
    cache_dir = Path(configured) if configured else default_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _atomic_write_wav(dest: Path, data: np.ndarray, samplerate: int) -> None:
    """Write a WAV file atomically so concurrent requests never see a partial file."""

    tmp = dest.with_name(f"{dest.name}.tmp-{os.getpid()}-{time.time_ns()}")
    sf.write(str(tmp), data, samplerate, format="WAV")
    os.replace(tmp, dest)


def _trim_wav_file(src: Path, max_length_seconds: int) -> Path:
    """Head-truncate a WAV file on disk to ``max_length_seconds``, with caching.

    Returns ``src`` unchanged if it is already at or under the cap. Results
    are cached in ``AUDEX_TRIMMED_CACHE_DIR`` keyed by the resolved source
    path and the cap, so repeated/resumed runs reuse the trimmed clip instead
    of re-truncating it.

    Returns
    -------
    Path
        The (possibly trimmed) path to serve upstream.
    """

    resolved = src.resolve()
    info = sf.info(str(resolved))
    if info.samplerate <= 0 or info.frames <= 0:
        return resolved
    duration = info.frames / info.samplerate
    if duration <= max_length_seconds:
        return resolved

    cache_key = f"{resolved}::{max_length_seconds}".encode()
    key = hashlib.sha256(cache_key).hexdigest()
    cached = _trimmed_cache_dir() / f"{key}.wav"
    if cached.exists():
        return cached

    data, sr = sf.read(str(resolved), dtype="float32", always_2d=False)
    target_frames = int(sr * max_length_seconds)
    _atomic_write_wav(cached, data[:target_frames], sr)
    return cached


def _trim_base64_wav(data_b64: str, max_length_seconds: int) -> str:
    """Head-truncate a base64-encoded WAV payload to ``max_length_seconds``.

    Returns the input unchanged if it is already at or under the cap.

    Returns
    -------
    str
        The (possibly trimmed) base64-encoded WAV payload.
    """

    raw = base64.b64decode(data_b64)
    with io.BytesIO(raw) as buf:
        info = sf.info(buf)
    if info.samplerate <= 0 or info.frames <= 0:
        return data_b64
    duration = info.frames / info.samplerate
    if duration <= max_length_seconds:
        return data_b64

    with io.BytesIO(raw) as buf:
        data, sr = sf.read(buf, dtype="float32", always_2d=False)
    target_frames = int(sr * max_length_seconds)
    out = io.BytesIO()
    sf.write(out, data[:target_frames], sr, format="WAV")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _audio_url_content_item(
    a: HttpAudioInput, max_length_seconds: int | None
) -> dict[str, Any]:
    """Build an OpenAI ``audio_url`` content item for one audio input.

    ``file_path`` payloads are sent as a ``file://`` URL so the runner's
    ``--preserve-file-paths`` mode never has to base64-encode audio. Any
    other payload type falls back to a ``data:`` URI. When
    ``max_length_seconds`` is set, the clip is head-truncated to that many
    seconds first (see ``_trim_wav_file`` / ``_trim_base64_wav``).

    Returns
    -------
    dict[str, Any]
        An OpenAI chat content item of type ``audio_url``.
    """

    if a.payload_type == "file_path":
        abs_path = Path(a.data).resolve()
        if max_length_seconds is not None:
            abs_path = _trim_wav_file(abs_path, int(max_length_seconds))
        return {"type": "audio_url", "audio_url": {"url": abs_path.as_uri()}}
    if a.payload_type == "file_url":
        return {"type": "audio_url", "audio_url": {"url": a.data}}
    # base64_wav
    data = a.data
    if max_length_seconds is not None:
        data = _trim_base64_wav(data, int(max_length_seconds))
    return {
        "type": "audio_url",
        "audio_url": {"url": f"data:audio/wav;base64,{data}"},
    }


def _call_upstream_chat_completion(
    *,
    messages: list[dict[str, Any]],
    audio_inputs: list[HttpAudioInput],
    generation_config: HttpGenerationConfig,
) -> str:
    if not CFG.upstream_base_url:
        raise RuntimeError(
            "AUDEX_UPSTREAM_BASE_URL is required when AUDEX_ADAPTER_STUB=0"
        )

    openai_messages: list[dict[str, Any]] = [
        {"role": m.get("role", ""), "content": m.get("content", "")} for m in messages
    ]

    max_length_seconds = generation_config.max_length_seconds
    audio_items = [
        _audio_url_content_item(a, max_length_seconds) for a in audio_inputs
    ]

    if audio_items:
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
        content_list = [*audio_items, *content_list]
        openai_messages[last_user]["content"] = content_list

    body: dict[str, Any] = {"model": CFG.model_id, "messages": openai_messages}
    gen = generation_config.model_dump(mode="json", exclude_none=True)
    if "max_tokens" in gen:
        body["max_tokens"] = gen["max_tokens"]
    if "temperature" in gen:
        body["temperature"] = gen["temperature"]
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
            text = _extract_chat_completion_text(payload)
            return _strip_think_and_codec_tokens(text)
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


def _process_request_batch(
    raw_requests: list[dict[str, Any]],
) -> list[PredictionsV1ResponseItem]:
    """Process a validated request batch, with concurrent upstream calls.

    A 30B model behind a single vLLM server is fast enough that per-request
    latency dominates unless requests are in flight concurrently; the stub
    path and single-item batches run inline.

    Returns
    -------
    list[PredictionsV1ResponseItem]
        Per-sample responses, in the input order of ``raw_requests``.
    """

    if CFG.stub or CFG.max_concurrency <= 1 or len(raw_requests) <= 1:
        return [
            _item_response_or_error(str(raw["sample_id"]), raw) for raw in raw_requests
        ]

    max_workers = min(CFG.max_concurrency, len(raw_requests))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            pool.map(
                lambda raw: _item_response_or_error(str(raw["sample_id"]), raw),
                raw_requests,
            )
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

    responses = _process_request_batch(raw_requests)
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

    host = os.environ.get("AUDEX_ADAPTER_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
