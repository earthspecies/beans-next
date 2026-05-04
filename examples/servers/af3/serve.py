"""BEANS-Next launcher: Audio Flamingo Next (``predictions_v1`` contract).

Serves ``nvidia/audio-flamingo-next-hf`` (8B, BF16) via the standard
BEANS-Next HTTP contract.  Defaults to **stub mode** so the launcher can pass
contract conformance on CPU-only machines without loading any model weights.

Modes
-----
Stub mode (``AF3_STUB=1``):
    Returns deterministic hashed strings.  Used for contract conformance
    testing and CPU-only validation.

Real inference (``AF3_STUB=0``, default when env var is absent):
    Loads the model at startup (requires GPU + transformers ≥ 4.47).
    Audio is accepted as ``base64_wav``, ``file_path``, or ``file_url``;
    base64 and URL payloads are written to a per-request temp directory
    that is cleaned up after each call.

License
-------
``nvidia/audio-flamingo-next-hf`` is released under the NVIDIA OneWay
Noncommercial License.  Only non-commercial research use is permitted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"

LAUNCHER_NAME: str = "beans-next-af-next"
_DEFAULT_MODEL_ID: str = "nvidia/audio-flamingo-next-hf"
_DEFAULT_MAX_NEW_TOKENS: int = 512
_DEFAULT_REPETITION_PENALTY: float = 1.2

MAX_BATCH_SIZE: int = int(os.environ.get("AF3_MAX_BATCH_SIZE", "4"))

_AUDIO_PAYLOAD_TYPES: list[str] = ["base64_wav", "file_path", "file_url"]
_SUPPORTS_BATCHING: bool = True
_SCHEMA_VERSIONS: list[str] = [PREDICTIONS_V1]

_AUDIO_PLACEHOLDER: re.Pattern[str] = re.compile(r"<Audio><AudioHere></Audio>")

# Module-level model handles; populated by _load_model() at startup.
_model: Any = None
_processor: Any = None
_loaded_model_id: str = _DEFAULT_MODEL_ID
_loaded_model_revision: str = "unknown"


@dataclass
class _LoadState:
    """Track model load status for `/health` and `/info`."""

    status: Literal["stub", "ready", "loading", "failed"] = "stub"
    stage: str | None = None
    stage_started_at: float | None = None
    last_error: str | None = None


_LOAD_STATE = _LoadState()
_LOAD_LOCK = threading.Lock()


def _stub_enabled() -> bool:
    return os.environ.get("AF3_STUB", "").strip() not in ("", "0", "false", "False")


def _load_model() -> None:
    global _model, _processor, _loaded_model_id, _loaded_model_revision
    import torch
    from transformers import AutoModel, AutoProcessor

    if os.environ.get("AF3_ALLOW_CPU", "").strip() not in (
        "1",
        "true",
        "True",
    ) and not torch.cuda.is_available():
        raise RuntimeError(
            "AF-Next real mode requires CUDA, but torch.cuda.is_available() is False. "
            "The cluster driver must match the PyTorch CUDA build. "
            "Install the launcher "
            "with `uv sync --group gpu` (torch from the PyTorch cu124 index; see "
            "examples/servers/af3/pyproject.toml), or on misconfigured nodes set "
            "AF3_ALLOW_CPU=1 for debugging only."
        )

    model_id = os.environ.get("AF3_MODEL", _DEFAULT_MODEL_ID)
    revision = os.environ.get("AF3_MODEL_REVISION", None)

    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if revision:
        kwargs["revision"] = revision

    _processor = AutoProcessor.from_pretrained(
        model_id, **({"revision": revision} if revision else {})
    )
    _model = AutoModel.from_pretrained(model_id, **kwargs).eval()

    _loaded_model_id = model_id
    # Best-effort: read the resolved revision from the model config or env.
    _loaded_model_revision = (
        revision
        or getattr(getattr(_model, "config", None), "_commit_hash", None)
        or os.environ.get("AF3_MODEL_REVISION", "unknown")
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:  # type: ignore[type-arg]
    if _stub_enabled():
        with _LOAD_LOCK:
            _LOAD_STATE.status = "stub"
            _LOAD_STATE.stage = None
            _LOAD_STATE.stage_started_at = None
            _LOAD_STATE.last_error = None
        yield
        return

    # Real mode: start loading in a background thread so `/health` responds fast.
    with _LOAD_LOCK:
        _LOAD_STATE.status = "loading"
        _LOAD_STATE.stage = "loading_model_runtime"
        _LOAD_STATE.stage_started_at = time.time()
        _LOAD_STATE.last_error = None

    def _bg_load() -> None:
        try:
            _load_model()
            with _LOAD_LOCK:
                _LOAD_STATE.status = "ready"
                _LOAD_STATE.stage = None
                _LOAD_STATE.stage_started_at = None
                _LOAD_STATE.last_error = None
        except Exception as exc:  # noqa: BLE001
            with _LOAD_LOCK:
                _LOAD_STATE.status = "failed"
                _LOAD_STATE.last_error = str(exc)

    t = threading.Thread(target=_bg_load, daemon=True)
    t.start()
    yield


# ---------------------------------------------------------------------------
# Wire schemas (self-contained; intentionally does not import beans_next)
# ---------------------------------------------------------------------------


class HttpChatMessage(BaseModel):
    """A single chat message in a `predictions_v1` request item."""

    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class HttpAudioInput(BaseModel):
    """Audio payload metadata in a `predictions_v1` request item."""

    model_config = ConfigDict(extra="forbid")
    payload_type: Literal["base64_wav", "file_path", "file_url"]
    data: str
    sample_rate: int


class HttpGenerationConfig(BaseModel):
    """Generation parameters; extra keys forwarded as-is to ``model.generate``."""

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


class PredictionsV1ResponseItem(BaseModel):
    """One element of the ``responses`` list in a ``predictions_v1`` envelope."""

    model_config = ConfigDict(extra="forbid")
    sample_id: str
    predictions: list[str]
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_sec: float | None = None
    error: str | None = None


class PredictionsV1Response(BaseModel):
    """Top-level ``predictions_v1`` response envelope."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    responses: list[PredictionsV1ResponseItem]


class InfoResponse(BaseModel):
    """Capability document returned by ``GET /info``."""

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


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _deterministic_prediction(sample_id: str, item: PredictionsV1RequestItem) -> str:
    key = json.dumps(
        {
            "launcher": LAUNCHER_NAME,
            "sample_id": sample_id,
            "messages": [{"role": m.role, "content": m.content} for m in item.messages],
            "audio_meta": [
                {
                    "payload_type": a.payload_type,
                    "sample_rate": a.sample_rate,
                    "data_len": len(a.data),
                }
                for a in item.audio_inputs
            ],
            "gen": item.generation_config.model_dump(mode="json", exclude_none=True),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"stub:{digest}"


# ---------------------------------------------------------------------------
# Real inference helpers
# ---------------------------------------------------------------------------


def _resolve_audio_path(audio: HttpAudioInput, tmp_dir: str) -> str:
    """Return a local file path for ``audio``, writing to ``tmp_dir`` if needed.

    Parameters
    ----------
    audio:
        Audio input descriptor from the request.
    tmp_dir:
        Temporary directory for decoded/downloaded files.

    Returns
    -------
    str
        Absolute path to a WAV file on the local filesystem.

    Raises
    ------
    ValueError
        If ``audio.payload_type`` is not one of ``file_path``, ``base64_wav``,
        or ``file_url``.
    """
    if audio.payload_type == "file_path":
        return audio.data
    if audio.payload_type == "base64_wav":
        wav_bytes = base64.b64decode(audio.data)
        fd, path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
        os.write(fd, wav_bytes)
        os.close(fd)
        return path
    if audio.payload_type == "file_url":
        with urllib.request.urlopen(audio.data, timeout=30) as resp:  # noqa: S310
            wav_bytes = resp.read()
        fd, path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
        os.write(fd, wav_bytes)
        os.close(fd)
        return path
    raise ValueError(f"Unsupported audio payload_type: {audio.payload_type!r}")


def _build_conversation(
    messages: list[HttpChatMessage],
    audio_paths: list[str],
) -> list[dict[str, Any]]:
    """Convert ``predictions_v1`` messages + resolved paths to AF-Next format.

    Audio placeholders (``<Audio><AudioHere></Audio>``) in message content are
    replaced with ``{"type": "audio", "path": ...}`` content items in order.
    Text segments surrounding placeholders become ``{"type": "text", "text": ...}``
    items.  Messages with no placeholders are passed as plain string content.

    Returns
    -------
    list[dict[str, Any]]
        Conversation in AF-Next format.
    """
    audio_idx = 0
    conv: list[dict[str, Any]] = []
    for msg in messages:
        parts = _AUDIO_PLACEHOLDER.split(msg.content)
        items: list[dict[str, Any]] = []
        for i, text_part in enumerate(parts):
            if text_part:
                items.append({"type": "text", "text": text_part})
            if i < len(parts) - 1 and audio_idx < len(audio_paths):
                items.append({"type": "audio", "path": audio_paths[audio_idx]})
                audio_idx += 1
        conv.append({"role": msg.role, "content": items})

    if audio_idx < len(audio_paths):
        target = next(
            (m for m in conv if m["role"] == "user"),
            conv[0] if conv else None,
        )
        if target is None:
            target = {"role": "user", "content": []}
            conv.append(target)
        target["content"].extend(
            {"type": "audio", "path": path} for path in audio_paths[audio_idx:]
        )
    return conv


def _run_inference(item: PredictionsV1RequestItem) -> PredictionsV1ResponseItem:
    """Run AF-Next inference for one request item.

    Returns
    -------
    PredictionsV1ResponseItem
        Response item with generated text and latency.
    """
    import torch

    t0 = time.perf_counter()
    tmp_dir = tempfile.mkdtemp(prefix="af3-audio-")
    try:
        audio_paths = [_resolve_audio_path(a, tmp_dir) for a in item.audio_inputs]
        conversation = [_build_conversation(item.messages, audio_paths)]

        gen_cfg = item.generation_config
        max_new_tokens = gen_cfg.max_tokens or _DEFAULT_MAX_NEW_TOKENS
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
        }
        if gen_cfg.temperature is not None:
            if gen_cfg.temperature > 0:
                generate_kwargs["temperature"] = gen_cfg.temperature
                generate_kwargs["do_sample"] = True
            else:
                generate_kwargs["do_sample"] = False

        batch = _processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        batch = {
            k: (v.to(_model.device) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(_model.dtype)

        with torch.inference_mode():
            generated = _model.generate(**batch, **generate_kwargs)

        prompt_len = batch["input_ids"].shape[1]
        completion = generated[:, prompt_len:]
        text = _processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return PredictionsV1ResponseItem(
            sample_id=item.sample_id,
            predictions=[text],
            finish_reason="stop",
            usage={
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(completion.shape[1]),
            },
            latency_sec=time.perf_counter() - t0,
            error=None,
        )
    except Exception as exc:
        return PredictionsV1ResponseItem(
            sample_id=item.sample_id,
            predictions=[],
            latency_sec=time.perf_counter() - t0,
            error=str(exc),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-item dispatch
# ---------------------------------------------------------------------------


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

    if not item.messages:
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

    with _LOAD_LOCK:
        load_status = _LOAD_STATE.status
        stage = _LOAD_STATE.stage
        last_err = _LOAD_STATE.last_error

    if load_status == "failed":
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error=f"model failed: {last_err}",
        )
    if load_status != "ready":
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error=f"model loading in progress (stage={stage or 'initializing'})",
        )

    return _run_inference(item)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BEANS-Next launcher: Audio Flamingo Next",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    if _stub_enabled():
        return {"status": "ok", "mode": "stub"}

    with _LOAD_LOCK:
        status = _LOAD_STATE.status
        stage = _LOAD_STATE.stage
        started = _LOAD_STATE.stage_started_at
        last_err = _LOAD_STATE.last_error

    if status == "ready":
        return {"status": "ok", "mode": "real"}
    if status == "failed":
        raise HTTPException(status_code=503, detail=f"model failed: {last_err}")
    if status == "loading":
        elapsed = f"{time.time() - started:.1f}s" if started is not None else "unknown"
        raise HTTPException(
            status_code=503,
            detail=f"model loading in progress; stage={stage or 'initializing'}; "
            f"stage_elapsed={elapsed}",
        )
    raise HTTPException(status_code=503, detail="model not ready")


@app.get("/info")
def info() -> InfoResponse:
    if _stub_enabled():
        model_name = os.environ.get("AF3_MODEL", _DEFAULT_MODEL_ID)
        model_revision = "stub"
    else:
        model_name = _loaded_model_id
        model_revision = _loaded_model_revision
    with _LOAD_LOCK:
        load_status = _LOAD_STATE.status
        loading_stage = _LOAD_STATE.stage
        loading_elapsed_sec = (
            time.time() - _LOAD_STATE.stage_started_at
            if _LOAD_STATE.stage_started_at is not None and load_status == "loading"
            else None
        )
        last_error = _LOAD_STATE.last_error

    return InfoResponse(
        name=LAUNCHER_NAME,
        model=model_name,
        model_revision=model_revision,
        audio_payload_types=list(_AUDIO_PAYLOAD_TYPES),
        max_batch_size=MAX_BATCH_SIZE,
        supports_batching=_SUPPORTS_BATCHING,
        schema_versions=list(_SCHEMA_VERSIONS),
        load_status=load_status,
        loading_stage=loading_stage,
        loading_elapsed_sec=loading_elapsed_sec,
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
            status_code=400, detail="body.requests must be a JSON array"
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
        sid = raw["sample_id"]
        responses.append(_item_response_or_error(sid, raw))

    envelope = PredictionsV1Response(responses=responses)
    return Response(
        content=envelope.model_dump_json(),
        media_type="application/json",
        status_code=200,
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("AF3_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
