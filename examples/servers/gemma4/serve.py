"""BEANS-Next Gemma 4 12B adapter sidecar HTTP launcher (``predictions_v1`` contract).

This launcher is part of the BEANS-Next serving kit. It intentionally keeps an
isolated environment and does not import ``beans_next``.

Modes
-----
Stub mode (``GEMMA4_ADAPTER_STUB=1``, the default):
    Returns deterministic fake predictions. Used for contract conformance
    testing and CPU-only validation.

Proxy mode (``GEMMA4_ADAPTER_STUB=0``, ``GEMMA4_BACKEND=vllm``):
    Proxies each request item to a local OpenAI-compatible upstream (``vllm
    serve google/gemma-4-12B-it`` with ``vllm[audio]>=0.23.0``) and translates
    the response into a ``predictions_v1`` response item.

In-process mode (``GEMMA4_ADAPTER_STUB=0``, ``GEMMA4_BACKEND=transformers``):
    Loads ``Gemma4UnifiedForConditionalGeneration`` + ``AutoProcessor``
    in-process (bf16, one GPU) and runs batched generation directly, as a
    costed fallback if the vLLM audio path turns out not to be
    audio-sensitive (see the plan's Decision 1 probe). This path cannot be
    exercised without a GPU and model weights and needs on-cluster
    validation; see ``README.md``.

Gemma4-specific behavior (on top of the shared vLLM-adapter pattern used by
``examples/servers/vllm/adapter.py`` and cloned from
``examples/servers/audex/serve.py``):

- ``file_path`` audio inputs are sent upstream as
  ``{"type": "audio_url", "audio_url": {"url": "file:///..."}}`` instead of
  being base64-encoded, so the runner's ``--preserve-file-paths`` mode avoids
  re-encoding audio per request. ``base64_wav`` payloads fall back to a
  ``data:audio/wav;base64,...`` URI.
- ``/predict`` batches are proxied concurrently via a thread pool.
- Returned text has any ``<think>...</think>`` span stripped before being
  handed to the shared BEANS-Next postprocess/scoring pipeline. Unlike the
  Audex launcher, there is **no** codec-token suppression/stripping here:
  Gemma 4 12B has no audio-generation path, so the ``logit_bias`` on token
  ids 29/30/31 and the codec-token regex are both Audex-specific and are not
  carried over (see the plan's Decision 4).
- ``GEMMA4_EXTRA_BODY_JSON`` (mirroring ``AUDEX_EXTRA_BODY_JSON`` /
  ``VLLM_EXTRA_BODY_JSON``) is merged into the upstream request body. This is
  how ``chat_template_kwargs={"enable_thinking": false}``, ``temperature``,
  ``top_p``, and ``top_k`` are set, so the greedy-vs-vendor-sampling pilot
  arms are just two different values of this env var with no code change.
- Every request is length-capped: the *effective* cap is
  ``min(item.generation_config.max_length_seconds or infinity,
  GEMMA4_MAX_AUDIO_SECONDS)`` (default ``GEMMA4_MAX_AUDIO_SECONDS=30``,
  Gemma's documented hard audio limit). This differs from the Audex launcher,
  which only trims when the per-subset protocol cap is present; here the
  30 s model-capacity ceiling always applies, even on suites (BEANS-Next
  tiers 1-3) that send no ``max_length_seconds`` at all. Clips already at or
  under the effective cap pass through untouched (no copy). Truncated clips
  are cached under ``GEMMA4_TRIMMED_CACHE_DIR`` keyed by ``(resolved source
  path, effective cap)`` so a resumed/chained run reuses them instead of
  re-truncating. Truncate only, never zero-pad.
- Audio is placed **after** the text in the assembled prompt by default
  (``GEMMA4_AUDIO_POSITION=after_text``), the opposite of the Audex/generic
  vLLM-adapter convention (audio before text, per Qwen3-Omni's guidance).
  This is switchable to ``GEMMA4_AUDIO_POSITION=before_text`` for the
  pilot's audio-position A/B test (Decision 3).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
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

_THINK_SPAN_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_GEMMA4_TARGET_SAMPLE_RATE = 16_000


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


def _strip_think(text: str) -> str:
    """Remove any ``<think>...</think>`` span.

    Unlike the Audex launcher, no codec-token stripping is applied here:
    Gemma 4 12B has no audio-codec/placeholder token vocabulary to suppress
    (see the plan's Decision 4).

    Returns
    -------
    str
        The cleaned text, stripped of leading/trailing whitespace so the
        shared postprocess pipeline sees the same shape it sees from other
        launchers.
    """

    cleaned = _THINK_SPAN_RE.sub("", text)
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
    max_audio_seconds: float
    backend: str
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
    return f"gemma4_adapter_stub:{digest}"


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
    """Runtime configuration for the Gemma 4 adapter sidecar."""

    stub: bool
    backend: Literal["vllm", "transformers"]
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

    max_audio_seconds: float
    audio_position: Literal["after_text", "before_text"]


def _load_config() -> _AdapterConfig:
    stub = _get_bool_env("GEMMA4_ADAPTER_STUB", True)

    backend_raw = os.environ.get("GEMMA4_BACKEND", "vllm").strip().lower()
    if backend_raw not in {"vllm", "transformers"}:
        raise ValueError(
            f"GEMMA4_BACKEND must be 'vllm' or 'transformers', got {backend_raw!r}"
        )

    max_batch = _get_int_env("GEMMA4_MAX_BATCH_SIZE", 32)
    max_concurrency = _get_int_env("GEMMA4_MAX_CONCURRENCY", 8)
    supports_batching = True
    schema_versions = [PREDICTIONS_V1]
    audio_payload_types = ["base64_wav", "file_path", "file_url"]

    launcher_name = "beans-next-gemma4-adapter"
    model_id = os.environ.get("GEMMA4_MODEL_ID", "google/gemma-4-12B-it")
    model_revision = os.environ.get("GEMMA4_MODEL_REVISION", "unknown")

    upstream_base_url = os.environ.get("GEMMA4_UPSTREAM_BASE_URL", "").strip()
    upstream_timeout_sec = _get_float_env("GEMMA4_UPSTREAM_TIMEOUT_SEC", 120.0)
    upstream_retries = _get_int_env("GEMMA4_UPSTREAM_RETRIES", 1)

    extra_body_raw = os.environ.get("GEMMA4_EXTRA_BODY_JSON", "").strip()
    extra_body: dict[str, Any] = {}
    if extra_body_raw:
        loaded = json.loads(extra_body_raw)
        if not isinstance(loaded, dict):
            raise ValueError("GEMMA4_EXTRA_BODY_JSON must decode to a JSON object")
        extra_body = loaded

    max_audio_seconds = _get_float_env("GEMMA4_MAX_AUDIO_SECONDS", 30.0)

    audio_position_raw = (
        os.environ.get("GEMMA4_AUDIO_POSITION", "after_text").strip().lower()
    )
    if audio_position_raw not in {"after_text", "before_text"}:
        raise ValueError(
            "GEMMA4_AUDIO_POSITION must be 'after_text' or 'before_text', "
            f"got {audio_position_raw!r}"
        )

    if max_batch < 1:
        raise ValueError("GEMMA4_MAX_BATCH_SIZE must be >= 1")
    if max_concurrency < 1:
        raise ValueError("GEMMA4_MAX_CONCURRENCY must be >= 1")
    if max_audio_seconds <= 0:
        raise ValueError("GEMMA4_MAX_AUDIO_SECONDS must be > 0")

    return _AdapterConfig(
        stub=stub,
        backend=backend_raw,  # type: ignore[arg-type]
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
        max_audio_seconds=max_audio_seconds,
        audio_position=audio_position_raw,  # type: ignore[arg-type]
    )


CFG = _load_config()

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

HTTP = httpx.Client(timeout=httpx.Timeout(CFG.upstream_timeout_sec))

app = FastAPI(title="BEANS-Next Gemma 4 adapter sidecar", version="0.1.0")


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
        max_audio_seconds=CFG.max_audio_seconds,
        backend=CFG.backend,
        load_status="stub" if CFG.stub else "ready",
    )


def _trimmed_cache_dir() -> Path:
    configured = os.environ.get("GEMMA4_TRIMMED_CACHE_DIR", "").strip()
    default_dir = Path(tempfile.gettempdir()) / "gemma4-trimmed-cache"
    cache_dir = Path(configured) if configured else default_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _atomic_write_wav(dest: Path, data: np.ndarray, samplerate: int) -> None:
    """Write a WAV file atomically so concurrent requests never see a partial file."""

    tmp = dest.with_name(f"{dest.name}.tmp-{os.getpid()}-{time.time_ns()}")
    sf.write(str(tmp), data, samplerate, format="WAV")
    os.replace(tmp, dest)


def _effective_cap_seconds(max_length_seconds: int | None) -> float:
    """Compose the per-subset protocol cap with Gemma's hard audio limit.

    Returns ``min(max_length_seconds or infinity, GEMMA4_MAX_AUDIO_SECONDS)``.
    Unlike the Audex launcher (which only trims when a protocol cap is
    present), the returned cap is always finite here because
    ``GEMMA4_MAX_AUDIO_SECONDS`` (default 30s) always applies, even on suites
    that send no ``max_length_seconds`` at all (e.g. BEANS-Next tiers 1-3).

    Returns
    -------
    float
        The effective cap, in seconds.
    """

    protocol_cap = (
        float(max_length_seconds) if max_length_seconds is not None else math.inf
    )
    return min(protocol_cap, CFG.max_audio_seconds)


def _trim_wav_file(src: Path, cap_seconds: float) -> Path:
    """Head-truncate a WAV file on disk to ``cap_seconds``, with caching.

    Returns ``src`` unchanged if it is already at or under the cap. Results
    are cached in ``GEMMA4_TRIMMED_CACHE_DIR`` keyed by the resolved source
    path and the effective cap, so repeated/resumed runs reuse the trimmed
    clip instead of re-truncating it.

    Returns
    -------
    Path
        The (possibly trimmed) path to serve upstream.
    """

    resolved = src.resolve()
    if not math.isfinite(cap_seconds):
        return resolved

    info = sf.info(str(resolved))
    if info.samplerate <= 0 or info.frames <= 0:
        return resolved
    duration = info.frames / info.samplerate
    if duration <= cap_seconds:
        return resolved

    cache_key = f"{resolved}::{cap_seconds:.6f}".encode()
    key = hashlib.sha256(cache_key).hexdigest()
    cached = _trimmed_cache_dir() / f"{key}.wav"
    if cached.exists():
        return cached

    data, sr = sf.read(str(resolved), dtype="float32", always_2d=False)
    target_frames = int(sr * cap_seconds)
    _atomic_write_wav(cached, data[:target_frames], sr)
    return cached


def _trim_base64_wav(data_b64: str, cap_seconds: float) -> str:
    """Head-truncate a base64-encoded WAV payload to ``cap_seconds``.

    Returns the input unchanged if it is already at or under the cap.

    Returns
    -------
    str
        The (possibly trimmed) base64-encoded WAV payload.
    """

    if not math.isfinite(cap_seconds):
        return data_b64

    raw = base64.b64decode(data_b64)
    with io.BytesIO(raw) as buf:
        info = sf.info(buf)
    if info.samplerate <= 0 or info.frames <= 0:
        return data_b64
    duration = info.frames / info.samplerate
    if duration <= cap_seconds:
        return data_b64

    with io.BytesIO(raw) as buf:
        data, sr = sf.read(buf, dtype="float32", always_2d=False)
    target_frames = int(sr * cap_seconds)
    out = io.BytesIO()
    sf.write(out, data[:target_frames], sr, format="WAV")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _audio_url_content_item(
    a: HttpAudioInput, max_length_seconds: int | None
) -> dict[str, Any]:
    """Build an OpenAI ``audio_url`` content item for one audio input.

    ``file_path`` payloads are sent as a ``file://`` URL so the runner's
    ``--preserve-file-paths`` mode never has to base64-encode audio. Any
    other payload type falls back to a ``data:`` URI. The clip is always
    head-truncated to the effective cap (``min(max_length_seconds or
    infinity, GEMMA4_MAX_AUDIO_SECONDS)``, see ``_effective_cap_seconds``)
    first; clips already at or under the cap pass through untouched.

    Returns
    -------
    dict[str, Any]
        An OpenAI chat content item of type ``audio_url``.
    """

    cap_seconds = _effective_cap_seconds(max_length_seconds)

    if a.payload_type == "file_path":
        abs_path = Path(a.data).resolve()
        abs_path = _trim_wav_file(abs_path, cap_seconds)
        return {"type": "audio_url", "audio_url": {"url": abs_path.as_uri()}}
    if a.payload_type == "file_url":
        return {"type": "audio_url", "audio_url": {"url": a.data}}
    # base64_wav
    data = _trim_base64_wav(a.data, cap_seconds)
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
            "GEMMA4_UPSTREAM_BASE_URL is required when GEMMA4_ADAPTER_STUB=0"
        )

    openai_messages: list[dict[str, Any]] = [
        {"role": m.get("role", ""), "content": m.get("content", "")} for m in messages
    ]

    max_length_seconds = generation_config.max_length_seconds
    audio_items = [_audio_url_content_item(a, max_length_seconds) for a in audio_inputs]

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
        # Decision 3: Gemma wants audio *after* the text prompt (the opposite
        # of the Audex/generic-vLLM-adapter convention), switchable for the
        # pilot's audio-position A/B test.
        if CFG.audio_position == "before_text":
            content_list = [*audio_items, *content_list]
        else:
            content_list = [*content_list, *audio_items]
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
            return _strip_think(text)
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


# ---------------------------------------------------------------------------
# Transformers in-process backend (GEMMA4_BACKEND=transformers)
#
# This is the Decision-1 costed fallback: if the audio-sensitivity probe
# shows the vLLM audio path is not actually listening to the audio, the run
# switches to this in-process path instead. It cannot be exercised without a
# GPU and the real model weights, so this implementation gets the structure
# right (bf16, one GPU, batched /predict items, 16kHz mono float32 audio
# normalised to [-1, 1], defensive resampling with a hard rate assertion) but
# needs on-cluster validation before it is trusted for a full run. See
# README.md.
# ---------------------------------------------------------------------------


class _TransformersEngine:
    """Lazily-loaded in-process Gemma 4 Unified model + processor.

    Loaded once, on first use, so stub mode and the vLLM proxy path never
    import ``torch``/``transformers``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import (
                AutoProcessor,
                Gemma4UnifiedForConditionalGeneration,
            )

            model_id = CFG.model_id
            revision = None if CFG.model_revision == "unknown" else CFG.model_revision
            self._processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            if not hasattr(self._processor, "feature_extractor"):
                raise RuntimeError(
                    f"AutoProcessor for {model_id!r} exposes no feature_extractor; "
                    "this is the documented symptom of audio being silently "
                    "ignored (see the plan's Decision 1)."
                )
            self._model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self._model.eval()

    def _prepare_audio(
        self, audio_input: HttpAudioInput, cap_seconds: float
    ) -> np.ndarray:
        """Load one audio input as 16kHz mono float32 in [-1, 1].

        Resamples defensively (even though ``esp_data`` supplies
        ``audio_path_16KHz``) and asserts the final rate rather than trusting
        the declared ``sample_rate``, per the plan.

        Returns
        -------
        numpy.ndarray
            1-D float32 array, normalised to [-1, 1], at 16 kHz, truncated to
            ``cap_seconds`` (never zero-padded).

        Raises
        ------
        RuntimeError
            If the audio cannot be resampled to 16 kHz.
        """

        if audio_input.payload_type == "file_path":
            path = _trim_wav_file(Path(audio_input.data).resolve(), cap_seconds)
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        elif audio_input.payload_type == "base64_wav":
            b64 = _trim_base64_wav(audio_input.data, cap_seconds)
            raw = base64.b64decode(b64)
            with io.BytesIO(raw) as buf:
                data, sr = sf.read(buf, dtype="float32", always_2d=False)
        else:
            raise RuntimeError(
                "transformers backend cannot resolve "
                f"payload_type={audio_input.payload_type!r}"
            )

        if data.ndim > 1:
            data = data.mean(axis=1).astype("float32")

        if sr != _GEMMA4_TARGET_SAMPLE_RATE:
            import librosa

            data = librosa.resample(
                data.astype("float32"), orig_sr=sr, target_sr=_GEMMA4_TARGET_SAMPLE_RATE
            )
            sr = _GEMMA4_TARGET_SAMPLE_RATE

        assert sr == _GEMMA4_TARGET_SAMPLE_RATE, (
            f"audio must be resampled to {_GEMMA4_TARGET_SAMPLE_RATE}Hz, got {sr}"
        )

        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.0:
            data = data / peak
        return data.astype("float32")

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        audio_inputs: list[HttpAudioInput],
        generation_config: HttpGenerationConfig,
    ) -> str:
        """Run one batched-of-one generation through the in-process model.

        Returns
        -------
        str
            The cleaned generated text (``<think>`` stripped).
        """

        self._ensure_loaded()

        cap_seconds = _effective_cap_seconds(generation_config.max_length_seconds)
        audio_arrays = [self._prepare_audio(a, cap_seconds) for a in audio_inputs]

        # Gemma wants audio after the text prompt by default (Decision 3);
        # the processor's chat template determines final placement from the
        # message content ordering, mirroring the vLLM proxy path.
        gen_kwargs: dict[str, Any] = {}
        if generation_config.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = generation_config.max_tokens
        if generation_config.temperature is not None:
            gen_kwargs["temperature"] = generation_config.temperature
            gen_kwargs["do_sample"] = generation_config.temperature > 0.0

        inputs = self._processor.apply_chat_template(
            messages,
            audio=audio_arrays or None,
            sampling_rate=_GEMMA4_TARGET_SAMPLE_RATE,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        output_ids = self._model.generate(**inputs, **gen_kwargs)
        text = self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )[0]
        return _strip_think(text)


_TRANSFORMERS_ENGINE = _TransformersEngine()


def _call_transformers_inference(
    *,
    messages: list[dict[str, Any]],
    audio_inputs: list[HttpAudioInput],
    generation_config: HttpGenerationConfig,
) -> str:
    return _TRANSFORMERS_ENGINE.generate(
        messages=messages,
        audio_inputs=audio_inputs,
        generation_config=generation_config,
    )


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
        if CFG.backend == "transformers":
            text = _call_transformers_inference(
                messages=messages,
                audio_inputs=item.audio_inputs,
                generation_config=item.generation_config,
            )
        else:
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

    A single vLLM server is fast enough that per-request latency dominates
    unless requests are in flight concurrently; the stub path, the
    transformers backend (no server round-trip to hide behind concurrency,
    and GPU-serialised regardless), and single-item batches run inline.

    Returns
    -------
    list[PredictionsV1ResponseItem]
        Per-sample responses, in the input order of ``raw_requests``.
    """

    if (
        CFG.stub
        or CFG.backend == "transformers"
        or CFG.max_concurrency <= 1
        or len(raw_requests) <= 1
    ):
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

    host = os.environ.get("GEMMA4_ADAPTER_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
