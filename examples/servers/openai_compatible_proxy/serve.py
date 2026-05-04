"""BEANS-Next OpenAI-compatible proxy HTTP launcher (``predictions_v1`` contract).

This launcher is part of the BEANS-Next serving kit. It intentionally keeps an
isolated environment and does not import ``beans_next``.

Modes
-----
Stub mode (``OPENAI_PROXY_STUB=1``):
    Returns deterministic hashed strings. Used for contract conformance
    testing and CPU-only validation.

Real proxy mode (``OPENAI_PROXY_STUB=0``):
    Proxies each request item to an OpenAI-compatible Chat Completions
    upstream (``POST /v1/chat/completions``) and translates the response
    into a ``predictions_v1`` response item.

Key env vars
------------
OPENAI_PROXY_SYSTEM_PROMPT
    System message prepended to every upstream request when the incoming
    messages contain no ``system`` role entry.  Defaults to a bioacoustics
    annotator persona. Set to an empty string to disable entirely.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"

_AUDIO_TAG_PATTERN = re.compile(r"<Audio><AudioHere></Audio>")
_DEFAULT_OPENAI_CFG_PATH = os.path.expanduser("~/.config/openai/cfg")
_DEFAULT_GEMINI_CFG_PATH = os.path.expanduser("~/.config/gemini/cfg")

_DEFAULT_SYSTEM_PROMPT = (
    "You are a bioacoustics annotator. Reply to the question by listening to the "
    "audio. If you aren't sure, give your best guess."
)


def _count_audio_placeholders(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(_AUDIO_TAG_PATTERN.findall(content))
    return total


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


def _read_api_key_from_cfg(path: str, accepted_keys: set[str]) -> str | None:
    """Return an API key value from a simple cfg file.

    The proxy launcher supports a minimal "dotenv-like" format to keep secrets
    out of shells and job scripts. OpenAI defaults to ``~/.config/openai/cfg``;
    Gemini defaults to ``~/.config/gemini/cfg``.

    Supported formats (first match wins):

    - ``OPENAI_API_KEY=sk-...``
    - ``OPENAI_API_KEY: sk-...``
    - ``GEMINI_API_KEY=...``
    - ``GOOGLE_API_KEY=...``
    - ``api_key=sk-...`` (case-insensitive)
    - ``api_key: sk-...`` (case-insensitive)
    - ``sk-...`` / ``AIza...`` as a bare token line

    Blank lines and lines starting with ``#`` are ignored.

    Parameters
    ----------
    path
        Path to the cfg file.
    accepted_keys
        Environment-style key names to accept from the cfg file.

    Returns
    -------
    str | None
        The API key if found, else ``None``.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Strip optional `export ` prefix used in shell env files.
        if line.lower().startswith("export "):
            line = line[7:].strip()

        m = re.match(r"^(?P<k>[A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?P<v>.+?)\s*$", line)
        if not m:
            if ":" not in line and "=" not in line:
                return line.strip("'").strip('"')
            continue

        key = m.group("k").strip()
        val = m.group("v").strip().strip("'").strip('"')
        if not val:
            continue

        if key in accepted_keys or key.lower() == "api_key":
            return val

    return None


def _is_gemini_upstream() -> bool:
    """Return whether the configured upstream appears to be Gemini.

    Returns
    -------
    bool
        True when the base URL or model id matches Gemini conventions.
    """

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip().lower()
    model = os.environ.get("OPENAI_MODEL", "").strip().lower()
    return "generativelanguage.googleapis.com" in base_url or model.startswith(
        "gemini-"
    )


def _canonicalize_base64_wav(
    data_b64: str,
    *,
    clip_seconds: float | None,
) -> str:
    """Best-effort: re-encode to canonical PCM16 WAV (optionally clipped).

    Some upstreams reject certain valid WAV encodings. When enabled, this normalizes the
    payload to a standard PCM16 WAV and can clip to the first N seconds to bound request
    size and avoid edge-case decoder failures upstream.

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


def _ensure_openai_api_key_loaded() -> None:
    """Populate OPENAI_API_KEY env var from provider cfg files when missing."""

    if os.environ.get("OPENAI_API_KEY"):
        return

    if _is_gemini_upstream():
        for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = os.environ.get(env_name, "").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
                return

        cfg_path = os.environ.get("GEMINI_CFG_PATH", "").strip()
        cfg_path = cfg_path or _DEFAULT_GEMINI_CFG_PATH
        key = _read_api_key_from_cfg(cfg_path, {"GEMINI_API_KEY", "GOOGLE_API_KEY"})
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return

    cfg_path = os.environ.get("OPENAI_CFG_PATH", "").strip() or _DEFAULT_OPENAI_CFG_PATH
    key = _read_api_key_from_cfg(cfg_path, {"OPENAI_API_KEY"})
    if key:
        os.environ["OPENAI_API_KEY"] = key


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
    This model allows extra keys so the launcher can ignore unknown generation
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
    return f"openai_proxy_stub:{digest}"


def _content_with_audio_placeholders_replaced(
    content: str,
    audio_items: list[dict[str, Any]],
    start_idx: int,
) -> tuple[list[dict[str, Any]], int]:
    parts: list[dict[str, Any]] = []
    pos = 0
    audio_idx = start_idx

    for match in _AUDIO_TAG_PATTERN.finditer(content):
        prefix = content[pos : match.start()]
        if prefix:
            parts.append({"type": "text", "text": prefix})
        if audio_idx >= len(audio_items):
            raise ValueError("not enough audio_inputs for audio placeholders")
        parts.append(audio_items[audio_idx])
        audio_idx += 1
        pos = match.end()

    suffix = content[pos:]
    if suffix:
        parts.append({"type": "text", "text": suffix})
    return parts, audio_idx


def _inject_audio_items(
    messages: list[dict[str, Any]],
    audio_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    msgs = [dict(m) for m in messages]
    audio_idx = 0

    for msg in msgs:
        content = msg.get("content")
        if not isinstance(content, str) or not _AUDIO_TAG_PATTERN.search(content):
            continue
        parts, audio_idx = _content_with_audio_placeholders_replaced(
            content,
            audio_items,
            audio_idx,
        )
        msg["content"] = parts

    if audio_idx:
        if audio_idx != len(audio_items):
            raise ValueError("unused audio_inputs after placeholder replacement")
        return msgs

    last_user = next(
        (i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"),
        None,
    )
    if last_user is None:
        msgs.append({"role": "user", "content": []})
        last_user = len(msgs) - 1

    existing = msgs[last_user].get("content")
    content_list: list[dict[str, Any]]
    if isinstance(existing, list):
        content_list = [*existing]
    elif isinstance(existing, str):
        content_list = [{"type": "text", "text": existing}]
    else:
        content_list = []
    # If no explicit <Audio> placeholder exists, choose a stable default ordering.
    #
    # Gemini and Qwen guidance tends to work best when multimodal content comes
    # first and the textual instruction follows. OpenAI models are usually fine
    # either way, so we only change ordering for Gemini upstreams.
    if _is_gemini_upstream():
        content_list = [*audio_items, *content_list]
    else:
        content_list.extend(audio_items)
    msgs[last_user]["content"] = content_list
    return msgs


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
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            joined = "".join(parts).strip()
            if joined:
                return joined
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
class _ProxyConfig:
    """Configuration for the OpenAI-compatible proxy launcher.

    Attributes
    ----------
    stub
        If ``True``, enable deterministic stub predictions (no upstream calls).
    max_batch_size
        Maximum number of request items accepted in a single `/predict` call.
    max_concurrency
        Maximum number of upstream Chat Completions calls in flight per
        `/predict` batch.
    supports_batching
        Whether the launcher supports multiple request items per `/predict`.
    schema_versions
        Wire schema versions advertised by `/info`.
    audio_payload_types
        Supported `predictions_v1` audio payload types advertised by `/info`.
    launcher_name
        Human-readable launcher name advertised by `/info`.
    upstream_model
        Upstream model id sent to the OpenAI-compatible Chat Completions API.
    model_revision
        Revision string advertised by `/info` for reproducibility.
    base_url
        Upstream base URL (without path), e.g. ``http://127.0.0.1:4000``.
    auth_header
        Header name used to send upstream authentication.
    api_key
        Upstream API key value, or ``None`` for unauthenticated upstreams.
    timeout_sec
        Upstream request timeout in seconds.
    retries
        Number of retry attempts for transient upstream failures.
    system_prompt
        System message prepended to every request when the incoming messages
        contain no ``system`` role entry.  Defaults to
        ``_DEFAULT_SYSTEM_PROMPT``.  Set ``OPENAI_PROXY_SYSTEM_PROMPT=""``
        to disable entirely.
    """

    stub: bool
    max_batch_size: int
    max_concurrency: int
    supports_batching: bool
    schema_versions: list[str]
    audio_payload_types: list[str]

    launcher_name: str
    upstream_model: str
    model_revision: str

    base_url: str
    auth_header: str
    api_key: str | None
    timeout_sec: float
    retries: int
    system_prompt: str | None


def _load_config() -> _ProxyConfig:
    _ensure_openai_api_key_loaded()

    stub = _get_bool_env("OPENAI_PROXY_STUB", True)
    max_batch = _get_int_env("OPENAI_PROXY_MAX_BATCH_SIZE", 32)
    max_concurrency = _get_int_env("OPENAI_PROXY_MAX_CONCURRENCY", 1)
    supports_batching = True
    schema_versions = [PREDICTIONS_V1]
    audio_payload_types = ["base64_wav", "file_path", "file_url"]

    launcher_name = "beans-next-openai-compatible-proxy"
    upstream_model = os.environ.get("OPENAI_MODEL", "openai-compatible/unknown")
    model_revision = os.environ.get("OPENAI_MODEL_REVISION", "unknown")

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    auth_header = os.environ.get("OPENAI_AUTH_HEADER", "Authorization")
    api_key = os.environ.get("OPENAI_API_KEY")
    timeout_sec = _get_float_env("OPENAI_PROXY_TIMEOUT_SEC", 120.0)
    retries = _get_int_env("OPENAI_PROXY_RETRIES", 2)

    raw_system_prompt = os.environ.get(
        "OPENAI_PROXY_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT
    )
    system_prompt = raw_system_prompt if raw_system_prompt.strip() else None

    return _ProxyConfig(
        stub=stub,
        max_batch_size=max_batch,
        max_concurrency=max(1, max_concurrency),
        supports_batching=supports_batching,
        schema_versions=schema_versions,
        audio_payload_types=audio_payload_types,
        launcher_name=launcher_name,
        upstream_model=upstream_model,
        model_revision=model_revision,
        base_url=base_url,
        auth_header=auth_header,
        api_key=api_key,
        timeout_sec=timeout_sec,
        retries=retries,
        system_prompt=system_prompt,
    )


CFG = _load_config()

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

_client_headers: dict[str, str] = {"Content-Type": "application/json"}
if CFG.api_key:
    auth_is_bearer = CFG.auth_header.lower() == "authorization"
    key_has_bearer_prefix = CFG.api_key.lower().startswith("bearer ")
    if auth_is_bearer and not key_has_bearer_prefix:
        _client_headers[CFG.auth_header] = f"Bearer {CFG.api_key}"
    else:
        _client_headers[CFG.auth_header] = CFG.api_key

HTTP = httpx.Client(
    timeout=httpx.Timeout(CFG.timeout_sec),
    headers=_client_headers,
)

app = FastAPI(title="BEANS-Next OpenAI-compatible proxy launcher", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    mode = "stub" if CFG.stub else "proxy"
    load_status = "stub" if CFG.stub else "ready"
    return {"status": "ok", "mode": mode, "load_status": load_status}


@app.get("/info")
def info() -> InfoResponse:
    model_name = CFG.upstream_model if not CFG.stub else "stub"
    return InfoResponse(
        name=CFG.launcher_name,
        model=model_name,
        model_revision=CFG.model_revision if not CFG.stub else "stub",
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
    if not CFG.base_url:
        raise RuntimeError("OPENAI_BASE_URL is required when OPENAI_PROXY_STUB=0")

    # Convert BEANS-Next messages to OpenAI-compatible messages.
    openai_messages: list[dict[str, Any]] = [
        {"role": m.get("role", ""), "content": m.get("content", "")} for m in messages
    ]

    # Prepend system prompt when the request has no system message already.
    has_system = any(m.get("role") == "system" for m in openai_messages)
    if CFG.system_prompt and not has_system:
        openai_messages = [
            {"role": "system", "content": CFG.system_prompt},
            *openai_messages,
        ]

    clip_seconds = _get_optional_float_env("OPENAI_PROXY_MAX_AUDIO_SECONDS")
    canonicalize = _get_bool_env("OPENAI_PROXY_CANONICALIZE_WAV", default=False)

    # Gemini is more sensitive to WAV encoding / long audio payloads than OpenAI in
    # our BEANS-Zero runs. Unless explicitly overridden, apply the same 30s clip
    # policy used for ChatGPT troubleshooting and canonicalize to PCM16 WAV.
    if _is_gemini_upstream():
        if clip_seconds is None:
            clip_seconds = 30.0
        if not canonicalize:
            canonicalize = True

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
        audio_items.append(
            {
                "type": "input_audio",
                "input_audio": {"data": data_b64, "format": "wav"},
            }
        )
    if audio_items:
        openai_messages = _inject_audio_items(openai_messages, audio_items)

    body: dict[str, Any] = {
        "model": CFG.upstream_model,
        "messages": openai_messages,
    }

    gen = generation_config.model_dump(mode="json", exclude_none=True)
    if "max_tokens" in gen:
        body["max_tokens"] = gen["max_tokens"]
    if "temperature" in gen:
        body["temperature"] = gen["temperature"]

    # Gemini OpenAI-compat supports `reasoning_effort`, but we should not force a
    # default here because users may want to rely on upstream defaults or test
    # different settings. Only send it when explicitly requested.
    if _is_gemini_upstream() and "reasoning_effort" not in body:
        raw_effort = os.environ.get("GEMINI_REASONING_EFFORT")
        if isinstance(raw_effort, str):
            raw_effort = raw_effort.strip()
        if raw_effort:
            body["reasoning_effort"] = raw_effort

    # Gemini thinking models (2.5 Pro, 3.1 Pro-preview, etc.) spend thinking
    # tokens from the max_tokens budget; a small task-level value leaves zero
    # tokens for the actual response. GEMINI_MIN_MAX_TOKENS (default 1024)
    # overrides any lower value coming from the task generation_config.
    if _is_gemini_upstream():
        min_max_tokens = _get_int_env("GEMINI_MIN_MAX_TOKENS", 1024)
        mt = body.get("max_tokens")
        if not isinstance(mt, int) or mt < min_max_tokens:
            body["max_tokens"] = min_max_tokens

    url = f"{CFG.base_url.rstrip('/')}{_CHAT_COMPLETIONS_PATH}"

    last_exc: Exception | None = None
    for _ in range(1 + max(0, CFG.retries)):
        try:
            resp = HTTP.post(url, json=body)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise ValueError("upstream response is not a JSON object")
            try:
                return _extract_chat_completion_text(payload)
            except ValueError as exc:
                # Gemini preview models occasionally return a syntactically valid
                # Chat Completions response with an assistant message that has no
                # content fields (completion_tokens=0). Treat as transient and retry.
                if _is_gemini_upstream():
                    try:
                        choices = payload.get("choices")
                        if isinstance(choices, list) and choices:
                            first = choices[0]
                            if isinstance(first, dict):
                                msg = first.get("message")
                                if (
                                    isinstance(msg, dict)
                                    and msg.get("role") == "assistant"
                                ):
                                    if (
                                        msg.get("content") is None
                                        and first.get("text") is None
                                    ):
                                        last_exc = exc
                                        continue
                    except Exception:  # noqa: BLE001
                        pass

                # Surface enough context to debug OpenAI-compatible variations
                # (especially Gemini previews) without dumping huge payloads.
                keys = sorted(payload.keys())
                snippet = ""
                try:
                    snippet = resp.text[:500]
                except Exception:  # noqa: BLE001
                    snippet = ""
                raise ValueError(
                    f"{exc}; payload_keys={keys!r}; payload_snippet={snippet!r}"
                ) from exc
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
            predictions=[],
            error=f"invalid request item: {exc}",
        )

    if len(item.messages) == 0:
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error="missing or empty messages",
        )

    audio_tags = _count_audio_placeholders(
        [{"role": m.role, "content": m.content} for m in item.messages]
    )
    if audio_tags > 0 and audio_tags != len(item.audio_inputs):
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error=(
                f"audio_inputs length ({len(item.audio_inputs)}) does not match "
                "number of '<Audio><AudioHere></Audio>' tags in messages "
                f"({audio_tags})"
            ),
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
            predictions=[],
            finish_reason=None,
            usage=None,
            latency_sec=latency,
            error=_http_error_to_message(exc),
        )


def _process_request_batch(
    raw_requests: list[dict[str, Any]],
) -> list[PredictionsV1ResponseItem]:
    """Process a validated request batch, optionally with concurrent upstream calls.

    Parameters
    ----------
    raw_requests
        Request items already validated as objects with unique ``sample_id`` values.

    Returns
    -------
    list[PredictionsV1ResponseItem]
        Per-sample responses. Ordering is normalized by caller.
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

    responses = _process_request_batch(raw_requests)

    # Response order is not meaningful; match by sample_id.
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

    host = os.environ.get("OPENAI_PROXY_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
