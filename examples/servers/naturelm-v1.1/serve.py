"""BEANS-Next NatureLM-audio 1.1 HTTP launcher (``predictions_v1`` contract).

This launcher is intentionally self-contained and does not import `beans_next`.
It implements the mandatory BEANS-Next HTTP endpoints:

- `POST /predict`
- `GET /info`
- `GET /health`

Weights are loaded from a GCS checkpoint directory (``NATURELM_GCS_CHECKPOINT_URI``).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

# --- Contract constants (aligned with beans_next.api.http_schemas) ---

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"

# --- Launcher identity ---

LAUNCHER_NAME: str = "beans-next-naturelm-v1.1"

# GCS checkpoint directory containing the model weights.
# Example:
#   export NATURELM_GCS_CHECKPOINT_URI="gs://foundation-models/naturelm-audio-1.1/base_model/1290000"
GCS_CHECKPOINT_URI: str = os.environ.get("NATURELM_GCS_CHECKPOINT_URI", "").strip()

# Optional HuggingFace identity for /info when not using a GCS checkpoint.
# (Used by tests and by setups that want /info to reflect gated repo identity.)
HF_REPO_ID: str = os.environ.get("NATURELM_HF_REPO_ID", "").strip()
HF_REVISION: str = os.environ.get("NATURELM_HF_REVISION", "").strip()

# This launcher can be used purely for contract conformance.
STUB_MODE: bool = os.environ.get("NATURELM_STUB_MODE", "0").strip() == "1"

# Batch sizing (contract requires 413 above this).
MAX_BATCH_SIZE: int = int(os.environ.get("NATURELM_MAX_BATCH_SIZE", "8"))

AUDIO_PAYLOAD_TYPES: list[str] = ["base64_wav", "file_path", "file_url"]
SUPPORTS_BATCHING: bool = True
SCHEMA_VERSIONS: list[str] = [PREDICTIONS_V1]

AUDIO_TAG_PATTERN = re.compile(r"<Audio><AudioHere></Audio>")


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
    # Optional audio clip length override. BEANS-Next runner injects this for
    # BEANS-Zero subsets (e.g. ESC-50 = 5s).
    max_length_seconds: int | None = None


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
    audio_sample_rate_hz: int | None = None
    max_batch_size: int
    supports_batching: bool
    schema_versions: list[str]
    load_status: str | None = None
    loading_stage: str | None = None
    loading_elapsed_sec: float | None = None
    last_error: str | None = None


def _gcs_checkpoint_basename(gcs_uri: str) -> str:
    """Derive a stable revision string from a GCS checkpoint URI.

    Parameters
    ----------
    gcs_uri
        Directory-like GCS URI pointing at a specific checkpoint.

    Returns
    -------
    str
        Basename of the URI (last path component), with any trailing slash removed.
    """
    return gcs_uri.rstrip("/").rsplit("/", 1)[-1]


def _deterministic_stub_prediction(
    sample_id: str, item: PredictionsV1RequestItem
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
    return f"naturelm_v1_1_stub:{digest}"


def _count_audio_tags(messages: list[HttpChatMessage]) -> int:
    total = 0
    for msg in messages:
        total += len(AUDIO_TAG_PATTERN.findall(msg.content))
    return total


def _decode_audio_input(inp: HttpAudioInput) -> tuple[Any, int]:
    """Decode audio into (array, sample_rate).

    Returns
    -------
    tuple[Any, int]
        `(audio_array, sample_rate)`; array is typically float32 numpy array.

    Raises
    ------
    ValueError
        If the payload cannot be decoded.
    """
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "audio decoding requires the optional dependency 'soundfile'. "
            "Install server requirements or disable audio validation."
        ) from exc

    if inp.payload_type == "base64_wav":
        try:
            wav_bytes = base64.b64decode(inp.data)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid base64_wav payload: {exc}") from exc
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        return data, int(sr)

    if inp.payload_type == "file_path":
        data, sr = sf.read(inp.data, dtype="float32", always_2d=False)
        return data, int(sr)

    if inp.payload_type == "file_url":
        try:
            resp = requests.get(inp.data, timeout=30)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"failed to download file_url: {exc}") from exc
        data, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=False)
        return data, int(sr)

    raise ValueError(f"unsupported audio payload_type: {inp.payload_type!r}")


def _ensure_gcs_checkpoint_available(gcs_uri: str) -> str:
    """Download a GCS checkpoint dir to a local cache and return its path.

    Returns
    -------
    str
        Path to the local directory containing the downloaded checkpoint files.

    Raises
    ------
    HTTPException
        If the URI is invalid, GCS access fails, or `gcsfs` is unavailable.

    Notes
    -----
    - This expects a *directory-like* GCS URI (prefix) that contains the model
      files needed by `NatureLM.from_pretrained(<local_dir>)`.
    - Download uses `gcsfs` if installed. If your environment authenticates via
      GCE metadata, workload identity, or ADC, `gcsfs` will usually “just work”.
    """
    try:
        import gcsfs  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                "GCS checkpoint requested but 'gcsfs' is not installed in this "
                "launcher venv. Install it (launcher-only) or use HuggingFace "
                "weights instead."
            ),
        ) from exc

    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail=f"invalid GCS uri: {gcs_uri!r}")

    # Local cache root (launcher-only).
    cache_root = os.environ.get(
        "NATURELM_GCS_CACHE_DIR",
        os.path.expanduser("~/.cache/beans-next/naturelm-v1.1-gcs"),
    )
    os.makedirs(cache_root, exist_ok=True)

    # Make a stable-ish path from the URI.
    safe = gcs_uri.removeprefix("gs://").replace("/", "__")
    local_dir = os.path.join(cache_root, safe)
    done_marker = os.path.join(local_dir, ".download_complete")

    if os.path.exists(done_marker):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)

    fs = gcsfs.GCSFileSystem()
    prefix = gcs_uri.removeprefix("gs://")
    # gcsfs uses "bucket/path" format internally.
    objects = fs.find(prefix)
    if not objects:
        raise HTTPException(
            status_code=503, detail=f"no objects found under {gcs_uri!r}"
        )

    # Download every object into local_dir, preserving relative structure.
    for obj in objects:
        rel = obj[len(prefix) :].lstrip("/")
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        fs.get(obj, dest)

    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("ok\n")

    return local_dir


def _maybe_load_naturelm(snapshot_path: str) -> Any | None:  # noqa: ANN401
    """Best-effort real model import+load.

    Parameters
    ----------
    snapshot_path
        Local checkpoint directory (downloaded from GCS).

    Returns
    -------
    Any | None
        Loaded NatureLM model instance, or `None` if inference is disabled.

    Raises
    ------
    HTTPException
        If inference is enabled but the NatureLM package cannot be imported or
        the model fails to load.
    ModuleNotFoundError
        If the esp-research NatureLM project directory cannot be located.
    """
    if os.environ.get("NATURELM_ENABLE_INFERENCE", "0").strip() != "1":
        return None

    try:
        import sys
        from pathlib import Path

        import torch  # type: ignore[import-not-found]

        # Prefer esp-research's authoritative NatureLM-audio-v1.5 implementation.
        esp_research_root = os.environ.get("ESP_RESEARCH_LOCAL_PATH", "").strip()
        project_dir_env = os.environ.get(
            "ESP_RESEARCH_NATURELM_PROJECT_DIR", ""
        ).strip()
        project_dir = (
            Path(project_dir_env)
            if project_dir_env
            else (
                Path(esp_research_root) / "projects" / "NatureLM-audio-v1.5"
                if esp_research_root
                else None
            )
        )
        if project_dir is None or not project_dir.exists():
            raise ModuleNotFoundError(
                "esp-research NatureLM-audio-v1.5 project not found. "
                "Set ESP_RESEARCH_LOCAL_PATH (preferred) or "
                "ESP_RESEARCH_NATURELM_PROJECT_DIR."
            )
        project_dir_str = str(project_dir)
        if project_dir_str not in sys.path:
            sys.path.insert(0, project_dir_str)

        # ------------------------------------------------------------------
        # Compatibility shim: esp-research expects `from esp_data.io import read_yaml`
        # but some esp-data builds do not export it. Define it dynamically so that
        # esp-research imports succeed without pinning esp-data.
        # ------------------------------------------------------------------
        try:
            import esp_data.io as _esp_io  # type: ignore[import-not-found]

            if not hasattr(_esp_io, "read_yaml"):
                import yaml  # type: ignore[import-not-found]

                def _read_yaml(path: object) -> object:
                    p = _esp_io.anypath(path)  # supports local + cloud paths
                    with p.open("r") as f:
                        return yaml.safe_load(f)

                _esp_io.read_yaml = _read_yaml
        except Exception:
            # Best-effort only; if esp-data isn't importable, downstream imports
            # will raise an actionable error.
            pass

        from naturelm import GenerationConfig  # type: ignore[import-not-found]
        from naturelm import NatureLM as NatureLMModel
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                "NatureLM-audio runtime is not available. "
                "This launcher expects esp-research's NatureLM-audio-v1.5 "
                "implementation on PYTHONPATH. "
                "Set ESP_RESEARCH_LOCAL_PATH or ESP_RESEARCH_NATURELM_PROJECT_DIR, "
                "or run with NATURELM_STUB_MODE=1 for conformance-only mode. "
                f"Root cause: {exc!r}"
            ),
        ) from exc

    # Note: the NatureLM-audio v1.x codebase historically expects a concrete torch
    # device ("cuda", "cuda:0", "cpu") rather than HF-style device maps.
    device = os.environ.get("NATURELM_DEVICE", "cuda:0")
    try:
        # esp-research v1.5 loads from a checkpoint directory (not a transformers repo).
        # The gated HF snapshot is expected to contain the checkpoint layout.
        # Override config path if needed.
        cfg_path_env = os.environ.get("NATURELM_CONFIG_PATH", "").strip()
        if cfg_path_env:
            cfg_path = Path(cfg_path_env)
        else:
            snap = Path(snapshot_path)
            # Different checkpoints use different names; prefer config.json.
            cfg_path = snap / "config.json"
            if not cfg_path.exists():
                cfg_path = snap / "model.json"
        snap = Path(snapshot_path)
        ckpt_dir = snap / "checkpoint"
        if not ckpt_dir.exists():
            # GCS checkpoints commonly store weights under `model/` with
            # subdirectories like `llm/`, `audio_encoder/`, etc. The esp-research
            # loader expects those as direct children of the checkpoint dir.
            model_dir = snap / "model"
            ckpt_dir = model_dir if model_dir.exists() else snap

        model = NatureLMModel.from_checkpoint_dir(
            checkpoint_dir=ckpt_dir,
            config=cfg_path,
        ).to(torch.device(device)).eval()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"weights downloaded but NatureLM model failed to load: {exc}",
        ) from exc

    # Align tokenizer padding behavior with the reference inference script.
    try:
        model.llama_tokenizer.pad_token_id = model.llama_tokenizer.eos_token_id
        model.llama_model.generation_config.pad_token_id = (
            model.llama_tokenizer.pad_token_id
        )
    except Exception:
        # If the internal objects differ, this is best-effort only.
        pass

    # Audio handling:
    # - resampling/pad/truncate is handled inside the esp-research model helpers;
    # - we still keep the max-length control (per BEANS request) for parity with v1.0.
    sample_rate = int(os.environ.get("NATURELM_SAMPLE_RATE", "16000"))
    max_len_sec = int(os.environ.get("NATURELM_MAX_AUDIO_SECONDS", "10"))
    return {
        "model": model,
        "generation_config_cls": GenerationConfig,
        "sample_rate": sample_rate,
        "default_max_length_seconds": max_len_sec,
        "device": device,
    }


@dataclass
class _LauncherState:
    """In-memory launcher state used to cache HF checks and downloads."""

    resolved_revision: str | None = None
    snapshot_path: str | None = None
    model: Any | None = None
    last_error: str | None = None
    loading: bool = False
    loading_started_at: float | None = None
    loading_stage: str | None = None
    loading_stage_started_at: float | None = None
    loading_thread: threading.Thread | None = None


_state = _LauncherState()
_lock = threading.Lock()


def _ensure_ready_or_raise() -> None:
    if STUB_MODE:
        return

    # Never hold `_lock` while downloading weights or loading the model. That
    # would block `/health` and/or request handling under heavy init.
    with _lock:
        if _state.model is not None or _state.snapshot_path is not None:
            # Already passed download stage; still report last error if present.
            if _state.last_error is None:
                return

    resolved_revision: str
    snapshot_path: str
    model: Any | None
    try:
        if not GCS_CHECKPOINT_URI:
            raise HTTPException(
                status_code=503,
                detail=(
                    "NATURELM_GCS_CHECKPOINT_URI is not set. "
                    "Export it to point at a GCS checkpoint directory, e.g. "
                    "gs://foundation-models/naturelm-audio-1.1/base_model/1290000"
                ),
            )

        with _lock:
            _state.loading_stage = "downloading_gcs_checkpoint"
            _state.loading_stage_started_at = time.time()
        resolved_revision = _gcs_checkpoint_basename(GCS_CHECKPOINT_URI)
        snapshot_path = _ensure_gcs_checkpoint_available(GCS_CHECKPOINT_URI)

        with _lock:
            _state.loading_stage = "loading_model_runtime"
            _state.loading_stage_started_at = time.time()
        model = _maybe_load_naturelm(snapshot_path)
    except HTTPException as exc:
        with _lock:
            _state.last_error = str(exc.detail)
        raise

    with _lock:
        _state.resolved_revision = resolved_revision
        _state.snapshot_path = snapshot_path
        _state.model = model
        _state.last_error = None
        _state.loading_stage = None
        _state.loading_stage_started_at = None


def _start_async_load() -> None:
    """Start model download+load in a background thread.

    The server's `/health` endpoint should remain responsive even while
    checkpoint/model initialization is in progress.
    """
    if STUB_MODE:
        return

    with _lock:
        # Avoid double-starting while preserving the final terminal state.
        if _state.model is not None or _state.snapshot_path is not None:
            return
        if _state.loading:
            return
        _state.loading = True
        _state.loading_started_at = time.time()
        _state.last_error = None

    def _bg() -> None:
        try:
            _ensure_ready_or_raise()
        except HTTPException as exc:
            with _lock:
                _state.last_error = str(exc.detail)
        except Exception as exc:  # noqa: BLE001
            # Preserve a best-effort error message so `/health` surfaces failures
            # that are not expressed as HTTPException (e.g. unexpected runtime
            # import/load errors inside esp-research / model code).
            with _lock:
                _state.last_error = f"unexpected init failure: {exc!r}"
        finally:
            with _lock:
                _state.loading = False

    t = threading.Thread(target=_bg, daemon=True)
    with _lock:
        _state.loading_thread = t
    t.start()


def _predict_stub(
    sample_id: str, item: PredictionsV1RequestItem
) -> PredictionsV1ResponseItem:
    audio_tags = _count_audio_tags(item.messages)
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

    if os.environ.get("NATURELM_STUB_VALIDATE_AUDIO", "0").strip() == "1":
        try:
            for inp in item.audio_inputs:
                _decode_audio_input(inp)
        except Exception as exc:  # noqa: BLE001
            return PredictionsV1ResponseItem(
                sample_id=sample_id,
                predictions=[],
                error=f"failed to decode audio_inputs for stub validation: {exc}",
            )

    pred = _deterministic_stub_prediction(sample_id, item)
    return PredictionsV1ResponseItem(
        sample_id=sample_id,
        predictions=[pred],
        finish_reason="stop",
        usage=HttpUsage(prompt_tokens=None, completion_tokens=None),
        latency_sec=0.0,
        error=None,
    )


def _first_user_prompt(messages: list[HttpChatMessage]) -> str:
    for msg in messages:
        if msg.role == "user":
            return msg.content
    return messages[0].content


def _predict_real(
    sample_id: str, item: PredictionsV1RequestItem, model_bundle: dict[str, Any]
) -> PredictionsV1ResponseItem:
    started = time.perf_counter()

    audio_tags = _count_audio_tags(item.messages)
    if audio_tags != len(item.audio_inputs):
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error=(
                f"audio_inputs length ({len(item.audio_inputs)}) does not match "
                "number of '<Audio><AudioHere></Audio>' tags in messages "
                f"({audio_tags})"
            ),
        )

    if len(item.audio_inputs) == 0:
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error="audio_inputs must be non-empty",
        )

    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error=f"missing NatureLM runtime deps in venv: {exc}",
        )

    # Decode each audio input. These are treated as multiple "chunks" for a
    # *single* prompt, spliced into the prompt in order of `<AudioHere>` tags.
    decoded_audios: list[Any] = []
    decoded_srs: list[int] = []
    for idx, inp in enumerate(item.audio_inputs):
        try:
            audio_raw, input_sr = _decode_audio_input(inp)
        except Exception as exc:  # noqa: BLE001
            return PredictionsV1ResponseItem(
                sample_id=sample_id,
                predictions=[],
                error=f"failed to decode audio_inputs[{idx}]: {exc}",
            )
        decoded_audios.append(audio_raw)
        decoded_srs.append(int(input_sr))

    # NatureLM v1.x expects "<AudioHere>" placeholders; BEANS prompts include
    # "<Audio><AudioHere></Audio>" which is compatible.

    model = model_bundle["model"]
    generation_config_cls = model_bundle["generation_config_cls"]
    sample_rate = int(model_bundle["sample_rate"])
    default_max_length_seconds = int(model_bundle["default_max_length_seconds"])
    device = model_bundle["device"]

    try:
        max_length_seconds = (
            int(item.generation_config.max_length_seconds)
            if item.generation_config.max_length_seconds is not None
            else default_max_length_seconds
        )

        # esp-research NatureLM-audio-v1.5 expects a single waveform per request.
        # For BEANS, we currently support exactly 1 audio input per sample.
        if len(decoded_audios) != 1:
            return PredictionsV1ResponseItem(
                sample_id=sample_id,
                predictions=[],
                latency_sec=time.perf_counter() - started,
                error=(
                    "NatureLM v1.1 launcher currently supports exactly 1 audio input "
                    "per request item"
                ),
            )

        # Resample/truncate/pad on our side to enforce BEANS per-subset clip length.
        wav = np.asarray(decoded_audios[0], dtype=np.float32)
        sr_in = int(decoded_srs[0])
        if sr_in != sample_rate:
            try:
                import resampy  # type: ignore[import-not-found]
            except Exception as exc:
                return PredictionsV1ResponseItem(
                    sample_id=sample_id,
                    predictions=[],
                    latency_sec=time.perf_counter() - started,
                    error=(
                        "resampy is required for resampling "
                        f"(sr_in={sr_in} -> {sample_rate}): {exc}"
                    ),
                )
            wav = resampy.resample(wav, sr_in, sample_rate).astype(np.float32)

        target_len = int(max_length_seconds * sample_rate)
        if wav.shape[0] < target_len:
            pad = target_len - wav.shape[0]
            wav = np.pad(wav, (0, pad), mode="constant")
        elif wav.shape[0] > target_len:
            wav = wav[:target_len]

        # Move to torch and call model.generate (esp-research implementation).
        raw_wav = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
        padding_mask = torch.zeros_like(raw_wav, dtype=torch.bool)
        raw_wav = raw_wav.to(device)
        padding_mask = padding_mask.to(device)

        # Translate contract generation_config → NatureLM generate cfg.
        # Fall back to conservative deterministic defaults.
        max_new_tokens = (
            item.generation_config.max_tokens
            if item.generation_config.max_tokens is not None
            else int(os.environ.get("NATURELM_MAX_NEW_TOKENS", "128"))
        )
        temperature = (
            float(item.generation_config.temperature)
            if item.generation_config.temperature is not None
            else float(os.environ.get("NATURELM_TEMPERATURE", "0.0"))
        )

        gen_cfg = generation_config_cls(
            max_new_tokens=int(max_new_tokens),
            num_beams=int(os.environ.get("NATURELM_NUM_BEAMS", "1")),
            do_sample=os.environ.get("NATURELM_DO_SAMPLE", "0").strip() == "1",
            min_length=1,
            temperature=float(temperature),
            repetition_penalty=float(
                os.environ.get("NATURELM_REPETITION_PENALTY", "1.0")
            ),
            length_penalty=float(os.environ.get("NATURELM_LENGTH_PENALTY", "1.0")),
        )

        # esp-research NatureLM expects a batch of conversations (list[list[dict]]).
        # We pass BEANS messages through verbatim (raw content; no role-prefix
        # rewriting).
        conversations = [[m.model_dump() for m in item.messages]]
        pred_text = model.generate(
            audio=raw_wav,
            padding_mask=padding_mask,
            messages=conversations,
            generation_config=gen_cfg,
        )[0]
    except Exception as exc:  # noqa: BLE001
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            latency_sec=time.perf_counter() - started,
            error=f"NatureLM inference failed: {exc}",
        )

    return PredictionsV1ResponseItem(
        sample_id=sample_id,
        predictions=[pred_text],
        finish_reason="stop",
        latency_sec=time.perf_counter() - started,
    )


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

    if STUB_MODE:
        return _predict_stub(sample_id, item)

    if _state.model is None:
        if _state.loading:
            stage = _state.loading_stage or "initializing"
            return PredictionsV1ResponseItem(
                sample_id=sample_id,
                predictions=[],
                error=f"model loading in progress; stage={stage}; retry later",
            )
        if _state.last_error:
            return PredictionsV1ResponseItem(
                sample_id=sample_id,
                predictions=[],
                error=f"model failed: {_state.last_error}",
            )
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error="model not ready",
        )

    if not isinstance(_state.model, dict):
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            error="internal error: model bundle missing",
        )
    return _predict_real(sample_id, item, _state.model)


app = FastAPI(title="BEANS-Next NatureLM-audio 1.1 launcher", version="0.1.0")


@app.on_event("startup")
def _on_startup() -> None:
    # Kick off model initialization early so `/health` stays responsive.
    _start_async_load()


@app.get("/health")
def health() -> dict[str, Any]:
    """Readiness probe.

    In normal mode, this endpoint fails fast with actionable messages for:
    missing ``NATURELM_GCS_CHECKPOINT_URI`` and weight download/load failures.

    Returns
    -------
    dict[str, Any]
        Health payload (HTTP 200) when ready.

    Raises
    ------
    fastapi.HTTPException
        When the model is still loading or failed to load.
    """
    if STUB_MODE:
        return {
            "status": "ok",
            "mode": "stub",
            "checkpoint_uri": GCS_CHECKPOINT_URI or "(not set)",
        }

    _start_async_load()
    with _lock:
        if _state.model is not None or _state.snapshot_path is not None:
            return {
                "status": "ok",
                "mode": "real" if _state.model is not None else "weights-only",
                "checkpoint_uri": GCS_CHECKPOINT_URI,
                "model_revision": _state.resolved_revision or "",
            }

        if _state.loading:
            started_s = (
                f"{time.time() - _state.loading_started_at:.1f}s"
                if _state.loading_started_at is not None
                else "unknown"
            )
            stage = _state.loading_stage or "initializing"
            stage_s = (
                f"{time.time() - _state.loading_stage_started_at:.1f}s"
                if _state.loading_stage_started_at is not None
                else "unknown"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "model loading in progress; "
                    f"checkpoint_uri={GCS_CHECKPOINT_URI!r}; stage={stage}; "
                    f"stage_elapsed={stage_s}; waited={started_s}"
                ),
            )

        if _state.last_error:
            raise HTTPException(
                status_code=503,
                detail=f"model failed: {_state.last_error}",
            )

    if not GCS_CHECKPOINT_URI:
        raise HTTPException(
            status_code=503,
            detail=(
                "NATURELM_GCS_CHECKPOINT_URI is not set. "
                "Export it to point at a GCS checkpoint directory, e.g. "
                "gs://foundation-models/naturelm-audio-1.1/base_model/1290000"
            ),
        )

    # Terminal "not ready" state with no explicit failure message.
    raise HTTPException(status_code=503, detail="model not ready")


@app.get("/info")
def info() -> InfoResponse:
    """Server capability discovery (DESIGN §4.3).

    Returns
    -------
    InfoResponse
        Capability document advertised by this launcher.
    """
    model_id = GCS_CHECKPOINT_URI if GCS_CHECKPOINT_URI else HF_REPO_ID
    revision = _state.resolved_revision or (
        _gcs_checkpoint_basename(GCS_CHECKPOINT_URI)
        if GCS_CHECKPOINT_URI
        else HF_REVISION
    )
    sample_rate = int(os.environ.get("NATURELM_SAMPLE_RATE", "16000"))
    load_status: str | None = None
    loading_stage: str | None = None
    loading_elapsed_sec: float | None = None
    last_error: str | None = None
    with _lock:
        if STUB_MODE:
            load_status = "stub"
        elif _state.model is not None:
            load_status = "ready"
        elif _state.loading:
            load_status = "loading"
            loading_stage = _state.loading_stage
            if _state.loading_stage_started_at is not None:
                loading_elapsed_sec = time.time() - _state.loading_stage_started_at
        elif _state.last_error is not None:
            load_status = "failed"
        last_error = _state.last_error
    return InfoResponse(
        name=LAUNCHER_NAME,
        model=model_id,
        model_revision=revision,
        audio_payload_types=list(AUDIO_PAYLOAD_TYPES),
        audio_sample_rate_hz=sample_rate,
        max_batch_size=MAX_BATCH_SIZE,
        supports_batching=SUPPORTS_BATCHING,
        schema_versions=list(SCHEMA_VERSIONS),
        load_status=load_status,
        loading_stage=loading_stage,
        loading_elapsed_sec=loading_elapsed_sec,
        last_error=last_error,
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

    # Trigger background load if needed, but never block this request.
    _start_async_load()

    started = time.perf_counter()
    responses: list[PredictionsV1ResponseItem] = []
    for raw in raw_requests:
        sample_id = raw["sample_id"]
        item_resp = _item_response_or_error(sample_id, raw)
        responses.append(item_resp)

    latency = time.perf_counter() - started
    for r in responses:
        if r.latency_sec is None:
            r.latency_sec = latency

    envelope = PredictionsV1Response(responses=responses)
    return Response(
        content=envelope.model_dump_json(),
        media_type="application/json",
        status_code=200,
    )


def main() -> None:
    """Run the launcher via Uvicorn."""
    import uvicorn

    host = os.environ.get("NATURELM_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
