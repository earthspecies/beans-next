"""BEANS-Next NatureLM-audio v1.0 HTTP launcher (``predictions_v1`` contract).

This launcher is part of the BEANS-Next serving kit. It intentionally keeps an
isolated environment and does not import ``beans_next``.

Modes
-----
Stub mode (``NATURELM_V1_0_STUB=1``):
    Returns deterministic hashed strings. No GPU, no weights. Used for contract
    conformance testing and CI.

Real mode (default when ``NATURELM_V1_0_STUB`` is unset or falsy):
    Loads EarthSpeciesProject/NatureLM-audio from HuggingFace and runs inference
    in an isolated launcher environment. This mode requires additional
    **launcher-local** dependencies (not installed by default on CPU-only
    validation hosts), including at minimum `torch` and the NatureLM-audio
    package from GitHub.

    If the GitHub-installed NatureLM-audio package is unavailable, the launcher
    will attempt a best-effort HuggingFace fallback using `transformers` with
    `trust_remote_code=True`. This only works if the model repo exposes a
    compatible Transformers interface for audio-conditioned text generation.

    Notes
    -----
    - Real mode can be configured to run on CPU (it will warn when CUDA is not
      available), but it is expected to be extremely slow and may be impractical
      for anything beyond a minimal smoke check.
    - See `README.md` for the recommended install commands for real mode.

    Key environment variables::

        NATURELM_V1_0_STUB=1          enable stub mode (contract-only)
        NATURELM_V1_0_MODEL=...  HF model id; default EarthSpeciesProject/NatureLM-audio
        NATURELM_V1_0_DEVICE=cuda     torch device (default: cuda if available else cpu)
        NATURELM_V1_0_MAX_BATCH_SIZE=4  batch size hint for /info
        HF_TOKEN=hf_...               needed only for gated repos
        HF_HOME=/path/to/cache        HuggingFace cache directory
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
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, ValidationError

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"

_AUDIO_TAG_PATTERN = re.compile(r"<Audio><AudioHere></Audio>")
_NATURELM_SYSTEM_HEADER_PATTERN = re.compile(
    r"<\|start_header_id\|>system<\|end_header_id\|>\n\n"
    r"Cutting Knowledge Date: [^\n]+\n"
    r"Today Date: [^\n]+\n\n"
    r"<\|eot_id\|>"
)


def _normalize_query_for_v1_0(query: str) -> str:
    """Normalize user queries for NatureLM v1.0 inference.

    NatureLM v1.0 is sensitive to prompt formatting. In particular, some BEANS-Zero
    ESC-50 rows delivered via certain backends include the entire label set in the
    instruction ("one of the following categories: ..."), which effectively turns
    an "official instruction" run into closed-set classification. For closed-set
    classification tasks, we **do want** the candidate labels present.

    This normalization is intentionally **server-local** (model-specific) rather
    than a dataset-loader policy.

    Returns
    -------
    str
        Normalized query text to pass to the NatureLM v1.0 prompt builder.
    """
    if not _get_bool_env("NATURELM_V1_0_NORMALIZE_PROMPTS", False):
        return query

    # Formatting-only normalization: keep the instruction content intact (including
    # candidate labels). Only ensure the audio placeholder exists, since NatureLM's
    # internal prompt builder expects it.
    if "<Audio><AudioHere></Audio>" not in query:
        return f"<Audio><AudioHere></Audio> {query.lstrip()}".strip()
    return query


def _count_audio_placeholders(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(_AUDIO_TAG_PATTERN.findall(content))
    return total


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


def _expected_sample_rate_hz() -> int:
    # NatureLM-audio v1.0 is typically served at 16 kHz.
    return _get_int_env("NATURELM_V1_0_EXPECTED_SAMPLE_RATE_HZ", 16000)


def _prepare_waveform_like_gradio_app(
    audio_arr: "object",
    *,
    input_sample_rate: int,
    device_is_cuda: bool,
    max_length_seconds: int,
) -> tuple["object", int]:
    """Match NatureLM-audio webapp preprocessing if available.

    The NatureLM Gradio app uses `NatureLM.utils.prepare_sample_waveforms`, which
    handles resampling and device placement. We reuse it when installed; otherwise
    fall back to passing the raw waveform through.

    Returns
    -------
    tuple[object, int]
        The (possibly transformed) waveform and the sample rate that should be
        treated as its sampling rate for downstream processing.
    """
    expected_sr = _expected_sample_rate_hz()
    try:
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from NatureLM.utils import (  # type: ignore[import-not-found]  # noqa: PLC0415
            prepare_sample_waveforms,
        )
    except Exception:
        return audio_arr, int(input_sample_rate)

    # The upstream webapp passes file paths. To reuse the exact preprocessing
    # (resample + pad/truncate), materialize a temporary WAV.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="beans-next-naturelm-", suffix=".wav", delete=False
        ) as f:
            tmp_path = f.name

        arr = np.asarray(audio_arr, dtype=np.float32)
        sf.write(tmp_path, arr, int(input_sample_rate), format="WAV")

        samples = prepare_sample_waveforms(
            [tmp_path],
            cuda_enabled=device_is_cuda,
            sr=int(expected_sr),
            max_length_seconds=int(max_length_seconds),
        )
        raw_wav = samples.get("raw_wav")
        if not isinstance(raw_wav, torch.Tensor):
            return audio_arr, int(input_sample_rate)
        wav = raw_wav.detach().to("cpu").float().numpy()
        if wav.ndim == 2:
            wav = wav[0]
        return wav.astype(np.float32), int(expected_sr)
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _deterministic_stub_prediction(
    sample_id: str,
    item: PredictionsV1RequestItem,
) -> str:
    messages_payload = [
        {"role": m.role, "content": m.content} for m in item.messages
    ]
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
    return f"naturelm_v1_0_stub:{digest}"


@dataclass
class _ModelState:
    """In-memory launcher state for readiness and identity reporting."""

    stub: bool
    model_revision: str = "unknown"
    pipeline: Any = field(default=None, repr=False)


def _resolve_hf_revision(model_name: str) -> str:
    """Return the HEAD commit SHA for a HuggingFace repo, or ``'unknown'``.

    Returns
    -------
    str
        Commit SHA string, or ``'unknown'`` if the hub is unreachable.
    """
    try:
        from huggingface_hub import model_info as _hf_model_info  # noqa: PLC0415

        info = _hf_model_info(model_name)
        return str(info.sha) if info.sha else "unknown"
    except Exception:
        return "unknown"


def _load_real_pipeline(model_name: str, device: str) -> object:  # noqa: ANN401
    """Load the NatureLM-audio inference pipeline.

    Parameters
    ----------
    model_name:
        HuggingFace repo id, e.g. ``"EarthSpeciesProject/NatureLM-audio"``.
    device:
        PyTorch device string, e.g. ``"cuda"`` or ``"cuda:0"``.

    Returns
    -------
    object
        A loaded pipeline object with an ``infer(messages, audio, sample_rate)``
        method (or equivalent — adapt to the actual NatureLM API below).

    Notes
    -----
    The NatureLM-audio package is installed from::

        pip install git+https://github.com/earthspeciesproject/NatureLM-audio.git

    The launcher prefers the in-repo `NatureLM` Python package API first
    (``NatureLM.infer.Pipeline``). If that import fails, it falls back to a
    best-effort Transformers load using ``trust_remote_code=True``.

    Raises
    ------
    RuntimeError
        If the NatureLM package load is requested but unavailable, or if a
        required environment variable (e.g. `NATURELM_CFG_PATH`) is missing.
    """
    load_mode = os.environ.get("NATURELM_V1_0_LOAD_MODE", "pipeline").strip().lower()

    def _load_via_naturelm_package() -> object:  # noqa: ANN401
        from NatureLM.infer import Pipeline, load_model_and_config  # noqa: PLC0415

        print(
            f"[naturelm-v1.0] Loading NatureLM Pipeline from {model_name!r} "
            f"on {device!r} …"
        )
        t0 = time.time()
        # NatureLM-audio's `Pipeline` does not expose `from_pretrained`; it expects
        # either:
        # - Pipeline() (downloads model, picks device internally), or
        # - Pipeline(model=...) with a pre-loaded model already moved to the desired
        #   device.
        cfg_path = os.environ.get("NATURELM_CFG_PATH")
        if not cfg_path:
            raise RuntimeError(
                "NATURELM_CFG_PATH must be set to a readable inference.yml path"
            )
        model, _cfg = load_model_and_config(cfg_path=cfg_path, device=device)
        loaded = Pipeline(model=model, cfg_path=cfg_path)
        elapsed = time.time() - t0
        print(f"[naturelm-v1.0] NatureLM Pipeline loaded in {elapsed:.1f}s")
        return loaded

    try:
        return _load_via_naturelm_package()
    except ImportError as exc:
        print(
            "[naturelm-v1.0] NatureLM package import failed; attempting Transformers "
            f"fallback (ImportError: {exc})",
            flush=True,
        )

    if load_mode in {"pipeline", "naturelm", "naturelm_package"}:
        raise RuntimeError(
            "NatureLM package load failed and transformers fallback is disabled "
            f"(NATURELM_V1_0_LOAD_MODE={load_mode!r})."
        )

    # Fallback: HuggingFace Transformers with trust_remote_code=True (best-effort).
    return _load_real_pipeline_via_transformers(model_name=model_name, device=device)


class _HFTransformersPipeline:
    """Best-effort wrapper exposing `infer(messages, audio, sample_rate)`.

    This is used only when `NatureLM.infer.Pipeline` is unavailable (e.g. GitHub
    install path is blocked on this host).
    """

    def __init__(self, model: object, processor: object, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def infer(
        self, messages: list[dict[str, Any]], audio: object, sample_rate: int
    ) -> str:
        import torch  # noqa: PLC0415

        prompt = _messages_to_text_prompt(messages)

        # Many audio-capable processors accept one of:
        # - text=..., audios=..., sampling_rate=...
        # - text=..., audio=..., sampling_rate=...
        # Keep this defensive and fail-fast if the repo's remote code doesn't match.
        processor = self._processor
        tried: list[str] = []
        inputs = None
        for kwargs, label in (
            (
                {
                    "text": prompt,
                    "audios": audio,
                    "sampling_rate": sample_rate,
                    "return_tensors": "pt",
                },
                "text+audios",
            ),
            (
                {
                    "text": prompt,
                    "audio": audio,
                    "sampling_rate": sample_rate,
                    "return_tensors": "pt",
                },
                "text+audio",
            ),
            ({"text": prompt, "return_tensors": "pt"}, "text-only"),
        ):
            tried.append(label)
            try:
                inputs = processor(**kwargs)
                break
            except TypeError:
                continue

        if inputs is None:
            raise RuntimeError(
                "Transformers fallback could not build model inputs. "
                f"Tried processor call patterns: {', '.join(tried)}. "
                "This likely means the model repo's `trust_remote_code` interface "
                "is not compatible with this launcher. Install the official "
                "NatureLM-audio package (GitHub) instead."
            )

        if isinstance(inputs, dict):
            inputs = {
                k: v.to(self._device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        gen_kwargs = _generation_kwargs_from_messages(messages=messages)
        try:
            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **gen_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers fallback model.generate(...) failed. "
                "This likely indicates an API mismatch for this model repo when "
                "loaded via `transformers` with `trust_remote_code=True`. "
                "Install the official NatureLM-audio package (GitHub) instead."
            ) from exc

        return _decode_transformers_output(processor=processor, output_ids=output_ids)


def _messages_to_text_prompt(messages: list[dict[str, Any]]) -> str:
    """Lossy conversion from chat messages to a single text prompt.

    Parameters
    ----------
    messages:
        List of dicts with `role` and `content`.

    Returns
    -------
    str
        Joined text prompt suitable for best-effort `transformers` generation.
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue
        if role:
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return "\n".join(parts)


def _messages_to_naturelm_query(messages: list[dict[str, Any]]) -> str:
    """Extract the raw user prompt content for NatureLM Pipeline.

    NatureLM-audio's `Pipeline` applies its own prompt template via
    `NatureLMAudioProcessor.prepare_instruction`. To match the official BEANS-Zero
    evaluation, we should pass the raw dataset-provided instruction (which already
    includes `<Audio><AudioHere></Audio>` when applicable) without adding role
    prefixes or escaping newlines.

    Returns
    -------
    str
        The raw user prompt content to pass to the NatureLM pipeline.
    """
    # Prefer the most recent user message content.
    for msg in reversed(messages):
        if str(msg.get("role", "")).strip().lower() == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)

    # Fallback: join message contents losslessly.
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if content.strip():
            parts.append(content)
    return "\n\n".join(parts)


def _generation_kwargs_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Best-effort generation kwargs; ignores unknown config.

    Parameters
    ----------
    messages:
        Chat messages. Currently unused beyond placeholder for future extraction of
        generation config.

    Returns
    -------
    dict[str, Any]
        Keyword arguments passed to `model.generate`.
    """
    # This launcher only needs minimal compatibility for a bring-up check. If the
    # model requires a richer API (chat templates, etc.), we fail fast elsewhere.
    return {"max_new_tokens": 256}


def _decode_transformers_output(processor: object, output_ids: object) -> str:
    # Common patterns:
    # - processor.decode(ids[0], skip_special_tokens=True)
    # - processor.tokenizer.decode(...)
    # - processor.batch_decode(...)
    try:
        if hasattr(processor, "decode"):
            return str(processor.decode(output_ids[0], skip_special_tokens=True))
    except Exception:
        pass

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        try:
            if hasattr(tok, "decode"):
                return str(tok.decode(output_ids[0], skip_special_tokens=True))
        except Exception:
            pass

    try:
        if hasattr(processor, "batch_decode"):
            texts = processor.batch_decode(output_ids, skip_special_tokens=True)
            if isinstance(texts, list) and texts:
                return str(texts[0])
    except Exception:
        pass

    return str(output_ids)


def _load_real_pipeline_via_transformers(model_name: str, device: str) -> object:  # noqa: ANN401
    """Load model via `transformers` with `trust_remote_code=True` (best-effort).

    Returns
    -------
    object
        A wrapper exposing an `infer(messages, audio, sample_rate) -> str` method.

    Raises
    ------
    RuntimeError
        If `transformers` is not installed or the model cannot be loaded into a
        minimal `generate`-based inference wrapper.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "NatureLM Transformers fallback requested but `transformers` is not "
            "installed. Install (launcher-local): uv pip install transformers"
        ) from exc

    print(
        f"[naturelm-v1.0] Loading via transformers from {model_name!r} on {device!r} "
        "(trust_remote_code=True) …",
        flush=True,
    )
    t0 = time.time()

    # Processor first (some repos rely on remote-code processor classes).
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Use AutoModelForCausalLM as the least-bad default for audio-conditioned chat
    # models with a text generation head.
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    if device != "cpu":
        model.to(device)

    elapsed = time.time() - t0
    print(f"[naturelm-v1.0] Transformers model loaded in {elapsed:.1f}s", flush=True)
    return _HFTransformersPipeline(model=model, processor=processor, device=device)


def _run_real_inference(pipeline: object, item: "PredictionsV1RequestItem") -> str:  # noqa: ANN401
    """Run one sample through the NatureLM pipeline.

    Parameters
    ----------
    pipeline:
        Loaded pipeline from :func:`_load_real_pipeline`.
    item:
        Validated request item (audio + messages).

    Returns
    -------
    str
        Model text prediction.

    Notes
    -----
    Audio decode: ``base64_wav`` → ``io.BytesIO`` → ``soundfile.read`` → numpy
    float32 mono array at whatever sample rate is in the WAV header (the runner
    always sends 16 kHz from HF BEANS-Zero, but the pipeline should handle
    resampling internally if needed).

    NatureLM-audio's `Pipeline` is callable (see `NatureLM/infer.py` in the
    upstream repo). It does not implement `infer(...)`; instead we call it with
    `audios=[...]` and `queries=[...]`.

    Raises
    ------
    ValueError
        If ``item.audio_inputs`` is empty or the ``payload_type`` is not
        ``base64_wav``, ``file_path``, or ``file_url``.
    RuntimeError
        If the pipeline cannot be loaded or returns an invalid response.
    """
    import urllib.request  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    # --- Decode audio --------------------------------------------------------
    if not item.audio_inputs:
        raise ValueError("Request has no audio_inputs")
    audio_slot = item.audio_inputs[0]  # NatureLM-audio: one audio per sample

    if audio_slot.payload_type == "base64_wav":
        raw_bytes = base64.b64decode(audio_slot.data)
        audio_arr, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    elif audio_slot.payload_type == "file_path":
        audio_arr, sample_rate = sf.read(audio_slot.data, dtype="float32")
    elif audio_slot.payload_type == "file_url":
        with urllib.request.urlopen(audio_slot.data, timeout=30) as resp:  # noqa: S310
            raw_bytes = resp.read()
        audio_arr, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    else:
        raise ValueError(
            f"Unsupported payload_type {audio_slot.payload_type!r} for real inference. "
            "Use base64_wav, file_path, or file_url."
        )

    # Stereo → mono
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)
    audio_arr = audio_arr.astype(np.float32)

    # --- Preprocess (resample/pad/etc) ---------------------------------------
    # Mirror NatureLM-audio's own webapp preprocessing when available.
    try:
        import torch  # noqa: PLC0415

        device_is_cuda = torch.cuda.is_available()
    except Exception:
        device_is_cuda = False

    prepared, expected_sr = _prepare_waveform_like_gradio_app(
        audio_arr,
        input_sample_rate=int(sample_rate),
        device_is_cuda=device_is_cuda,
        max_length_seconds=(
            int(item.generation_config.max_length_seconds)
            if getattr(item.generation_config, "max_length_seconds", None)
            not in (None, 0)
            else 10
        ),
    )

    # --- Build prompt ---------------------------------------------------------
    messages = [{"role": m.role, "content": m.content} for m in item.messages]
    query = _normalize_query_for_v1_0(_messages_to_naturelm_query(messages).strip())

    # --- Inference -----------------------------------------------------------
    # NatureLM Pipeline returns a list[str], one entry per audio.
    results = pipeline(
        audios=[prepared],
        queries=[query],
        input_sample_rate=int(expected_sr),
    )
    if not results:
        raise RuntimeError("NatureLM pipeline returned no results")
    return str(results[0])


def _init_state() -> _ModelState:
    """Initialise launcher state: stub mode or real GPU inference.

    Returns
    -------
    _ModelState
        Populated state; ``pipeline`` is ``None`` in stub mode.

    Raises
    ------
    RuntimeError
        If real mode is requested (``NATURELM_V1_0_STUB=0``) but required
        dependencies are missing or model initialization fails.
    """
    # IMPORTANT: For Increment 7 (I7-A) "real-mode feasibility" work, the launcher
    # treats `NATURELM_V1_0_STUB` as an *opt-in* flag:
    # - unset / falsy => real mode
    # - truthy => stub mode
    stub = _get_bool_env("NATURELM_V1_0_STUB", False)

    if stub:
        revision = os.environ.get("NATURELM_V1_0_MODEL_REVISION", "stub")
        return _ModelState(stub=True, model_revision=revision)

    # --- Real inference mode -------------------------------------------------
    try:
        import torch  # noqa: PLC0415 — only imported in real mode
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Real mode requested (NATURELM_V1_0_STUB=0) but required dependency "
            f"'torch' could not be imported: {exc}"
        ) from exc

    device_env = os.environ.get("NATURELM_V1_0_DEVICE", "")
    if device_env:
        device = device_env
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print(
            "[naturelm-v1.0] WARNING: CUDA not available; running inference on CPU. "
            "This is very slow. Set NATURELM_V1_0_STUB=1 for conformance-only mode.",
            flush=True,
        )

    print(f"[naturelm-v1.0] Real inference mode — device={device!r}", flush=True)

    pipeline = _load_real_pipeline(MODEL_NAME, device)
    revision = os.environ.get(
        "NATURELM_V1_0_MODEL_REVISION",
        _resolve_hf_revision(MODEL_NAME),
    )
    return _ModelState(
        stub=False,
        model_revision=revision,
        pipeline=pipeline,
    )


MAX_BATCH_SIZE: int = _get_int_env("NATURELM_V1_0_MAX_BATCH_SIZE", 8)

LAUNCHER_NAME: str = "beans-next-naturelm-v1.0"
MODEL_NAME: str = os.environ.get(
    "NATURELM_V1_0_MODEL",
    "EarthSpeciesProject/NatureLM-audio",
)

AUDIO_PAYLOAD_TYPES: list[str] = ["base64_wav", "file_path", "file_url"]
SUPPORTS_BATCHING: bool = True
SCHEMA_VERSIONS: list[str] = [PREDICTIONS_V1]

STATE = _init_state()

app = FastAPI(title="BEANS-Next NatureLM-audio v1.0 launcher", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    mode = "stub" if STATE.stub else "real"
    load_status = "stub" if STATE.stub else "ready"
    return {"status": "ok", "mode": mode, "load_status": load_status}


@app.get("/info")
def info() -> InfoResponse:
    return InfoResponse(
        name=LAUNCHER_NAME,
        model=MODEL_NAME,
        model_revision=STATE.model_revision,
        audio_payload_types=list(AUDIO_PAYLOAD_TYPES),
        audio_sample_rate_hz=_expected_sample_rate_hz(),
        max_batch_size=MAX_BATCH_SIZE,
        supports_batching=SUPPORTS_BATCHING,
        schema_versions=list(SCHEMA_VERSIONS),
        load_status="stub" if STATE.stub else "ready",
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

    if STATE.stub:
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

    # Real inference (NATURELM_V1_0_STUB=0)
    if STATE.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="pipeline not initialised",
        )

    try:
        pred_text = _run_real_inference(STATE.pipeline, item)
    except Exception as exc:  # noqa: BLE001
        latency = time.time() - t0
        return PredictionsV1ResponseItem(
            sample_id=sample_id,
            predictions=[],
            finish_reason=None,
            latency_sec=latency,
            error=f"inference error: {exc}",
        )

    latency = time.time() - t0
    return PredictionsV1ResponseItem(
        sample_id=sample_id,
        predictions=[pred_text],
        finish_reason="stop",
        usage=None,
        latency_sec=latency,
        error=None,
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
        sample_id = raw["sample_id"]
        responses.append(_item_response_or_error(sample_id, raw))

    envelope = PredictionsV1Response(responses=responses)
    return Response(
        content=envelope.model_dump_json(),
        media_type="application/json",
        status_code=200,
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("NATURELM_V1_0_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "19081"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
