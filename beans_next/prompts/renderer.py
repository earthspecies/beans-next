"""Jinja2-based rendering from ``DatasetExample`` rows to ``ModelRequest``.

Templates are loaded from YAML under ``beans_next/registry/prompt/``. Rendered
chat text must contain one ``<Audio><AudioHere></Audio>`` span per audio slot
listed in the prompt spec; mismatch raises :exc:`AudioPlaceholderAlignmentError`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import yaml

from beans_next.api.types import (
    PREDICTIONS_V1,
    AudioInput,
    ChatMessage,
    DatasetExample,
    GenerationConfig,
    ModelRequest,
)
from beans_next.audio.gaussian_noise import GaussianNoiseConfig, GaussianNoiseRecord
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER

_AUDIO_PLACEHOLDER_RE = re.compile(re.escape(AUDIO_PLACEHOLDER))
_TEXT_ONLY_AUDIO_TOKEN_RE = re.compile(
    rf"{re.escape(AUDIO_PLACEHOLDER)}|\[audio\]",
    flags=re.IGNORECASE,
)

ModalityMode = Literal["audio", "gaussian-noise", "text-only", "text-only-informed"]

TEXT_ONLY_INSTRUCTION = (
    "Audio is intentionally unavailable. Answer using only the text and answer "
    "choices. Make your best guess and follow the requested answer format. Do not "
    "refuse or explain."
)


class AudioPlaceholderAlignmentError(ValueError):
    """Raised when rendered audio placeholders do not match ``audio_inputs``.

    Attributes
    ----------
    sample_id
        Dataset sample id included for debugging.
    n_placeholders
        Count of ``AUDIO_PLACEHOLDER`` spans across rendered messages.
    n_audio_inputs
        Number of ``AudioInput`` rows produced for the example.
    """

    def __init__(
        self,
        message: str,
        *,
        sample_id: str,
        n_placeholders: int,
        n_audio_inputs: int,
    ) -> None:
        super().__init__(message)
        self.sample_id = sample_id
        self.n_placeholders = n_placeholders
        self.n_audio_inputs = n_audio_inputs


@dataclass(frozen=True)
class AudioSlotSpec:
    """Declarative mapping from ``DatasetExample.metadata`` to ``AudioInput`` objects.

    Attributes
    ----------
    metadata_key
        Dotted lookup path under ``example.metadata`` for the wire ``data``
        field (path, URL, or base64 payload depending on ``payload_type``).
        Must be non-empty unless ``list_metadata_key`` is set.
    payload_type
        Transport encoding for this slot.
    sample_rate
        Fixed sample rate in Hz when applicable, used when no metadata override
        is present.
    sample_rate_metadata_key
        Optional dotted path under ``example.metadata`` for sample rate; when
        present and resolves to an ``int``, it overrides ``sample_rate``.
    list_metadata_key
        Optional dotted lookup path under ``example.metadata`` for a list of
        data values.  When set, this single slot expands to one ``AudioInput``
        per list element (multi-audio).  ``metadata_key`` is ignored when this
        field is non-empty.
    """

    metadata_key: str = ""
    payload_type: Literal["base64_wav", "file_path", "file_url"] = "file_path"
    sample_rate: int | None = None
    sample_rate_metadata_key: str | None = None
    list_metadata_key: str | None = None


@dataclass(frozen=True)
class PromptSpec:
    """Parsed prompt configuration (messages, audio slots, generation defaults).

    Attributes
    ----------
    prompt_id
        Stable identifier for logging and reproducibility.
    message_templates
        Ordered ``(role, jinja_template)`` pairs rendered with a per-example
        context (see :meth:`PromptRenderer.render`).
    audio_slots
        Slot definitions whose length must match the number of
        ``AUDIO_PLACEHOLDER`` occurrences after rendering.
    generation_config
        Optional decoding parameters forwarded to ``ModelRequest``.
    """

    prompt_id: str
    message_templates: tuple[tuple[str, str], ...]
    audio_slots: tuple[AudioSlotSpec, ...]
    generation_config: GenerationConfig | None = None


def builtin_prompt_registry_path() -> Path:
    """Return the directory containing bundled ``*.yaml`` prompt specs.

    Returns
    -------
    pathlib.Path
        Absolute path to ``beans_next/registry/prompt``.
    """
    return Path(__file__).resolve().parent.parent / "registry" / "prompt"


def load_prompt_spec_from_path(path: Path | str) -> PromptSpec:
    """Load a :class:`PromptSpec` from a YAML file on disk.

    Parameters
    ----------
    path
        Path to the prompt YAML document.

    Returns
    -------
    PromptSpec
        Parsed, validated prompt configuration.

    Raises
    ------
    ValueError
        If the document is missing required keys or has invalid structure.

    Notes
    -----
    ``FileNotFoundError`` may propagate from :meth:`pathlib.Path.read_text`.
    ``yaml.YAMLError`` may propagate from :func:`yaml.safe_load`.
    """
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        msg = f"Prompt YAML root must be a mapping, got {type(raw).__name__}"
        raise ValueError(msg)
    data = cast(Mapping[str, Any], raw)
    return _parse_prompt_spec(data, source=str(p))


def load_builtin_prompt_yaml(filename: str) -> PromptSpec:
    """Load a bundled prompt spec by file name from the registry directory.

    Parameters
    ----------
    filename
        File name such as ``"classification_bioacoustic_v1.yaml"``.

    Returns
    -------
    PromptSpec
        Parsed prompt configuration.

    Notes
    -----
    Errors from :func:`load_prompt_spec_from_path` apply, including
    ``FileNotFoundError`` when the file is missing.
    """
    path = builtin_prompt_registry_path() / filename
    return load_prompt_spec_from_path(path)


class PromptRenderer:
    """Render ``DatasetExample`` rows into ``ModelRequest`` using Jinja2.

    Raises
    ------
    ImportError
        If ``jinja2`` is not installed (raised from the constructor).
    """

    def __init__(
        self,
        spec: PromptSpec,
        *,
        modality_mode: ModalityMode = "audio",
        gaussian_noise_config: GaussianNoiseConfig | None = None,
    ) -> None:
        if modality_mode not in {
            "audio",
            "gaussian-noise",
            "text-only",
            "text-only-informed",
        }:
            raise ValueError(f"Unsupported modality mode: {modality_mode!r}")
        if modality_mode == "gaussian-noise" and gaussian_noise_config is None:
            raise ValueError(
                "gaussian_noise_config is required for gaussian-noise mode"
            )
        self._spec = spec
        self._modality_mode = modality_mode
        self._gaussian_noise_config = gaussian_noise_config
        self._gaussian_noise_records: dict[tuple[str, int], dict[str, Any]] = {}
        self._gaussian_noise_lock = threading.Lock()
        env_cls = _load_sandboxed_environment_class()
        self._env = env_cls(autoescape=False, enable_async=False)
        self._compiled = [
            (role, self._env.from_string(template))
            for role, template in spec.message_templates
        ]

    @property
    def prompt_id(self) -> str:
        """Stable prompt identifier from the bound :class:`PromptSpec`."""
        return self._spec.prompt_id

    def render(self, example: DatasetExample) -> ModelRequest:
        """Render ``example`` to a ``ModelRequest`` for ``predictions_v1``.

        Parameters
        ----------
        example
            Normalized dataset row including ``metadata`` used by templates and
            audio slot bindings.

        Returns
        -------
        ModelRequest
            Request item compatible with :class:`beans_next.api.types.ModelRequest`.

        Notes
        -----
        Raises :exc:`AudioPlaceholderAlignmentError` when placeholder counts do not
        match built ``audio_inputs``. Template rendering may raise
        ``jinja2.TemplateError``; metadata resolution may raise ``ValueError``.
        """
        ctx = _build_template_context(example)
        messages: list[ChatMessage] = []
        for role, tmpl in self._compiled:
            content = tmpl.render(**ctx)
            messages.append(ChatMessage(role=role, content=content))
        if self._modality_mode in {"audio", "gaussian-noise"}:
            audio_inputs = _build_audio_inputs(example, self._spec.audio_slots)
            if self._modality_mode == "gaussian-noise":
                audio_inputs = self._replace_with_gaussian_noise(
                    example.sample_id, audio_inputs
                )
            _assert_audio_alignment(
                messages,
                audio_inputs,
                sample_id=example.sample_id,
            )
        else:
            messages = _render_text_only_messages(
                messages,
                informed=self._modality_mode == "text-only-informed",
            )
            audio_inputs = []
        return ModelRequest(
            schema_version=PREDICTIONS_V1,
            sample_id=example.sample_id,
            messages=messages,
            audio_inputs=audio_inputs,
            generation_config=self._spec.generation_config,
        )

    def _replace_with_gaussian_noise(
        self,
        sample_id: str,
        audio_inputs: list[AudioInput],
    ) -> list[AudioInput]:
        """Replace every resolved source slot while preserving its wire order.

        Returns
        -------
        list[AudioInput]
            File-path inputs pointing to deterministic cached noise.

        Raises
        ------
        ValueError
            If a source slot is not a locally resolved file path.
        RuntimeError
            If the renderer has no noise configuration.
        """
        config = self._gaussian_noise_config
        if config is None:  # pragma: no cover - constructor enforces this invariant
            raise RuntimeError("Gaussian-noise configuration is unavailable")
        replaced: list[AudioInput] = []
        for slot_index, source in enumerate(audio_inputs):
            if source.payload_type != "file_path":
                raise ValueError(
                    "gaussian-noise mode requires locally resolved file_path audio; "
                    f"sample_id={sample_id!r} slot_index={slot_index} "
                    f"payload_type={source.payload_type!r}"
                )
            record = config.materialize(
                source.data,
                source_identity=source.data,
                slot_index=slot_index,
            )
            replaced.append(
                AudioInput(
                    payload_type="file_path",
                    data=str(record.path),
                    sample_rate=record.sample_rate,
                )
            )
            manifest_record = _gaussian_noise_manifest_record(
                sample_id=sample_id,
                source_path=source.data,
                record=record,
            )
            with self._gaussian_noise_lock:
                self._gaussian_noise_records[(sample_id, slot_index)] = manifest_record
        return replaced

    def write_gaussian_noise_manifest(
        self,
        path: Path | str,
        *,
        code_commit: str | None,
    ) -> None:
        """Atomically write a deterministic run manifest, merging resume records.

        Raises
        ------
        ValueError
            If an existing manifest was created with a different protocol.
        """
        if self._modality_mode != "gaussian-noise":
            return
        config = self._gaussian_noise_config
        if config is None:  # pragma: no cover - constructor invariant
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            expected = {
                "schema_version": "beans_next.gaussian_noise_manifest.v1",
                "protocol_version": config.protocol_version,
                "dataset_revision": config.dataset_revision,
                "global_seed": config.global_seed,
                "rms_dbfs": config.rms_dbfs,
            }
            mismatched = [
                key for key, value in expected.items() if existing.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    "Cannot merge Gaussian-noise manifest with different protocol "
                    f"fields: {', '.join(mismatched)}"
                )
            for item in existing.get("records", []):
                records_by_key[(str(item["sample_id"]), int(item["slot_index"]))] = item
        with self._gaussian_noise_lock:
            records_by_key.update(self._gaussian_noise_records)
        payload = {
            "schema_version": "beans_next.gaussian_noise_manifest.v1",
            "protocol": "zero-mean-white-Gaussian-noise",
            "protocol_version": config.protocol_version,
            "dataset_revision": config.dataset_revision,
            "global_seed": config.global_seed,
            "rms_dbfs": config.rms_dbfs,
            "cache_dir": str(config.cache_dir),
            "code_commit": code_commit
            or os.environ.get("BEANS_NEXT_CODE_COMMIT")
            or "unknown",
            "records": [records_by_key[key] for key in sorted(records_by_key)],
        }
        encoded = (
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )
        fd, raw_temp = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temp_path = Path(raw_temp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)


def _gaussian_noise_manifest_record(
    *, sample_id: str, source_path: str, record: GaussianNoiseRecord
) -> dict[str, Any]:
    """Return the run-level provenance row for one sample/audio slot.

    Returns
    -------
    dict[str, Any]
        JSON-serializable manifest record.
    """
    return {
        "sample_id": sample_id,
        "slot_index": record.slot_index,
        "source_identity": record.source_identity,
        "source_path": source_path,
        "noise_path": str(record.path),
        "metadata_path": str(record.metadata_path),
        "seed": record.seed,
        "seed_sha256": record.seed_sha256,
        "audio_sha256": record.sha256,
        "source_sha256": record.source_sha256,
        "frames": record.frames,
        "sample_rate": record.sample_rate,
        "channels": record.channels,
        "duration_seconds": record.frames / record.sample_rate,
        "mean": record.mean,
        "rms": record.rms,
        "peak": record.peak,
        "rms_dbfs": record.rms_dbfs,
    }


def _render_text_only_messages(
    messages: list[ChatMessage],
    *,
    informed: bool,
) -> list[ChatMessage]:
    """Remove audio placeholders and optionally add the no-audio instruction.

    Returns
    -------
    list[ChatMessage]
        New messages that contain no audio placeholders.
    """
    rendered = [
        ChatMessage(
            role=message.role,
            content=_TEXT_ONLY_AUDIO_TOKEN_RE.sub("", message.content),
        )
        for message in messages
    ]
    if not informed:
        return rendered

    for index, message in enumerate(rendered):
        if message.role != "user":
            continue
        content = message.content.lstrip()
        rendered[index] = ChatMessage(
            role="user",
            content=(
                f"{TEXT_ONLY_INSTRUCTION}\n\n{content}"
                if content
                else TEXT_ONLY_INSTRUCTION
            ),
        )
        return rendered

    return [ChatMessage(role="user", content=TEXT_ONLY_INSTRUCTION), *rendered]


def _load_sandboxed_environment_class() -> type:
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError as exc:
        msg = (
            "Prompt rendering requires the optional `jinja2` dependency. "
            "Add `jinja2` to the project environment (for example in "
            "`pyproject.toml`) and run `uv sync`."
        )
        raise ImportError(msg) from exc
    return SandboxedEnvironment


def _parse_prompt_spec(data: Mapping[str, Any], *, source: str) -> PromptSpec:
    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        msg = f"`prompt_id` must be a non-empty string ({source})"
        raise ValueError(msg)
    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        msg = f"`messages` must be a non-empty list ({source})"
        raise ValueError(msg)
    pairs: list[tuple[str, str]] = []
    for i, item in enumerate(raw_messages):
        if not isinstance(item, Mapping):
            msg = f"`messages[{i}]` must be a mapping ({source})"
            raise ValueError(msg)
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role.strip():
            msg = f"`messages[{i}].role` must be a non-empty string ({source})"
            raise ValueError(msg)
        if not isinstance(content, str):
            msg = f"`messages[{i}].content` must be a string template ({source})"
            raise ValueError(msg)
        pairs.append((role, content))
    raw_slots = data.get("audio_slots", [])
    if raw_slots is None:
        raw_slots = []
    if not isinstance(raw_slots, list):
        msg = f"`audio_slots` must be a list when provided ({source})"
        raise ValueError(msg)
    slots: list[AudioSlotSpec] = []
    for i, slot in enumerate(raw_slots):
        if not isinstance(slot, Mapping):
            msg = f"`audio_slots[{i}]` must be a mapping ({source})"
            raise ValueError(msg)
        lmk = slot.get("list_metadata_key")
        if lmk is not None and (not isinstance(lmk, str) or not lmk.strip()):
            msg = (
                f"`audio_slots[{i}].list_metadata_key` must be a non-empty string "
                f"or null ({source})"
            )
            raise ValueError(msg)
        mk = slot.get("metadata_key", "")
        if not isinstance(mk, str):
            mk = ""
        if not lmk and not mk.strip():
            msg = (
                f"`audio_slots[{i}].metadata_key` must be a non-empty string "
                f"unless `list_metadata_key` is set ({source})"
            )
            raise ValueError(msg)
        pt = slot.get("payload_type")
        if pt not in ("base64_wav", "file_path", "file_url"):
            msg = (
                f"`audio_slots[{i}].payload_type` must be one of base64_wav, "
                f"file_path, file_url ({source})"
            )
            raise ValueError(msg)
        pt_typed = cast(Literal["base64_wav", "file_path", "file_url"], pt)
        sr = slot.get("sample_rate")
        if sr is not None and not isinstance(sr, int):
            msg = f"`audio_slots[{i}].sample_rate` must be an int or null ({source})"
            raise ValueError(msg)
        sr_mk = slot.get("sample_rate_metadata_key")
        if sr_mk is not None and (not isinstance(sr_mk, str) or not sr_mk.strip()):
            msg = (
                f"`audio_slots[{i}].sample_rate_metadata_key` must be a non-empty "
                f"string or null ({source})"
            )
            raise ValueError(msg)
        slots.append(
            AudioSlotSpec(
                metadata_key=mk,
                payload_type=pt_typed,
                sample_rate=cast(int | None, sr),
                sample_rate_metadata_key=cast(str | None, sr_mk),
                list_metadata_key=cast(str | None, lmk if lmk else None),
            )
        )
    gen: GenerationConfig | None = None
    if "generation_config" in data and data["generation_config"] is not None:
        gc = data["generation_config"]
        if not isinstance(gc, Mapping):
            msg = f"`generation_config` must be a mapping or null ({source})"
            raise ValueError(msg)
        gen = GenerationConfig(
            max_tokens=int(gc["max_tokens"]) if "max_tokens" in gc else 256,
            temperature=float(gc["temperature"]) if "temperature" in gc else 0.0,
            max_length_seconds=(
                int(gc["max_length_seconds"]) if "max_length_seconds" in gc else None
            ),
        )
    return PromptSpec(
        prompt_id=prompt_id.strip(),
        message_templates=tuple(pairs),
        audio_slots=tuple(slots),
        generation_config=gen,
    )


def _build_template_context(example: DatasetExample) -> dict[str, Any]:
    base = example.model_dump()
    labels = example.labels
    candidate_labels: list[str]
    if labels is None:
        candidate_labels = []
    elif isinstance(labels, str):
        candidate_labels = [p.strip() for p in labels.split(",") if p.strip()]
    elif isinstance(labels, list):
        candidate_labels = [str(x).strip() for x in labels if str(x).strip()]
    else:
        candidate_labels = []
    base["candidate_labels"] = candidate_labels
    base["audio_placeholder"] = AUDIO_PLACEHOLDER
    return base


def _get_dotted(mapping: Mapping[str, Any], dotted: str) -> object | None:
    cur: object = mapping
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _build_audio_inputs(
    example: DatasetExample,
    slots: tuple[AudioSlotSpec, ...],
) -> list[AudioInput]:
    out: list[AudioInput] = []
    meta = example.metadata
    for slot in slots:
        if slot.list_metadata_key:
            raw_list = _get_dotted(meta, slot.list_metadata_key)
            if not isinstance(raw_list, list) or not raw_list:
                msg = (
                    f"metadata[{slot.list_metadata_key!r}] must be a non-empty list "
                    f"for sample_id={example.sample_id!r}"
                )
                raise ValueError(msg)
            for j, item in enumerate(raw_list):
                if not isinstance(item, str) or not item.strip():
                    msg = (
                        f"metadata[{slot.list_metadata_key!r}][{j}] must be a "
                        f"non-empty str for sample_id={example.sample_id!r}"
                    )
                    raise ValueError(msg)
                out.append(
                    AudioInput(
                        payload_type=slot.payload_type,
                        data=item.strip(),
                        sample_rate=slot.sample_rate,
                    )
                )
            continue
        raw_data = _get_dotted(meta, slot.metadata_key)
        if not isinstance(raw_data, str) or not raw_data.strip():
            msg = (
                f"metadata[{slot.metadata_key!r}] must be a non-empty str for "
                f"sample_id={example.sample_id!r}"
            )
            raise ValueError(msg)
        data = raw_data.strip()
        sr: int | None = slot.sample_rate
        if slot.sample_rate_metadata_key is not None:
            sr_raw = _get_dotted(meta, slot.sample_rate_metadata_key)
            if sr_raw is not None:
                if not isinstance(sr_raw, int):
                    msg = (
                        f"metadata[{slot.sample_rate_metadata_key!r}] must be int "
                        f"or absent for sample_id={example.sample_id!r}"
                    )
                    raise ValueError(msg)
                sr = sr_raw
        out.append(
            AudioInput(
                payload_type=slot.payload_type,
                data=data,
                sample_rate=sr,
            )
        )
    return out


def _assert_audio_alignment(
    messages: list[ChatMessage],
    audio_inputs: list[AudioInput],
    *,
    sample_id: str,
) -> None:
    n_ph = sum(len(_AUDIO_PLACEHOLDER_RE.findall(m.content)) for m in messages)
    n_au = len(audio_inputs)
    if n_ph != n_au:
        msg = (
            f"Audio placeholder alignment failed for sample_id={sample_id!r}: "
            f"found {n_ph} placeholder(s) in rendered messages but built "
            f"{n_au} audio input(s). Each {AUDIO_PLACEHOLDER!r} must match one "
            f"audio slot in the prompt YAML (same count, left-to-right order "
            f"across messages)."
        )
        raise AudioPlaceholderAlignmentError(
            msg,
            sample_id=sample_id,
            n_placeholders=n_ph,
            n_audio_inputs=n_au,
        )
