"""Canonical pipeline and artifact data contracts.

This module defines the five primary Pydantic models used across
``DatasetLoader`` → ``HttpClient`` → ``ResultStore``. Shapes that mirror
the public HTTP contract use ``schema_version="predictions_v1"``; dataset
rows, scored rows, and run summaries use explicit ``beans_next.*`` schema
strings so JSONL artifacts can evolve without colliding with launcher wire
versions.

Raises
------
pydantic.ValidationError
    Raised when keyword/value data does not satisfy field types or
    constraints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PREDICTIONS_V1: Literal["predictions_v1"] = "predictions_v1"
DATASET_EXAMPLE_V1: Literal["beans_next.dataset_example.v1"] = (
    "beans_next.dataset_example.v1"
)
SCORED_PREDICTION_V1: Literal["beans_next.scored_prediction.v1"] = (
    "beans_next.scored_prediction.v1"
)
RUN_SUMMARY_V1: Literal["beans_next.run_summary.v1"] = "beans_next.run_summary.v1"


class ChatMessage(BaseModel):
    """One chat message in a ``ModelRequest`` ``messages`` list.

    Attributes
    ----------
    role
        Model role string (for example ``"system"`` or ``"user"``).
    content
        Message body, including any ``<Audio><AudioHere></Audio>`` tags
        aligned with ``audio_inputs`` ordering.

    Raises
    ------
    pydantic.ValidationError
        If ``role`` or ``content`` are not strings.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class AudioInput(BaseModel):
    """One audio payload aligned with the i-th ``<AudioHere>`` placeholder.

    Attributes
    ----------
    payload_type
        Transport encoding: ``base64_wav``, ``file_path``, or ``file_url``.
    data
        Base64 payload, filesystem path, or URL string, depending on
        ``payload_type``.
    sample_rate
        Audio sample rate in Hz when applicable (commonly required for
        ``base64_wav``).

    Raises
    ------
    pydantic.ValidationError
        If required fields are missing or invalid.
    """

    model_config = ConfigDict(extra="forbid")

    payload_type: Literal["base64_wav", "file_path", "file_url"]
    data: str
    sample_rate: int | None = None


class GenerationConfig(BaseModel):
    """Decoding parameters sent with a ``ModelRequest``.

    Attributes
    ----------
    max_tokens
        Maximum new tokens to generate.
    temperature
        Sampling temperature (``0.0`` requests greedy decoding when the
        server supports it).

    Raises
    ------
    pydantic.ValidationError
        If numeric fields are not finite numbers where required.
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 256
    temperature: float = 0.0
    max_length_seconds: int | None = None


class TokenUsage(BaseModel):
    """Optional token accounting returned on a prediction.

    Attributes
    ----------
    prompt_tokens
        Tokens in the prompt context, if reported by the server.
    completion_tokens
        Tokens generated for the completion, if reported.

    Raises
    ------
    pydantic.ValidationError
        If counts are not non-negative integers when provided.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class DatasetExample(BaseModel):
    """Normalized dataset row consumed by ``PromptRenderer``.

    This is a lightweight, JSONL-friendly row type. Ground truth may be a
    single label, a multi-label list, structured detection targets, or
    free-form reference text depending on the task.

    Attributes
    ----------
    schema_version
        Internal artifact schema string (not the HTTP wire schema).
    sample_id
        Stable unique id for pairing across requests, predictions, and
        scores.
    task_id
        Registry id for the eval task driving prompts and metrics, if
        known.
    split
        Dataset split or subset name (for example ``"test"``), if present.
    labels
        Ground truth label(s) or structured targets, task-dependent.
    metadata
        Additional loader-specific fields (paths, taxonomy ids, etc.).

    Raises
    ------
    pydantic.ValidationError
        If ``sample_id`` is missing or fields have wrong types.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["beans_next.dataset_example.v1"] = DATASET_EXAMPLE_V1
    sample_id: str
    task_id: str | None = None
    split: str | None = None
    labels: str | list[str] | dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(BaseModel):
    """Per-sample multimodal request matching ``predictions_v1`` batch items.

    This mirrors one element of the HTTP ``requests`` array (excluding the
    outer batch envelope). It is the in-core representation produced after
    prompt rendering and before ``HttpClient`` serialization.

    Attributes
    ----------
    schema_version
        Wire schema string; must be ``predictions_v1`` for iteration-1
        servers.
    sample_id
        Unique id within a batch; used for response matching.
    messages
        Chat turns including any audio placeholders.
    audio_inputs
        Audio payloads aligned with placeholders in ``messages``.
    generation_config
        Decoding parameters for the launcher.

    Raises
    ------
    pydantic.ValidationError
        If nested structures fail validation.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    sample_id: str
    messages: list[ChatMessage]
    audio_inputs: list[AudioInput]
    generation_config: GenerationConfig | None = None


class ModelPrediction(BaseModel):
    """Per-sample model output aligned with ``predictions_v1`` responses.

    Any field except ``sample_id`` and ``predictions`` may be omitted on the
    wire; optional fields here default to ``None``.

    Attributes
    ----------
    schema_version
        Wire schema string; must be ``predictions_v1``.
    sample_id
        Id pairing this row to the originating ``ModelRequest``.
    predictions
        One or more decoded strings (n-best); length ``1`` is typical.
    finish_reason
        Server-reported completion reason when available.
    usage
        Token accounting when the server provides it.
    latency_sec
        End-to-end latency for this sample when measured client-side or
        reported by the server.
    error
        Sample-level error string; ``None`` when the prediction succeeded.
    server_info
        Snapshot of launcher ``/info`` (or a subset) for reproducibility.

    Raises
    ------
    pydantic.ValidationError
        If fields have invalid types.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["predictions_v1"] = PREDICTIONS_V1
    sample_id: str
    predictions: list[str] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_sec: float | None = None
    error: str | None = None
    server_info: dict[str, Any] | None = None


class ScoredPrediction(BaseModel):
    """Post-processed and optionally scored row written to JSONL artifacts.

    ``processed_predictions.jsonl`` may omit ``scores``; ``scored_predictions.jsonl``
    should populate ``scores`` after deterministic metrics (and optional judges).

    Attributes
    ----------
    schema_version
        Internal artifact schema string.
    sample_id
        Id joining this row to ``DatasetExample`` / ``ModelPrediction``.
    task_id
        Eval task id used for metric selection, if known.
    predictions
        Raw model outputs (typically copied from ``ModelPrediction``).
    processed_prediction
        Parsed / cleaned representation used for metric computation.
    targets
        Ground truth derived from the dataset example, if stored denormalized.
    scores
        Metric name → numeric value when scoring has run.
    postprocess_version
        Version string participating in scoring-cache keys.
    error
        Propagated sample-level error, if any.

    Raises
    ------
    pydantic.ValidationError
        If fields have invalid types or ``scores`` maps to non-float values when
        provided.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["beans_next.scored_prediction.v1"] = SCORED_PREDICTION_V1
    sample_id: str
    task_id: str | None = None
    predictions: list[str] = Field(default_factory=list)
    processed_prediction: str | list[str] | None = None
    targets: str | list[str] | dict[str, Any] | None = None
    scores: dict[str, float] | None = None
    postprocess_version: str | None = None
    error: str | None = None


class RunSummary(BaseModel):
    """Aggregate record written to ``summary.json`` for a benchmark run.

    Attributes
    ----------
    schema_version
        Internal artifact schema string.
    run_id
        Unique directory key for this run.
    library_version
        Installed ``beans_next`` package version string.
    code_git_sha
        Repository commit when available.
    run_config_hash
        Hash of the resolved run configuration for reproducibility.
    prompt_version
        Prompt template version string when tracked.
    postprocess_version
        Post-processing pipeline version string when tracked.
    scorer_versions
        Metric or scorer name → version string.
    model_identity
        Launcher identity payload (often mirroring ``/info`` fields).
    seed
        Global RNG seed when the run is seeded.
    n_samples
        Total samples attempted in the run scope.
    n_errors
        Count of samples ending with a recorded error.
    metrics
        Aggregated metric payload (structure is task/run specific).
    task_results
        Optional per-task nested summaries.

    Raises
    ------
    pydantic.ValidationError
        If required counters are inconsistent types or missing required fields.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["beans_next.run_summary.v1"] = RUN_SUMMARY_V1
    run_id: str
    library_version: str
    code_git_sha: str | None = None
    run_config_hash: str | None = None
    prompt_version: str | None = None
    postprocess_version: str | None = None
    scorer_versions: dict[str, str] | None = None
    model_identity: dict[str, Any]
    seed: int | None = None
    n_samples: int = 0
    n_errors: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    task_results: dict[str, Any] | None = None
