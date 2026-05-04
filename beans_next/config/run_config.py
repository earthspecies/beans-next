"""Run-config YAML schema + loader.

This module implements the configuration shape consumed by `beans-next run --config`.
It validates user-provided YAML with Pydantic and resolves registry references into a
flat, deterministic execution plan (models × tasks) without performing any network
calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from beans_next.config.eval_task import EvalTaskConfig


class RunConfigError(ValueError):
    """Raised when a run-config file cannot be loaded or resolved."""


class RegistryResolutionError(RunConfigError):
    """Raised when a registry reference cannot be resolved."""


class ModelEndpointConfig(BaseModel):
    """Concrete model endpoint configuration (resolved).

    Parameters
    ----------
    name
        Stable identifier for this endpoint preset.
    description
        Optional human-readable description.
    predict_url
        Full URL of the launcher's `POST /predict` endpoint.
    info_url
        Full URL of the launcher's `GET /info` endpoint.
    health_url
        Full URL of the launcher's `GET /health` endpoint.
    response_schema
        Response schema identifier. For v1 this must be `predictions_v1`.
    audio_payload
        Audio transport mode (e.g., `base64_wav`).
    auth
        Auth config block (model launcher dependent). Empty mapping means no auth.
    retry_policy
        Optional retry policy mapping passed through to
        :class:`~beans_next.models.http.HttpClient` construction. When set, this
        overrides the default retry behavior for this endpoint.
    """

    name: str
    description: str | None = None
    predict_url: str
    info_url: str
    health_url: str
    response_schema: Literal["predictions_v1"] = "predictions_v1"
    audio_payload: str = "base64_wav"
    auth: dict[str, Any] = Field(default_factory=dict)
    retry_policy: dict[str, Any] | None = None


class ModelEndpointRef(BaseModel):
    """Reference to a model endpoint, by registry id or inline config."""

    id: str | None = None
    inline: ModelEndpointConfig | None = None

    @model_validator(mode="after")
    def _validate_one_of(self) -> Self:
        if (self.id is None) == (self.inline is None):
            raise ValueError("Expected exactly one of `id` or `inline`.")
        return self


class RunConfig(BaseModel):
    """User-facing run-config shape.

    Parameters
    ----------
    models
        One or more model endpoint references. Each entry may be a registry id string,
        or an object with `id`, or an object with `inline`.
    suite
        Suite id from `registry/suite/`. Mutually exclusive with `tasks`.
    tasks
        Explicit list of eval-task ids from `registry/eval_task/`. Mutually exclusive
        with `suite`.
    limit
        Optional cap applied by the runner (I10-B consumes this).
    output_dir
        Optional output directory override (I10-B consumes this).
    run_id
        Optional run id override (I10-B consumes this).
    """

    models: list[str | ModelEndpointRef] = Field(min_length=1)
    suite: str | None = None
    tasks: list[str] | None = None
    limit: int | None = Field(default=None, ge=1)
    output_dir: str | None = None
    run_id: str | None = None
    data_source: Literal["hf", "esp_data", "huggingface"] | None = None

    @model_validator(mode="after")
    def _validate_suite_xor_tasks(self) -> Self:
        if (self.suite is None) == (self.tasks is None):
            raise ValueError("Expected exactly one of `suite` or `tasks`.")
        if self.tasks is not None and len(self.tasks) == 0:
            raise ValueError("`tasks` must be a non-empty list.")
        return self


@dataclass(frozen=True, slots=True)
class ExecutionItem:
    """One concrete (model endpoint, eval task) pair for execution."""

    model: ModelEndpointConfig
    eval_task_id: str
    eval_task: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedRunConfig:
    """Resolved run-config suitable for the runner hook (I10-B).

    Attributes
    ----------
    source_path
        Path used to load the YAML file.
    config
        Parsed `RunConfig` (validated, but still contains refs).
    models
        Resolved endpoint configs, in deterministic order matching `config.models`.
    eval_task_ids
        Resolved eval-task ids, in deterministic order matching the suite/task list.
    plan
        Flat execution plan in deterministic order: models-major then tasks-minor.
    """

    source_path: Path
    config: RunConfig
    models: list[ModelEndpointConfig]
    eval_task_ids: list[str]
    plan: list[ExecutionItem]


def load_run_config(path: str | Path) -> LoadedRunConfig:
    """Load and resolve a run-config YAML into a flat execution plan.

    Parameters
    ----------
    path
        Path to a YAML file following the `RunConfig` schema.

    Returns
    -------
    LoadedRunConfig
        Resolved run-config with a flat models×tasks plan.

    Raises
    ------
    RunConfigError
        If the YAML cannot be read/parsed/validated, or if any registry id cannot be
        resolved to a bundled registry entry.
    """

    source_path = Path(path)
    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        raise RunConfigError(f"Failed to read run-config file: {source_path}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise RunConfigError(f"Invalid YAML in run-config file: {source_path}") from exc

    try:
        config = RunConfig.model_validate(data)
    except ValidationError as exc:
        raise RunConfigError(f"Invalid run-config: {source_path}\n{exc}") from exc

    models = [_resolve_model_ref(m) for m in config.models]
    eval_task_ids = _resolve_task_ids(config)
    plan = _build_plan(models=models, eval_task_ids=eval_task_ids)

    return LoadedRunConfig(
        source_path=source_path,
        config=config,
        models=models,
        eval_task_ids=eval_task_ids,
        plan=plan,
    )


def _build_plan(
    *,
    models: list[ModelEndpointConfig],
    eval_task_ids: list[str],
) -> list[ExecutionItem]:
    plan: list[ExecutionItem] = []
    for model in models:
        for eval_task_id in eval_task_ids:
            plan.append(
                ExecutionItem(
                    model=model,
                    eval_task_id=eval_task_id,
                    eval_task=_load_eval_task(eval_task_id),
                )
            )
    return plan


def _resolve_model_ref(model: str | ModelEndpointRef) -> ModelEndpointConfig:
    if isinstance(model, str):
        return _load_model_preset(model)
    if model.inline is not None:
        return model.inline
    if model.id is not None:
        return _load_model_preset(model.id)
    raise AssertionError("ModelEndpointRef invariant violated.")  # pragma: no cover


def _resolve_task_ids(config: RunConfig) -> list[str]:
    if config.tasks is not None:
        return list(config.tasks)

    if config.suite is None:
        raise AssertionError("RunConfig invariant violated.")  # pragma: no cover

    return _load_suite_eval_tasks(config.suite)


def _registry_root() -> Path:
    # Runtime-safe path without relying on pkgutil/pkg_resources.
    # In editable installs, this resolves to the workspace package directory.
    import importlib.resources

    return Path(importlib.resources.files("beans_next")).joinpath("registry")


def _load_model_preset(model_id: str) -> ModelEndpointConfig:
    path = _registry_root() / "model" / f"{model_id}.yaml"
    if not path.exists():
        raise RegistryResolutionError(f"Unknown model id: {model_id}")

    data = _read_yaml_file(path)
    try:
        return ModelEndpointConfig.model_validate(data)
    except ValidationError as exc:
        raise RegistryResolutionError(
            f"Invalid model registry entry for id {model_id!r}: {path}\n{exc}"
        ) from exc


def _load_eval_task(eval_task_id: str) -> dict[str, Any]:
    path = _registry_root() / "eval_task" / f"{eval_task_id}.yaml"
    if not path.exists():
        raise RegistryResolutionError(f"Unknown eval_task id: {eval_task_id}")

    data = _read_yaml_file(path)
    if not isinstance(data, dict) or eval_task_id not in data:
        raise RegistryResolutionError(
            f"Malformed eval_task registry entry: expected top-level key "
            f"{eval_task_id!r} in {path}"
        )
    payload = data[eval_task_id]
    if not isinstance(payload, dict):
        raise RegistryResolutionError(
            f"Malformed eval_task registry entry: expected mapping for "
            f"{eval_task_id!r} in {path}"
        )
    try:
        cfg = EvalTaskConfig.model_validate(payload)
    except ValidationError as exc:
        raise RegistryResolutionError(
            f"Invalid eval_task registry entry for id {eval_task_id!r}: {path}\n{exc}"
        ) from exc
    except ValueError as exc:
        raise RegistryResolutionError(
            f"Invalid eval_task registry entry for id {eval_task_id!r}: {path}\n{exc}"
        ) from exc
    return cfg.model_dump(exclude_none=True)


def _load_suite_eval_tasks(suite_id: str) -> list[str]:
    path = _registry_root() / "suite" / f"{suite_id}.yaml"
    if not path.exists():
        raise RegistryResolutionError(f"Unknown suite id: {suite_id}")

    data = _read_yaml_file(path)
    if not isinstance(data, dict) or suite_id not in data:
        raise RegistryResolutionError(
            f"Malformed suite registry entry: expected top-level key {suite_id!r} "
            f"in {path}"
        )
    payload = data[suite_id]
    if not isinstance(payload, dict):
        raise RegistryResolutionError(
            f"Malformed suite registry entry: expected mapping for {suite_id!r} "
            f"in {path}"
        )

    eval_tasks = payload.get("eval_tasks")
    if not isinstance(eval_tasks, list) or not all(
        isinstance(x, str) for x in eval_tasks
    ):
        raise RegistryResolutionError(
            f"Malformed suite registry entry: expected `eval_tasks: [..]` in {path}"
        )
    if len(eval_tasks) == 0:
        raise RegistryResolutionError(f"Suite has no eval_tasks: {suite_id}")
    return list(eval_tasks)


def _read_yaml_file(path: Path) -> object:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        raise RegistryResolutionError(f"Failed to read registry YAML: {path}") from exc
    try:
        return yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:  # pragma: no cover
        raise RegistryResolutionError(f"Invalid YAML in registry file: {path}") from exc
