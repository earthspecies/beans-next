"""Eval-task YAML schema + loader.

This module defines the validated shape of an eval-task registry entry (one task).
It intentionally stays lightweight and local-file-only: it reads bundled YAML files
from `beans_next/registry/` and performs no network calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class EvalTaskConfigError(ValueError):
    """Raised when an eval-task config cannot be loaded or validated."""


class JudgeRegistryResolutionError(EvalTaskConfigError):
    """Raised when a judge registry reference cannot be resolved."""


class MetricSpec(BaseModel):
    """One metric entry from an eval-task YAML.

    Parameters
    ----------
    name
        Metric registry name (e.g., `accuracy`, `f1`, `cider`).
    scorer_kwargs
        Optional kwargs passed to the metric/scorer implementation.
    """

    name: str
    scorer_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class JudgeConfig(BaseModel):
    """Resolved judge configuration.

    Parameters
    ----------
    template_id
        Judge rubric template id (see `beans_next.judges.base.list_judge_templates`).
    description
        Optional human-readable description.
    """

    template_id: str
    description: str | None = None

    model_config = {"extra": "allow"}


class EvalTaskConfig(BaseModel):
    """Validated eval-task configuration (resolved).

    The `judge` field is optional. When present, it can be:

    - a string registry id (resolved from `registry/judge/<id>.yaml`)
    - a mapping with `id: <registry id>`
    - a mapping with `inline: {template_id: ...}`
    - an inline mapping `{template_id: ...}`
    """

    task_type: str
    hf_path: str | None = None
    subset: str
    split: str
    prompt: str
    metrics: list[MetricSpec] = Field(min_length=1)
    labels: list[str] | None = None
    judge: JudgeConfig | None = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _coerce_and_resolve_judge(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        raw = data.get("judge")
        if raw is None:
            return data

        resolved: JudgeConfig
        if isinstance(raw, str):
            resolved = load_judge_preset(raw)
        elif isinstance(raw, dict):
            if "id" in raw:
                judge_id = raw.get("id")
                if not isinstance(judge_id, str):
                    raise EvalTaskConfigError("`judge.id` must be a string.")
                resolved = load_judge_preset(judge_id)
            elif "inline" in raw:
                inline = raw.get("inline")
                try:
                    resolved = JudgeConfig.model_validate(inline)
                except ValidationError as exc:
                    msg = f"Invalid `judge.inline` block:\n{exc}"
                    raise EvalTaskConfigError(msg) from exc
            else:
                try:
                    resolved = JudgeConfig.model_validate(raw)
                except ValidationError as exc:
                    raise EvalTaskConfigError(f"Invalid `judge` block:\n{exc}") from exc
        else:
            raise EvalTaskConfigError("`judge` must be a string or mapping.")

        out = dict(data)
        out["judge"] = resolved
        return out


def load_judge_preset(judge_id: str) -> JudgeConfig:
    """Load a bundled judge registry entry by id.

    Parameters
    ----------
    judge_id
        Judge registry id (YAML filename stem under `registry/judge/`).

    Returns
    -------
    JudgeConfig
        Validated judge config.

    Raises
    ------
    JudgeRegistryResolutionError
        If the registry entry is missing or malformed.
    """

    path = _registry_root() / "judge" / f"{judge_id}.yaml"
    if not path.exists():
        raise JudgeRegistryResolutionError(f"Unknown judge id: {judge_id}")

    data = _read_yaml_file(path)
    if not isinstance(data, dict) or judge_id not in data:
        msg = (
            "Malformed judge registry entry: expected top-level key "
            f"{judge_id!r} in {path}"
        )
        raise JudgeRegistryResolutionError(msg)
    payload = data[judge_id]
    if not isinstance(payload, dict):
        msg = (
            "Malformed judge registry entry: expected mapping for "
            f"{judge_id!r} in {path}"
        )
        raise JudgeRegistryResolutionError(msg)
    try:
        return JudgeConfig.model_validate(payload)
    except ValidationError as exc:
        raise JudgeRegistryResolutionError(
            f"Invalid judge registry entry for id {judge_id!r}: {path}\n{exc}"
        ) from exc


def _registry_root() -> Path:
    import importlib.resources

    return Path(importlib.resources.files("beans_next")).joinpath("registry")


def _read_yaml_file(path: Path) -> object:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        msg = f"Failed to read registry YAML: {path}"
        raise JudgeRegistryResolutionError(msg) from exc
    try:
        return yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:  # pragma: no cover
        msg = f"Invalid YAML in registry file: {path}"
        raise JudgeRegistryResolutionError(msg) from exc
