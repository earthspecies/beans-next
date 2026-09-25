"""Eval-task YAML schema + loader.

This module defines the validated shape of an eval-task registry entry (one task).
It intentionally stays lightweight and local-file-only: it reads bundled YAML files
from `beans_next/registry/` and performs no network calls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalTaskConfigError(ValueError):
    """Raised when an eval-task config cannot be loaded or validated."""


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


class EvalTaskConfig(BaseModel):
    """Validated eval-task configuration (resolved)."""

    task_type: str
    hf_path: str | None = None
    subset: str
    split: str
    prompt: str
    metrics: list[MetricSpec] = Field(min_length=1)
    labels: list[str] | None = None

    model_config = {"extra": "allow"}
