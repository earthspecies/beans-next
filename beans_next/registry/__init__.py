"""YAML-backed registry loader for bundled benchmark assets.

This package reads the YAML documents shipped under `beans_next/registry/` and
exposes a minimal typed container used by CLI and smoke checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Registry"]


def _registry_root() -> Path:
    return Path(__file__).resolve().parent


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"Registry YAML must be a mapping: {path}")
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(k, str) and k.strip():
            out[k] = v
    return out


def _load_kind(kind: str) -> dict[str, Any]:
    root = _registry_root() / kind
    if not root.is_dir():
        return {}
    merged: dict[str, Any] = {}
    for path in sorted(root.glob("*.yaml")):
        merged.update(_load_yaml_mapping(path))
    return merged


@dataclass(frozen=True)
class Registry:
    """Loaded view of the bundled registry YAMLs."""

    datasets: dict[str, Any]
    eval_tasks: dict[str, Any]
    suites: dict[str, Any]

    @classmethod
    def load(cls) -> "Registry":
        """Load all registry kinds shipped with the package.

        Returns
        -------
        Registry
            Loaded registry container.
        """
        return cls(
            datasets=_load_kind("dataset"),
            eval_tasks=_load_kind("eval_task"),
            suites=_load_kind("suite"),
        )
