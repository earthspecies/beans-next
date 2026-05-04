"""Configuration schemas + loaders.

Currently includes:

- Run-config loader for `beans-next run --config`
- Eval-task schema with optional `judge` support (for registry validation)
"""

from beans_next.config.eval_task import EvalTaskConfig, JudgeConfig, load_judge_preset
from beans_next.config.run_config import (
    ExecutionItem,
    LoadedRunConfig,
    ModelEndpointConfig,
    ModelEndpointRef,
    RegistryResolutionError,
    RunConfig,
    RunConfigError,
    load_run_config,
)

__all__ = [
    "EvalTaskConfig",
    "ExecutionItem",
    "JudgeConfig",
    "LoadedRunConfig",
    "ModelEndpointConfig",
    "ModelEndpointRef",
    "RegistryResolutionError",
    "RunConfig",
    "RunConfigError",
    "load_judge_preset",
    "load_run_config",
]
