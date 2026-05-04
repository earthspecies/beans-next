"""Benchmark runner package (incremental build-out)."""

from beans_next.runner._utils import aggregate_score_means, per_task_score_means
from beans_next.runner.batching import (
    DEFAULT_MAX_BATCH_FALLBACK,
    effective_max_batch_size,
    iter_batches,
)
from beans_next.runner.health import ensure_launcher_ready, probe_health
from beans_next.runner.runner import BenchmarkRunner, RunnerConfig, ScorerFn

__all__ = [
    "BenchmarkRunner",
    "DEFAULT_MAX_BATCH_FALLBACK",
    "RunnerConfig",
    "ScorerFn",
    "aggregate_score_means",
    "effective_max_batch_size",
    "ensure_launcher_ready",
    "iter_batches",
    "per_task_score_means",
    "probe_health",
]
