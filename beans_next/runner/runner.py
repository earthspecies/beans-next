"""End-to-end benchmark execution: HTTP inference, post-process, scoring, artifacts."""

from __future__ import annotations

import base64
import importlib.metadata
import json
import logging
import os
import sys
import time
from argparse import Namespace
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

import yaml

from beans_next.api.http_schemas import (
    HttpAudioInput,
    HttpChatMessage,
    HttpGenerationConfig,
    PredictionsV1Request,
    PredictionsV1RequestItem,
    PredictionsV1ResponseItem,
)
from beans_next.api.types import (
    DatasetExample,
    ModelPrediction,
    ModelRequest,
    RunSummary,
    ScoredPrediction,
    TokenUsage,
)
from beans_next.cache.two_layer import TwoLayerRunCache, scoring_cache_key
from beans_next.judges.scorer import JudgeScorer
from beans_next.models.http import HttpClient
from beans_next.post_process.pipeline import (
    PostProcessPipelineError,
    PostProcessResult,
    StepSpec,
    run_post_process_pipeline,
)
from beans_next.prompts.renderer import PromptRenderer, PromptSpec
from beans_next.results.store import BenchmarkArtifactWriter
from beans_next.runner._utils import (
    aggregate_score_means,
    compute_dataset_level_metrics,
    per_task_score_means,
)
from beans_next.runner.batching import effective_max_batch_size, iter_batches
from beans_next.runner.checkpoint import (
    completed_sample_ids_from_checkpoint_json,
    read_checkpoint_json,
)
from beans_next.runner.parallel import map_ordered

__all__ = [
    "BenchmarkRunner",
    "RunnerConfig",
    "ScorerFn",
    "run_from_cli_namespace",
]

_logger = logging.getLogger(__name__)

ScorerFn: TypeAlias = Callable[
    [DatasetExample, PostProcessResult, ModelPrediction],
    dict[str, float],
]


class _MetricsScoreSampleFn(Protocol):
    """Optional ``beans_next.metrics.score_sample`` hook (implemented in I3-A)."""

    def __call__(
        self,
        example: DatasetExample,
        *,
        post: PostProcessResult,
        raw_predictions: list[str],
        task_type: str | None = None,
    ) -> Mapping[str, float]:
        ...


_DEFAULT_WIRE_SAMPLE_RATE_HZ: int = 16_000


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for :class:`BenchmarkRunner` artifact layout and post-process.

    Attributes
    ----------
    output_dir
        Directory receiving JSONL / JSON artifacts.
    run_id
        Stable identifier for this run (directory names, ``RunSummary.run_id``).
    parser_steps
        Post-process parser :class:`~beans_next.post_process.pipeline.StepSpec` rows.
    cleaner_steps
        Post-process cleaner steps run after parsers.
    postprocess_version
        Optional version string stored on scored rows and the run summary.
    prompt_version
        Optional override for prompt version metadata (defaults to renderer spec id).
    seed
        Optional RNG seed recorded in the summary.
    scorer_versions
        Optional mapping of scorer name to version string for reproducibility.
    run_config_hash
        Optional hash of resolved YAML/config for reproducibility.
    code_git_sha
        Optional repository revision string for reproducibility.
    workers
        CPU-side worker threads used for per-sample post-process and scoring. This
        does not change HTTP inference behavior (still batched requests).
    cache_dir
        Optional directory holding ``inference.sqlite`` and ``scoring.sqlite``.
        When ``None``, no disk caching is performed (iteration-1 default).
    task_type
        Optional task type string (e.g. ``"classification"``, ``"detection"``,
        ``"captioning"``).  When set, passed to ``score_sample`` so it takes
        precedence over per-example metadata, and used to select which
        dataset-level metric to include in the run summary.
    gcs_upload_prefix
        Optional GCS destination prefix (``gs://<bucket>/<path>``) under which
        all run artifacts are uploaded once :meth:`~BenchmarkRunner.run`
        completes.  The prefix should already include the run-specific path
        component (e.g. ``gs://my-bucket/predictions/my-run-id``).  When
        ``None`` (default) no upload is performed.
    """

    output_dir: Path
    run_id: str
    parser_steps: tuple[StepSpec, ...] = ()
    cleaner_steps: tuple[StepSpec, ...] = ()
    postprocess_version: str | None = None
    prompt_version: str | None = None
    seed: int | None = None
    scorer_versions: dict[str, str] | None = None
    run_config_hash: str | None = None
    code_git_sha: str | None = None
    resume: bool = False
    workers: int = 1
    cache_dir: Path | None = None
    task_type: str | None = None
    gcs_upload_prefix: str | None = None


def _package_version() -> str:
    try:
        return importlib.metadata.version("beans-next")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _wire_sample_rate_hz(audio: ModelRequest) -> int:
    """Pick a concrete sample rate for wire ``HttpAudioInput`` rows.

    Returns
    -------
    int
        First positive ``sample_rate`` among ``audio.audio_inputs``, else a
        conservative default when the list is empty or unset.
    """
    if not audio.audio_inputs:
        return _DEFAULT_WIRE_SAMPLE_RATE_HZ
    sr0 = audio.audio_inputs[0].sample_rate
    if isinstance(sr0, int) and sr0 > 0:
        return sr0
    return _DEFAULT_WIRE_SAMPLE_RATE_HZ


def model_request_to_wire_item(model_request: ModelRequest) -> PredictionsV1RequestItem:
    """Convert a core :class:`~beans_next.api.types.ModelRequest` to wire item shape.

    Parameters
    ----------
    model_request
        Rendered per-sample request.

    Returns
    -------
    PredictionsV1RequestItem
        Item suitable for :class:`~beans_next.api.http_schemas.PredictionsV1Request`.

    Raises
    ------
    ValueError
        If any ``audio_inputs`` row is missing a resolvable positive ``sample_rate``
        and no slot provides a usable default.
    """
    gen = model_request.generation_config
    if gen is None:
        http_gen = HttpGenerationConfig(max_tokens=256, temperature=0.0)
    else:
        http_gen = HttpGenerationConfig(
            max_tokens=gen.max_tokens,
            temperature=gen.temperature,
            max_length_seconds=gen.max_length_seconds,
        )
    default_sr = _wire_sample_rate_hz(model_request)
    audio_rows: list[HttpAudioInput] = []
    for slot in model_request.audio_inputs:
        sr = slot.sample_rate if slot.sample_rate is not None else default_sr
        if not isinstance(sr, int) or sr < 1:
            msg = (
                f"Invalid sample_rate for sample_id={model_request.sample_id!r}: "
                f"{slot.sample_rate!r}"
            )
            raise ValueError(msg)
        payload_type = slot.payload_type
        data = slot.data

        # If the request was rendered with a local file path (common for HF rows),
        # but the model server is remote, the server can't read that path.
        # Transparently convert local WAV paths to base64_wav payloads.
        if payload_type == "file_path":
            try:
                p = Path(str(data))
            except Exception:  # noqa: BLE001
                p = None
            if p is not None and p.exists() and p.is_file():
                wav_bytes = p.read_bytes()
                payload_type = "base64_wav"
                data = base64.b64encode(wav_bytes).decode("ascii")

        audio_rows.append(
            HttpAudioInput(
                payload_type=payload_type,
                data=data,
                sample_rate=sr,
            )
        )
    return PredictionsV1RequestItem(
        sample_id=model_request.sample_id,
        messages=[
            HttpChatMessage(role=m.role, content=m.content)
            for m in model_request.messages
        ],
        audio_inputs=audio_rows,
        generation_config=http_gen,
    )


def wire_response_item_to_model_prediction(
    item: PredictionsV1ResponseItem,
    *,
    server_info: dict[str, Any] | None,
) -> ModelPrediction:
    """Map one wire response row to :class:`~beans_next.api.types.ModelPrediction`.

    Returns
    -------
    ModelPrediction
        Core prediction row including optional usage and ``server_info`` snapshot.
    """
    usage: TokenUsage | None = None
    if item.usage is not None:
        usage = TokenUsage(
            prompt_tokens=item.usage.prompt_tokens,
            completion_tokens=item.usage.completion_tokens,
        )
    snap = dict(server_info) if server_info is not None else None
    return ModelPrediction(
        sample_id=item.sample_id,
        predictions=list(item.predictions),
        finish_reason=item.finish_reason,
        usage=usage,
        latency_sec=item.latency_sec,
        error=item.error,
        server_info=snap,
    )


def _raw_prediction_text(pred: ModelPrediction) -> str:
    if pred.predictions:
        return pred.predictions[0]
    return ""


def _merge_row_error(pred: ModelPrediction, post_err: str | None) -> str | None:
    if pred.error:
        return pred.error
    return post_err


def _targets_from_example(
    example: DatasetExample,
) -> str | list[str] | dict[str, Any] | None:
    return example.labels


class BenchmarkRunner:
    """CPU-side orchestration for one model endpoint and one prompt template.

    Executes, in order:

    ``DatasetExample`` → :class:`~beans_next.prompts.renderer.PromptRenderer`
    → :class:`~beans_next.api.types.ModelRequest`
    → :class:`~beans_next.models.http.HttpClient`
    → :class:`~beans_next.api.types.ModelPrediction` → post-process
    → (optional) ``beans_next.metrics.score_sample``
    → :class:`~beans_next.api.types.ScoredPrediction`
    → (optional) :class:`~beans_next.judges.scorer.JudgeScorer` over all items
    → on-disk JSONL / JSON artifacts.

    Scoring uses the optional ``scorer`` callback when provided; otherwise the
    runner attempts ``from beans_next.metrics import score_sample`` and records
    empty ``scores`` when that module is absent.

    When ``judge`` is provided,
    :meth:`~beans_next.judges.scorer.JudgeScorer.score_batch` is called once after all
    inference and primary scoring complete. Results are written
    to ``judge_outputs.jsonl`` in the output directory. Errored samples are excluded
    from the judge batch.

    Parameters
    ----------
    client
        Connected :class:`~beans_next.models.http.HttpClient` (``/info`` probed if
        configured on the client).
    renderer
        Prompt renderer bound to a single
        :class:`~beans_next.prompts.renderer.PromptSpec`.
    config
        Output paths, post-process steps, and reproducibility metadata.
    scorer
        Optional override for metric computation.
    judge
        Optional :class:`~beans_next.judges.scorer.JudgeScorer` for LLM-as-judge
        scoring. When set, ``judge_outputs.jsonl`` is written after the run.

    Notes
    -----
    HTTP calls may raise :exc:`~beans_next.models.http.HttpClientFatalError` from
    :class:`~beans_next.models.http.HttpClient`.

    Raises
    ------
    ValueError
        From prompt rendering, post-process, or wire conversion when inputs are invalid.
    """

    def __init__(
        self,
        client: HttpClient,
        renderer: PromptRenderer,
        config: RunnerConfig,
        *,
        scorer: ScorerFn | None = None,
        judge: JudgeScorer | None = None,
    ) -> None:
        self._client = client
        self._renderer = renderer
        self._config = config
        if self._config.workers < 1:
            raise ValueError(
                f"RunnerConfig.workers must be >= 1, got {self._config.workers}"
            )
        self._scorer = scorer
        self._judge = judge
        self._metrics_score_sample: _MetricsScoreSampleFn | None
        if scorer is not None:
            self._metrics_score_sample = None
        else:
            try:
                mod = importlib.import_module("beans_next.metrics")
                raw_fn = getattr(mod, "score_sample", None)
                self._metrics_score_sample = (
                    cast(_MetricsScoreSampleFn, raw_fn) if callable(raw_fn) else None
                )
            except ImportError:
                self._metrics_score_sample = None
        self._postprocess_fingerprint_value: str = (
            self._compute_postprocess_fingerprint()
        )

    def _compute_postprocess_fingerprint(self) -> str:
        """Return a stable fingerprint of parser/cleaner configuration.

        Returns
        -------
        str
            Canonical JSON string used inside scoring-cache keys.
        """

        def rows(steps: tuple[StepSpec, ...]) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for step in steps:
                out.append(
                    (
                        step.name,
                        json.dumps(
                            dict(step.params),
                            sort_keys=True,
                            default=str,
                            separators=(",", ":"),
                        ),
                    )
                )
            return out

        return json.dumps(
            {
                "parser": rows(self._config.parser_steps),
                "cleaner": rows(self._config.cleaner_steps),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def client(self) -> HttpClient:
        """HTTP client used for ``POST /predict``."""
        return self._client

    @property
    def config(self) -> RunnerConfig:
        """Frozen runner configuration."""
        return self._config

    def run(self, examples: Iterable[DatasetExample]) -> RunSummary:
        """Run inference and scoring, writing JSONL / JSON artifacts.

        Parameters
        ----------
        examples
            Dataset rows to evaluate (order normalized for determinism).

        Returns
        -------
        RunSummary
            Aggregate counters and mean metric payload. When round-trip times
            are recorded, ``summary.metrics["latency"]`` is populated.

        Raises
        ------
        TypeError
            When ``RunnerConfig.cache_dir`` is set but the inference client has no
            ``predict_url`` string attribute (required to namespace cache keys).

        Notes
        -----
        May raise :exc:`ValueError` from rendering, wire conversion, or post-process
        steps. Propagates :exc:`~beans_next.models.http.HttpClientFatalError` from
        :meth:`~beans_next.models.http.HttpClient.generate`. When ``cache_dir`` is
        unset, behavior matches pre-I6-A runners (no SQLite).
        """
        all_rows = sorted(list(examples), key=lambda e: e.sample_id)
        out = self._config.output_dir
        resume_completed: set[str] = set()
        if self._config.resume:
            checkpoint_path = out / "checkpoint.json"
            if checkpoint_path.is_file():
                resume_completed = completed_sample_ids_from_checkpoint_json(
                    checkpoint_path
                )
        rows = [r for r in all_rows if r.sample_id not in resume_completed]
        batch_size = effective_max_batch_size(self._client.server_info)
        completed: list[str] = sorted(resume_completed)
        score_rows: list[dict[str, float]] = []
        # (processed_prediction, targets) pairs for dataset-level metrics.
        processed_pairs: list[tuple[str, Any]] = []
        error_state = [0]
        judge_inputs: list[tuple[DatasetExample, str]] = []
        round_trip_times: list[float] = []
        failures: list[dict[str, Any]] = []

        run_cache: TwoLayerRunCache | None = None
        if self._config.cache_dir is not None:
            predict_url_val = getattr(self._client, "predict_url", None)
            if not isinstance(predict_url_val, str) or not predict_url_val:
                msg = (
                    "RunnerConfig.cache_dir is set but the inference client has no "
                    "string predict_url; caching requires a stable launcher URL."
                )
                raise TypeError(msg)
            server_info = self._client.server_info or {}
            run_cache = TwoLayerRunCache.open(
                self._config.cache_dir,
                predict_url=predict_url_val,
                model_revision=server_info.get("model_revision"),
            )

        _pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self._config.workers)
            if self._config.workers > 1
            else None
        )
        try:
            with BenchmarkArtifactWriter(out) as writer:
                for batch in iter_batches(rows, batch_size):
                    self._run_one_batch(
                        list(batch),
                        writer=writer,
                        completed_ids=completed,
                        score_rows=score_rows,
                        processed_pairs=processed_pairs,
                        error_state=error_state,
                        run_cache=run_cache,
                        judge_inputs=judge_inputs if self._judge is not None else None,
                        round_trip_times=round_trip_times,
                        failures=failures,
                        executor=_pool,
                    )
                    writer.write_checkpoint(
                        {
                            "schema_version": "beans_next.checkpoint.v1",
                            "run_id": self._config.run_id,
                            "completed_sample_ids": sorted(completed),
                            "n_predictions_written": len(completed),
                            "n_errors": error_state[0],
                            "failures": failures,
                        },
                    )

                if self._judge is not None and judge_inputs:
                    ex_list = [pair[0] for pair in judge_inputs]
                    text_list = [pair[1] for pair in judge_inputs]
                    judge_results = self._judge.score_batch(ex_list, text_list)
                    writer.write_judge_outputs(
                        [r.model_dump(mode="json") for r in judge_results]
                    )

                summary = self._build_summary_from_artifacts_if_present(
                    fallback_rows=all_rows,
                    score_rows=score_rows,
                    processed_pairs=processed_pairs,
                    n_errors=error_state[0],
                    round_trip_times=round_trip_times,
                )
                writer.write_summary(summary)
                writer.write_model_identity(dict(self._client.server_info or {}))
            if self._config.gcs_upload_prefix:
                from beans_next.results.gcs_upload import upload_run_artifacts

                uploaded = upload_run_artifacts(out, self._config.gcs_upload_prefix)
                _logger.info("Uploaded %d artifact(s) to GCS", len(uploaded))
            return summary
        finally:
            if _pool is not None:
                _pool.shutdown(wait=True)
            if run_cache is not None:
                run_cache.close()

    def _build_summary_from_artifacts_if_present(
        self,
        *,
        fallback_rows: list[DatasetExample],
        score_rows: list[dict[str, float]],
        processed_pairs: list[tuple[str, Any]] | None = None,
        n_errors: int,
        round_trip_times: list[float] | None = None,
    ) -> RunSummary:
        scored_path = self._config.output_dir / "scored_predictions.jsonl"
        if not self._config.resume or not scored_path.is_file():
            return self._build_summary(
                fallback_rows,
                score_rows=score_rows,
                processed_pairs=processed_pairs,
                n_errors=n_errors,
                round_trip_times=round_trip_times,
            )

        try:
            scored_ids: list[str] = []
            scored_scores: list[dict[str, float]] = []
            scored_task_ids: list[str | None] = []
            scored_pairs: list[tuple[str, Any]] = []
            scored_error_count = 0
            n_parse_failures = 0
            n_null_targets = 0
            for lineno, line in enumerate(
                scored_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as parse_exc:
                    n_parse_failures += 1
                    _logger.warning(
                        "scored_predictions.jsonl line %d is not valid JSON (%s); "
                        "skipping row.",
                        lineno,
                        parse_exc,
                    )
                    continue
                sid = obj.get("sample_id")
                if isinstance(sid, str) and sid:
                    scored_ids.append(sid)
                tid = obj.get("task_id")
                scored_task_ids.append(tid if isinstance(tid, str) else None)
                has_error = obj.get("error") is not None
                if has_error:
                    scored_error_count += 1
                scores = obj.get("scores")
                if isinstance(scores, dict):
                    scored_scores.append(
                        {
                            k: float(v)
                            for k, v in scores.items()
                            if isinstance(v, (int, float))
                        }
                    )
                else:
                    scored_scores.append({})
                # Collect for dataset-level metrics (skip errored rows).
                targets = obj.get("targets")
                proc_pred = obj.get("processed_prediction") or ""
                if not has_error:
                    if targets is None:
                        n_null_targets += 1
                    else:
                        scored_pairs.append((proc_pred, targets))
            if n_parse_failures:
                _logger.warning(
                    "scored_predictions.jsonl had %d unparseable line(s); "
                    "those rows are excluded from the resume summary.",
                    n_parse_failures,
                )
            if n_null_targets:
                _logger.warning(
                    "scored_predictions.jsonl: %d non-error row(s) have null targets; "
                    "dataset-level metrics will be incomplete.",
                    n_null_targets,
                )
            means = aggregate_score_means(scored_scores)
            means.update(
                compute_dataset_level_metrics(scored_pairs, self._config.task_type)
            )
            per_task = {}
            if scored_task_ids:
                buckets: dict[str | None, list[dict[str, float]]] = {}
                for tid, scores in zip(scored_task_ids, scored_scores, strict=True):
                    buckets.setdefault(tid, []).append(scores)
                per_task = {
                    (tid if tid is not None else "default"): aggregate_score_means(
                        rows
                    )
                    for tid, rows in sorted(
                        buckets.items(),
                        key=lambda kv: (kv[0] is None, kv[0] or ""),
                    )
                }
            prompt_v = self._config.prompt_version or self._renderer_prompt_id()
            metrics: dict[str, Any] = {"mean": means, "per_task_mean": per_task}
            if round_trip_times:
                metrics["latency"] = {
                    "mean_roundtrip_sec": sum(round_trip_times) / len(round_trip_times),
                    "n_batches": len(round_trip_times),
                }
            return RunSummary(
                run_id=self._config.run_id,
                library_version=_package_version(),
                code_git_sha=self._config.code_git_sha,
                run_config_hash=self._config.run_config_hash,
                prompt_version=prompt_v,
                postprocess_version=self._config.postprocess_version,
                scorer_versions=self._config.scorer_versions,
                model_identity=dict(self._client.server_info or {}),
                seed=self._config.seed,
                n_samples=len(scored_ids) if scored_ids else len(fallback_rows),
                n_errors=max(n_errors, scored_error_count),
                metrics=metrics,
                task_results=None,
            )
        except Exception as exc:
            _logger.warning(
                "Failed to build summary from scored_predictions.jsonl (%s); "
                "falling back to in-memory score_rows.",
                exc,
            )
            return self._build_summary(
                fallback_rows,
                score_rows=score_rows,
                processed_pairs=processed_pairs,
                n_errors=n_errors,
                round_trip_times=round_trip_times,
            )

    def _score_sample(
        self,
        example: DatasetExample,
        post: PostProcessResult,
        pred: ModelPrediction,
    ) -> dict[str, float]:
        if self._scorer is not None:
            return dict(self._scorer(example, post, pred))
        if self._metrics_score_sample is None:
            return {}
        raw = self._metrics_score_sample(
            example,
            post=post,
            raw_predictions=list(pred.predictions),
            task_type=self._config.task_type,
        )
        return dict(raw)

    def _run_one_batch(
        self,
        batch: list[DatasetExample],
        *,
        writer: BenchmarkArtifactWriter,
        completed_ids: list[str],
        score_rows: list[dict[str, float]],
        processed_pairs: list[tuple[str, Any]] | None = None,
        error_state: list[int],
        run_cache: TwoLayerRunCache | None = None,
        judge_inputs: list[tuple[DatasetExample, str]] | None = None,
        round_trip_times: list[float] | None = None,
        failures: list[dict[str, Any]] | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        wire_items: list[PredictionsV1RequestItem] = []
        rendered: list[tuple[DatasetExample, ModelRequest]] = []
        for ex in batch:
            try:
                mr = self._renderer.render(ex)
            except Exception as exc:
                pred = ModelPrediction(
                    sample_id=ex.sample_id,
                    predictions=[],
                    finish_reason=None,
                    usage=None,
                    latency_sec=None,
                    error=str(exc),
                    server_info=(
                        dict(self._client.server_info)
                        if self._client.server_info is not None
                        else None
                    ),
                )
                row_err = pred.error
                processed_row = ScoredPrediction(
                    sample_id=ex.sample_id,
                    task_id=ex.task_id,
                    predictions=list(pred.predictions),
                    processed_prediction="",
                    targets=_targets_from_example(ex),
                    scores=None,
                    postprocess_version=self._config.postprocess_version,
                    error=row_err,
                )
                scored_row = processed_row
                writer.append_prediction_record(pred)
                completed_ids.append(ex.sample_id)
                writer.append_processed_prediction(processed_row)
                score_rows.append({})
                writer.append_scored_prediction(scored_row)
                error_state[0] += 1
                if failures is not None:
                    failures.append(
                        {
                            "sample_id": ex.sample_id,
                            "reason": str(exc),
                            "stage": "render",
                        }
                    )
                continue
            wire_items.append(model_request_to_wire_item(mr))
            rendered.append((ex, mr))

        if not wire_items:
            return
        by_id: dict[str, PredictionsV1ResponseItem] = {}
        pending: list[PredictionsV1RequestItem] = []
        if run_cache is None:
            envelope = PredictionsV1Request(requests=wire_items)
            t0 = time.perf_counter()
            response = self._client.generate(envelope)
            elapsed = time.perf_counter() - t0
            if round_trip_times is not None:
                round_trip_times.append(elapsed)
            by_id = {item.sample_id: item for item in response.responses}
            missing = [
                item.sample_id for item in wire_items if item.sample_id not in by_id
            ]
            if missing:
                raise ValueError(
                    f"Launcher response missing {len(missing)} sample_id(s): "
                    f"{missing[:5]!r}. Check launcher for partial responses."
                )
        else:
            for item in wire_items:
                hit = run_cache.get_inference_item(item)
                if hit is not None:
                    by_id[item.sample_id] = hit
                else:
                    pending.append(item)
            if pending:
                envelope = PredictionsV1Request(requests=pending)
                t0 = time.perf_counter()
                response = self._client.generate(envelope)
                elapsed = time.perf_counter() - t0
                if round_trip_times is not None:
                    round_trip_times.append(elapsed)
                pending_by_id = {r.sample_id: r for r in response.responses}
                for item in pending:
                    response_row = pending_by_id.get(item.sample_id)
                    if response_row is None:
                        msg = (
                            "Launcher response missing sample_id="
                            f"{item.sample_id!r} (required for caching + matching)."
                        )
                        raise ValueError(msg)
                    run_cache.put_inference_item(item, response_row)
                    by_id[item.sample_id] = response_row
        work_items: list[tuple[DatasetExample, PredictionsV1ResponseItem]] = []
        for ex, _mr in rendered:
            work_items.append((ex, by_id[ex.sample_id]))

        def _process_one(
            pair: tuple[DatasetExample, PredictionsV1ResponseItem],
        ) -> tuple[
            ModelPrediction,
            ScoredPrediction,
            ScoredPrediction,
            dict[str, float],
            int,
        ]:
            ex, item = pair
            pred = wire_response_item_to_model_prediction(
                item,
                server_info=self._client.server_info,
            )
            raw_text = _raw_prediction_text(pred)
            post_err: str | None = None
            try:
                post = run_post_process_pipeline(
                    raw_text,
                    parser_steps=self._config.parser_steps,
                    cleaner_steps=self._config.cleaner_steps,
                )
            except PostProcessPipelineError as exc:
                post = PostProcessResult(segments=[], text="", warnings=(str(exc),))
                post_err = str(exc)

            row_err = _merge_row_error(pred, post_err)
            targets = _targets_from_example(ex)
            if targets is None and row_err is None:
                _logger.warning(
                    "sample_id=%r has no targets (labels=None); "
                    "per-sample and dataset-level scores will be empty.",
                    ex.sample_id,
                )
            processed_row = ScoredPrediction(
                sample_id=ex.sample_id,
                task_id=ex.task_id,
                predictions=list(pred.predictions),
                processed_prediction=post.text,
                targets=targets,
                scores=None,
                postprocess_version=self._config.postprocess_version,
                error=row_err,
            )
            if row_err is not None:
                scores: dict[str, float] = {}
            elif (
                run_cache is not None
                and self._scorer is None
                and self._metrics_score_sample is not None
            ):
                score_key = scoring_cache_key(
                    predict_url=run_cache.predict_url,
                    sample_id=ex.sample_id,
                    task_id=ex.task_id,
                    labels=ex.labels,
                    raw_predictions=list(pred.predictions),
                    processed_prediction=post.text,
                    postprocess_fingerprint=self._postprocess_fingerprint_value,
                    postprocess_version=self._config.postprocess_version,
                    scorer_versions=self._config.scorer_versions,
                )
                cached_scores = run_cache.get_scores(score_key)
                if cached_scores is not None:
                    scores = cached_scores
                else:
                    scores = self._score_sample(ex, post, pred)
                    run_cache.put_scores(score_key, scores)
            else:
                scores = self._score_sample(ex, post, pred)
            scored_row = processed_row.model_copy(update={"scores": scores or None})
            n_err = 1 if row_err is not None else 0
            return pred, processed_row, scored_row, scores, n_err

        results = map_ordered(
            _process_one,
            work_items,
            workers=self._config.workers,
            executor=executor,
        )
        for (ex, _item), (pred, processed_row, scored_row, scores, n_err) in zip(
            work_items,
            results,
            strict=True,
        ):
            writer.append_prediction_record(pred)
            completed_ids.append(ex.sample_id)
            writer.append_processed_prediction(processed_row)
            score_rows.append(scores)
            writer.append_scored_prediction(scored_row)
            error_state[0] += n_err
            if n_err > 0 and failures is not None:
                failures.append(
                    {
                        "sample_id": ex.sample_id,
                        "reason": str(scored_row.error or pred.error),
                        "stage": "inference",
                    }
                )
            # Collect for dataset-level metrics (non-error rows with targets).
            if (
                processed_pairs is not None
                and n_err == 0
                and scored_row.targets is not None
            ):
                processed_pairs.append(
                    (scored_row.processed_prediction or "", scored_row.targets)
                )
            if judge_inputs is not None and scored_row.error is None:
                judge_inputs.append((ex, scored_row.processed_prediction or ""))

    def _renderer_prompt_id(self) -> str:
        """Return the renderer's bundled ``prompt_id``.

        Returns
        -------
        str
            ``prompt_id`` from the renderer's loaded prompt specification.
        """
        return self._renderer.prompt_id

    def _build_summary(
        self,
        rows: list[DatasetExample],
        *,
        score_rows: list[dict[str, float]],
        processed_pairs: list[tuple[str, Any]] | None = None,
        n_errors: int,
        round_trip_times: list[float] | None = None,
    ) -> RunSummary:
        means = aggregate_score_means(score_rows)
        means.update(
            compute_dataset_level_metrics(
                processed_pairs or [], self._config.task_type
            )
        )
        per_task = per_task_score_means(rows, score_rows) if rows else {}
        prompt_v = self._config.prompt_version or self._renderer_prompt_id()
        metrics: dict[str, Any] = {"mean": means, "per_task_mean": per_task}
        if round_trip_times:
            metrics["latency"] = {
                "mean_roundtrip_sec": sum(round_trip_times) / len(round_trip_times),
                "n_batches": len(round_trip_times),
            }

        return RunSummary(
            run_id=self._config.run_id,
            library_version=_package_version(),
            code_git_sha=self._config.code_git_sha,
            run_config_hash=self._config.run_config_hash,
            prompt_version=prompt_v,
            postprocess_version=self._config.postprocess_version,
            scorer_versions=self._config.scorer_versions,
            model_identity=dict(self._client.server_info or {}),
            seed=self._config.seed,
            n_samples=len(rows),
            n_errors=n_errors,
            metrics=metrics,
            task_results=None,
        )


def _registry_root() -> Path:
    """Return the root directory containing bundled registry YAML assets.

    Returns
    -------
    pathlib.Path
        Absolute path to the bundled ``beans_next/registry`` directory.
    """
    return Path(__file__).resolve().parent.parent / "registry"


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """Load a YAML document whose root is a mapping.

    Parameters
    ----------
    path
        YAML file to read.

    Returns
    -------
    collections.abc.Mapping[str, typing.Any]
        Parsed YAML mapping.

    Raises
    ------
    SystemExit
        If the file cannot be read, the YAML is invalid, or the root is not a
        mapping.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(str(exc)) from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"Invalid YAML ({path}): {exc}") from exc
    if not isinstance(raw, Mapping):
        msg = f"YAML root must be a mapping, got {type(raw).__name__} ({path})"
        raise SystemExit(msg)
    return cast(Mapping[str, Any], raw)


def _suite_yaml_path(suite_id: str) -> Path:
    """Return the on-disk path for a bundled suite YAML id.

    Returns
    -------
    pathlib.Path
        Absolute path to ``beans_next/registry/suite/<suite_id>.yaml``.
    """
    root = _registry_root()
    return (root / "suite" / f"{suite_id}.yaml").resolve()


def _eval_task_yaml_path(eval_task_id: str) -> Path:
    """Return the on-disk path for a bundled eval-task YAML id.

    Returns
    -------
    pathlib.Path
        Absolute path to ``beans_next/registry/eval_task/<eval_task_id>.yaml``.
    """
    root = _registry_root()
    return (root / "eval_task" / f"{eval_task_id}.yaml").resolve()


def _coerce_eval_task_mapping(
    data: Mapping[str, Any], *, source: Path
) -> Mapping[str, Any]:
    """Coerce an eval-task YAML mapping into a task-body mapping.

    Parameters
    ----------
    data
        Parsed YAML mapping.
    source
        Source path used for defaults.

    Returns
    -------
    collections.abc.Mapping[str, typing.Any]
        A mapping representing the eval-task body with an ``eval_task_id`` key
        populated (from the top-level id or the filename stem).
    """
    if len(data) == 1:
        (only_key,) = tuple(data.keys())
        body = data.get(only_key)
        if isinstance(only_key, str) and isinstance(body, Mapping):
            out = dict(body)
            out.setdefault("eval_task_id", only_key)
            return out
    out2 = dict(data)
    out2.setdefault("eval_task_id", source.stem)
    return out2


def _parse_suite_eval_task_ids(
    suite_doc: Mapping[str, Any], *, suite_id: str
) -> list[str]:
    """Extract eval task ids from a suite YAML document.

    Supports either:
    - `{<suite_id>: {eval_tasks: [...]}}`
    - `{suite_id: ..., eval_tasks: [...]}` (direct body)

    Returns
    -------
    list[str]
        Eval-task id stems (no ``.yaml`` suffix).

    Raises
    ------
    SystemExit
        If the suite document does not contain a non-empty task list with valid
        string ids.
    """
    body: Mapping[str, Any] | None = None
    if suite_id in suite_doc and isinstance(suite_doc.get(suite_id), Mapping):
        body = cast(Mapping[str, Any], suite_doc[suite_id])
    else:
        body = suite_doc
    raw = body.get("eval_tasks") or body.get("tasks") or body.get("eval_task_ids")
    if not isinstance(raw, list) or not raw:
        msg = (
            f"Suite {suite_id!r} must define a non-empty list under one of "
            f"`eval_tasks`, `tasks`, or `eval_task_ids`."
        )
        raise SystemExit(msg)
    out: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(f"Suite {suite_id!r} eval_tasks[{i}] must be a string id.")
        stem = item.strip()
        if stem.endswith(".yaml"):
            stem = Path(stem).stem
        out.append(stem)
    return out


def _cache_dir_from_args(args: Namespace) -> Path | None:
    """Resolve optional ``--cache-dir`` from a CLI namespace.

    Parameters
    ----------
    args
        Parsed CLI arguments (may omit ``cache_dir``).

    Returns
    -------
    pathlib.Path or None
        Expanded absolute path, or ``None`` when caching is disabled.
    """
    raw = getattr(args, "cache_dir", None)
    if raw is None:
        return None
    if isinstance(raw, Path):
        return raw.expanduser().resolve()
    return Path(str(raw)).expanduser().resolve()


_DEFAULT_LIMIT: int = sys.maxsize


def _effective_limit(args: Namespace) -> int:
    """Return an effective sample limit for a run namespace.

    Parameters
    ----------
    args
        CLI namespace containing an optional ``limit`` attribute.

    Returns
    -------
    int
        Positive integer sample cap. Defaults to ``_DEFAULT_LIMIT`` when
        ``--limit`` is absent or invalid. When ``--limit`` is absent, this is
        treated as unlimited (``sys.maxsize``).
    """
    raw = getattr(args, "limit", None)
    try:
        if raw is None:
            return _DEFAULT_LIMIT
        return max(1, int(raw))
    except Exception:
        return _DEFAULT_LIMIT


def _default_output_dir(args: Namespace, run_id: str) -> Path:
    """Resolve an output directory from CLI args.

    Returns
    -------
    pathlib.Path
        Resolved artifact directory.
    """
    out = getattr(args, "output_dir", None)
    if out is None:
        return (Path.cwd() / "results" / run_id).resolve()
    return Path(out).expanduser().resolve()


def _prompt_spec_from_eval_task(
    eval_task: Mapping[str, Any], *, args: Namespace
) -> PromptSpec:
    """Select a prompt spec for an eval task, falling back to bundled defaults.

    Parameters
    ----------
    eval_task
        Eval-task configuration mapping.
    args
        CLI namespace; if ``prompt_yaml`` is set, it wins.

    Returns
    -------
    beans_next.prompts.renderer.PromptSpec
        Prompt specification to render examples.
    """
    from beans_next.prompts.renderer import (
        load_builtin_prompt_yaml,
        load_prompt_spec_from_path,
    )

    if getattr(args, "prompt_yaml", None):
        spec = load_prompt_spec_from_path(
            Path(args.prompt_yaml).expanduser().resolve()
        )
        return spec
    prompt_key = (
        eval_task.get("prompt_yaml")
        or eval_task.get("prompt")
        or eval_task.get("prompt_filename")
        or eval_task.get("prompt_file")
    )
    if isinstance(prompt_key, str) and prompt_key.strip():
        name = prompt_key.strip()
        if not name.endswith(".yaml"):
            name = f"{name}.yaml"
        spec = load_builtin_prompt_yaml(name)
    else:
        spec = load_builtin_prompt_yaml("classification_bioacoustic_v1.yaml")

    # If this is a BEANS-Zero eval task with a known subset duration, attach the
    # per-subset clip length so model servers can match the official preprocessing.
    subset = eval_task.get("subset")
    hf_path = eval_task.get("hf_path")
    max_len = (
        _beans_zero_max_length_seconds_for_subset(str(subset).strip())
        if isinstance(subset, str) and subset.strip()
        else None
    )
    if (
        max_len is not None
        and isinstance(hf_path, str)
        and hf_path.strip() == "EarthSpeciesProject/BEANS-Zero"
        and spec.generation_config is not None
        and spec.generation_config.max_length_seconds is None
    ):
        new_gen = spec.generation_config.model_copy(
            update={"max_length_seconds": int(max_len)}
        )
        spec = replace(
            spec,
            generation_config=new_gen,
        )
    return spec


@lru_cache(maxsize=1)
def _load_beans_zero_max_duration_json() -> dict[str, Any]:
    """Load bundled per-subset clip length hints for BEANS-Zero.

    This is a minimal registry artifact so BEANS-Next can forward official
    `max_duration` values to model servers for consistent preprocessing.

    Returns
    -------
    dict[str, Any]
        Mapping from BEANS-Zero subset name to max duration (seconds). Returns
        an empty mapping when the bundled file is missing or cannot be parsed.
    """
    import importlib.resources

    path = Path(importlib.resources.files("beans_next")).joinpath(
        "registry", "beans_zero_max_duration_seconds.json"
    )
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _beans_zero_max_length_seconds_for_subset(subset: str) -> int | None:
    data = _load_beans_zero_max_duration_json()
    val = data.get(subset)
    if val is None:
        return None
    try:
        iv = int(val)
    except Exception:
        return None
    return iv if iv > 0 else None


def _load_examples_for_eval_task(
    eval_task: Mapping[str, Any], *, args: Namespace
) -> list[DatasetExample]:
    """Load `DatasetExample` rows for one eval task, respecting `--limit`.

    Returns
    -------
    list[beans_next.api.types.DatasetExample]
        Up to ``limit`` normalized dataset rows.

    Raises
    ------
    SystemExit
        If required config keys are missing or invalid.
    """
    from beans_next.datasets import dataset_name_equals
    from beans_next.datasets.esp_data import (
        iter_esp_data_beans_next_examples,
        iter_esp_data_beans_next_multiaudio_examples,
        iter_esp_data_beans_zero_examples,
        iter_esp_data_birdset_examples,
    )
    from beans_next.datasets.hf_multiaudio import (
        beans_next_multiaudio_row_filter,
        iter_hf_streaming_multiaudio_examples,
    )
    from beans_next.datasets.hf_streaming import iter_hf_streaming_examples

    # Configurable dataset backend switch:
    # - explicit CLI `--backend` wins
    # - then YAML `data_source` (if present in run-config or eval-task body)
    # - then env var `BEANS_NEXT_DATA_SOURCE` (compat: `BEANS_PRO_DATA_SOURCE`)
    # - else default esp_data
    data_source = getattr(args, "data_source", None) or eval_task.get("data_source")
    if not isinstance(data_source, str) or not data_source.strip():
        data_source = os.environ.get(
            "BEANS_NEXT_DATA_SOURCE",
            os.environ.get("BEANS_PRO_DATA_SOURCE", "esp_data"),
        )
    data_source = str(data_source).strip()

    hf_path = cast(
        str,
        eval_task.get("hf_path")
        or eval_task.get("dataset_hf_path")
        or getattr(args, "hf_path", None),
    )
    split = cast(str, eval_task.get("split") or getattr(args, "split", "test"))
    hf_config = (
        eval_task.get("hf_config")
        or eval_task.get("config_name")
        or getattr(args, "hf_config", None)
    )
    if hf_config == "":
        hf_config = None
    if hf_config is not None and not isinstance(hf_config, str):
        raise SystemExit("Eval task `hf_config` must be a string or null.")
    subset = eval_task.get("subset")
    dataset_name = (
        eval_task.get("dataset_name")
        or eval_task.get("dataset")
        or subset
        or getattr(args, "dataset_name", None)
    )
    row_filter = (
        dataset_name_equals(dataset_name)
        if isinstance(dataset_name, str) and dataset_name.strip()
        else None
    )
    task_id = cast(
        str | None,
        eval_task.get("eval_task_id") or eval_task.get("task_id") or None,
    )
    limit = _effective_limit(args)
    rows: list[DatasetExample] = []

    if data_source == "esp_data":
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise SystemExit(
                "esp_data loading requires a non-empty `subset`/`dataset_name`."
            )
        if dataset_name.strip() == "beans_next":
            subset_name = eval_task.get("subset") or split
            if not isinstance(subset_name, str) or not subset_name.strip():
                raise SystemExit(
                    "BeansPro esp_data loading requires a non-empty `subset`."
                )
            for ex in iter_esp_data_beans_next_examples(
                subset=subset_name.strip(),
                split=str(split),
                task_id=task_id,
                limit=limit,
            ):
                rows.append(ex)
                if len(rows) >= limit:
                    break
            return rows
        if dataset_name.strip() == "beans_next_multiaudio":
            subset_name = eval_task.get("subset") or split
            if not isinstance(subset_name, str) or not subset_name.strip():
                raise SystemExit(
                    "BeansProMultiAudio esp_data loading requires a non-empty `subset`."
                )
            for ex in iter_esp_data_beans_next_multiaudio_examples(
                split=subset_name.strip(),
                task_id=task_id,
                limit=limit,
            ):
                rows.append(ex)
                if len(rows) >= limit:
                    break
            return rows
        if dataset_name.strip() == "birdset":
            subset_name = eval_task.get("subset") or split
            if not isinstance(subset_name, str) or not subset_name.strip():
                raise SystemExit(
                    "BirdSet esp_data loading requires a non-empty `subset`."
                )
            for ex in iter_esp_data_birdset_examples(
                subset=subset_name.strip(),
                split=str(split),
                task_id=task_id,
                limit=limit,
            ):
                rows.append(ex)
                if len(rows) >= limit:
                    break
            return rows
        for ex in iter_esp_data_beans_zero_examples(
            subset=dataset_name.strip(),
            split=str(split),
            task_id=task_id,
            limit=limit,
        ):
            rows.append(ex)
            if len(rows) >= limit:
                break
        return rows

    if data_source == "huggingface":
        if isinstance(dataset_name, str) and dataset_name.strip() == "birdset":
            from beans_next.datasets.hf_birdset import iter_hf_birdset_examples

            subset_name = eval_task.get("subset") or split
            if not isinstance(subset_name, str) or not subset_name.strip():
                raise SystemExit(
                    "BirdSet huggingface loading requires a non-empty `subset` in the "
                    "eval task (e.g. subset: HSN-test_5s)."
                )
            for ex in iter_hf_birdset_examples(
                subset=subset_name.strip(),
                split=str(split),
                task_id=task_id,
                limit=limit,
            ):
                rows.append(ex)
                if len(rows) >= limit:
                    break
            return rows

        _BEANS_ZERO_REPO = "EarthSpeciesProject/BEANS-Zero"
        _dataset_key = (
            str(dataset_name).strip() if isinstance(dataset_name, str) else ""
        )
        _beans_next_hf_datasets = frozenset({"beans_next", "beans_next_multiaudio"})
        if (
            isinstance(hf_path, str)
            and hf_path.strip() == _BEANS_ZERO_REPO
            and _dataset_key not in _beans_next_hf_datasets
        ):
            from beans_next.datasets.hf import iter_hf_dataset_examples

            if not isinstance(dataset_name, str) or not dataset_name.strip():
                raise SystemExit(
                    "BEANS-Zero huggingface loading requires a non-empty `subset` or "
                    "`dataset_name` in the eval task."
                )
            revision = str(eval_task.get("revision") or "main")
            for ex in iter_hf_dataset_examples(
                hf_path.strip(),
                split=str(split),
                config_name=cast(str | None, hf_config),
                revision=revision,
                task_id=task_id,
                row_filter=row_filter,
            ):
                rows.append(ex)
                if len(rows) >= limit:
                    break
            return rows

        from beans_next.datasets.beans_next_hub import (
            BEANS_NEXT_HUB_REPO_ID,
            iter_hf_beans_next_examples,
        )

        # Use eval task hf_path if set; else canonical BEANS-Next repo (not CLI hf-path
        # default EarthSpeciesProject/BEANS-Zero).
        repo_id = (eval_task.get("hf_path") or "").strip() or BEANS_NEXT_HUB_REPO_ID
        subset_name = eval_task.get("subset") or split
        if not isinstance(subset_name, str) or not subset_name.strip():
            raise SystemExit(
                "huggingface backend requires a non-empty `subset` in the eval task."
            )
        revision = str(eval_task.get("revision") or "main")
        # BEANS-Next on Hugging Face is a single-table Parquet dataset. We treat the
        # benchmark split as "test" by default (older configs sometimes used
        # subset-named splits, and HF defaults can be "train" depending on the card).
        hf_split = str(eval_task.get("hf_split") or "test")
        for ex in iter_hf_beans_next_examples(
            repo_id,
            subset=subset_name.strip(),
            split=hf_split,
            revision=revision,
            task_id=task_id,
            limit=limit,
        ):
            rows.append(ex)
            if len(rows) >= limit:
                break
        return rows

    if not isinstance(hf_path, str) or not hf_path.strip():
        raise SystemExit("Eval task must define `hf_path` (or provide `--hf-path`).")

    if (
        isinstance(dataset_name, str)
        and dataset_name.strip() == "beans_next_multiaudio"
    ):
        tier_cfg = eval_task.get("tier")
        if not isinstance(tier_cfg, str) or not tier_cfg.strip():
            tier_cfg = "tier_4_in_context"
        else:
            tier_cfg = tier_cfg.strip()
        subset_cfg = subset if isinstance(subset, str) and subset.strip() else None
        row_filter_ma = beans_next_multiaudio_row_filter(
            tier=tier_cfg,
            subset=subset_cfg,
        )
        for ex in iter_hf_streaming_multiaudio_examples(
            hf_path,
            split=str(split),
            config_name=cast(str | None, hf_config),
            task_id=task_id,
            row_filter=row_filter_ma,
        ):
            rows.append(ex)
            if len(rows) >= limit:
                break
        return rows

    for ex in iter_hf_streaming_examples(
        hf_path,
        split=str(split),
        config_name=cast(str | None, hf_config),
        task_id=task_id,
        row_filter=row_filter,
    ):
        rows.append(ex)
        if len(rows) >= limit:
            break
    return rows


@lru_cache(maxsize=1)
def _load_beans_zero_labels_json() -> dict[str, Any]:
    """Load ``beans_zero_labels.json`` from the bundled registry, cached.

    Returns
    -------
    dict[str, Any]
        Parsed JSON mapping, or empty dict on any error.
    """
    import importlib.resources

    try:
        registry_path = Path(
            importlib.resources.files("beans_next")  # type: ignore[arg-type]
        ).joinpath("registry", "beans_zero_labels.json")
        if registry_path.is_file():
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:  # noqa: BLE001
        pass
    return {}


def _labels_for_eval_task(task_cfg: Mapping[str, Any]) -> list[str] | None:
    """Return the fixed label vocabulary for an eval task, if available.

    Checks (in order):
    1. ``task_cfg["labels"]`` — inline list in the eval-task YAML.
    2. ``beans_zero_labels.json`` registry keyed by ``task_cfg["subset"]``.

    Returns ``None`` when no fixed vocab is found.

    Parameters
    ----------
    task_cfg : Mapping[str, Any]
        Eval-task config mapping (from YAML or plan).

    Returns
    -------
    list[str] or None
        Ordered label list, or ``None``.
    """
    # BirdSet: prefer scientific vocab generated from dataset metadata.
    dataset = task_cfg.get("dataset")
    if isinstance(dataset, str) and dataset.strip() == "birdset":
        sci = task_cfg.get("scientific_labels")
        if isinstance(sci, list) and sci:
            return [str(x) for x in sci]

    inline = task_cfg.get("labels")
    if isinstance(inline, list) and inline:
        return [str(x) for x in inline]

    subset = task_cfg.get("subset")
    if not isinstance(subset, str) or not subset.strip():
        return None

    data = _load_beans_zero_labels_json()
    labels = data.get(subset.strip())
    if isinstance(labels, list) and labels:
        return [str(x) for x in labels]
    return None


def _postprocess_steps_for_examples(
    examples: list[DatasetExample],
    task_type: str | None = None,
    labels_override: list[str] | None = None,
) -> tuple[tuple[StepSpec, ...], tuple[StepSpec, ...]]:
    """Mirror the CLI's iteration-1 postprocess defaults for suite runs.

    Parameters
    ----------
    examples : list[DatasetExample]
        Dataset examples used to build the label vocabulary when no
        ``labels_override`` is provided.
    task_type : str or None
        Task type string from the eval-task config. Fuzzy label matching is
        skipped for ``"captioning"`` tasks.
    labels_override : list[str] or None
        When provided, used as the vocabulary for fuzzy matching instead of
        the dynamic vocab derived from ``examples``. Typically loaded from the
        per-dataset label registry (``beans_zero_labels.json``).

    Returns
    -------
    tuple
        ``(parser_steps, cleaner_steps)`` suitable for :class:`RunnerConfig`.
    """
    cleaners: list[StepSpec] = [
        StepSpec("normalize_whitespace", {}),
        StepSpec("strip_eos", {}),
    ]

    # Open-ended tasks preserve free text; label parsing would corrupt them.
    if task_type in {"captioning", "qa", "open_ended", "counting"}:
        # BirdSet open-set scientific naming: we still want a light-touch,
        # BirdSet-specific canonicalization step when a scientific vocab is
        # available, but we must avoid generic label parsing that rewrites
        # free text into unrelated labels.
        if labels_override and all(
            isinstance(v, str) and " " in v.strip() for v in labels_override
        ):
            cleaners.append(
                StepSpec(
                    "extract_scientific_name_from_text",
                    {"labels": tuple(labels_override), "min_score": 1.0},
                )
            )
            # Closed-set fallback: any segment that the scientific-name
            # canonicalizer left untouched gets snapped to the nearest label
            # via the three-stage exact -> substring -> Levenshtein extractor.
            # Threshold disabled so every prediction maps to *some* label —
            # BirdSet eval is genuinely closed-set, so "couldn't parse" is
            # never the right outcome; an unmapped output is always wrong.
            cleaners.append(
                StepSpec(
                    "extract_label_from_text",
                    {
                        "labels": tuple(labels_override),
                        "apply_threshold": False,
                    },
                )
            )
        return (), tuple(cleaners)

    # Build vocabulary from examples when no override is provided.
    vocab: list[str] = []
    if not labels_override:
        seen: set[str] = set()
        for ex in examples:
            lbs = ex.labels
            if isinstance(lbs, str) and lbs.strip():
                for part in lbs.split(","):
                    tok = part.strip()
                    if tok and tok not in seen:
                        seen.add(tok)
                        vocab.append(tok)
            elif isinstance(lbs, list):
                for item in lbs:
                    if isinstance(item, str) and item.strip() and item not in seen:
                        seen.add(item)
                        vocab.append(item)
    effective_vocab = tuple(labels_override or vocab)

    # Binary tasks ("Yes"/"No") must be treated as single-label extraction.
    # Comma splitting turns verbose answers like "Yes, ..." into fragments that
    # then fuzzy-match back to "Yes" repeatedly ("Yes, Yes, ...").
    vocab_lower = {v.lower() for v in effective_vocab if isinstance(v, str)}
    if vocab_lower == {"yes", "no"} and effective_vocab:
        cleaners.append(
            StepSpec("extract_label_from_text", {"labels": effective_vocab})
        )
        return (), tuple(cleaners)

    # MCQ tasks ("A/B/C/D" choices): extract the final chosen letter from
    # verbose enumerations rather than splitting/fuzzy-matching prose.
    if effective_vocab and all(
        isinstance(v, str) and len(v.strip()) == 1 for v in effective_vocab
    ):
        mcq = {v.strip().lower() for v in effective_vocab}
        if 2 <= len(mcq) <= 10 and all(t.isalpha() for t in mcq):
            cleaners.append(
                StepSpec(
                    "extract_mcq_choice_from_text",
                    {"labels": effective_vocab},
                )
            )
            return (), tuple(cleaners)

    # Hz bucket tasks (e.g. "4010 Hz"): map numeric text to closest bucket and
    # avoid comma-splitting prose (commas appear in sentences and in "1,654.75").
    if effective_vocab and all(
        isinstance(v, str) and v.strip().lower().endswith("hz") for v in effective_vocab
    ):
        if all(str(v).strip().split()[0].isdigit() for v in effective_vocab):
            cleaners.append(
                StepSpec(
                    "extract_hz_bucket_from_text",
                    {"labels": effective_vocab},
                )
            )
            return (), tuple(cleaners)

    # Classification: single-label output.
    if task_type == "classification":
        if effective_vocab:
            # BirdSet-style open-set scientific names: match label parts against
            # raw text instead of forcing free text through generic label
            # extraction + Levenshtein.
            if all(isinstance(v, str) and " " in v.strip() for v in effective_vocab):
                cleaners.append(
                    StepSpec(
                        "extract_scientific_name_from_text",
                        {"labels": effective_vocab, "min_score": 1.0},
                    )
                )
                # Closed-set fallback: any segment the strict canonicalizer
                # left untouched gets snapped to the nearest label via the
                # three-stage exact -> substring -> Levenshtein extractor.
                # Classification is closed-set, so an unmapped output is
                # always wrong, never "unparseable" — force every segment
                # to land on a label.
                cleaners.append(
                    StepSpec(
                        "extract_label_from_text",
                        {"labels": effective_vocab, "apply_threshold": False},
                    )
                )
            else:
                cleaners.append(
                    StepSpec("extract_label_from_text", {"labels": effective_vocab})
                )
        return (), tuple(cleaners)

    # Detection and unknown task types: comma-split, then fuzzy match.
    parsers = (StepSpec("parse_labels_comma", {}),)
    if effective_vocab:
        cleaners.append(
            StepSpec(
                "conservative_match_to_labels",
                {
                    "labels": effective_vocab,
                    "max_distance": 2,
                    "allow_fuzzy": True,
                    "drop_unmatched": True,
                },
            )
        )
    return parsers, tuple(cleaners)


def _judge_from_args_and_task(
    args: Namespace,
    eval_task: Mapping[str, Any],
) -> JudgeScorer | None:
    """Build a :class:`~beans_next.judges.scorer.JudgeScorer` from CLI args + eval-task.

    Returns ``None`` when ``--judge-url`` is absent or empty.

    Parameters
    ----------
    args
        CLI namespace; reads ``judge_url`` attribute.
    eval_task
        Eval-task config mapping; optional ``judge.template_id`` key overrides
        the default template.

    Returns
    -------
    JudgeScorer or None
        Configured scorer, or ``None`` when judge is not requested.
    """
    judge_url = getattr(args, "judge_url", None)
    if not isinstance(judge_url, str) or not judge_url.strip():
        return None
    template_id = "bioacoustic_open_qa_v1"
    judge_block = eval_task.get("judge") if isinstance(eval_task, Mapping) else None
    if isinstance(judge_block, Mapping):
        tid = judge_block.get("template_id")
        if isinstance(tid, str) and tid.strip():
            template_id = tid.strip()
    return JudgeScorer(judge_url.strip(), template_id=template_id)


def run_from_cli_namespace(args: Namespace) -> None:
    """CLI-dispatched hook for `beans-next run`.

    This hook owns execution when the CLI imports `beans_next.runner.runner` and finds
    a callable named `run_from_cli_namespace` (preferred), `main_run_from_cli`, or
    `cli_run`.

    The behavior is:
    - When `args.suite` is set: resolve `beans_next/registry/suite/<suite>.yaml`,
      expand it into a list of eval-task YAML ids, and run each task as its own
      `BenchmarkRunner` invocation under a deterministic output subdirectory.
    - Otherwise: execute the same minimal HuggingFace-slice run path as the CLI's
      built-in fallback (prompt default + HTTP endpoint + `--limit` cap).

    Raises
    ------
    SystemExit
        On invalid argument combinations, missing registry content, or missing
        optional dependencies such as `datasets` or `jinja2`.
    """
    config_path = getattr(args, "config", None)
    run_id = (
        (getattr(args, "run_id", None) or "beans-next-cli").strip()
        or "beans-next-cli"
    )
    raw_workers = getattr(args, "workers", 1)
    try:
        workers = max(1, int(raw_workers))
    except Exception:
        workers = 1

    cache_dir = _cache_dir_from_args(args)

    _upload_gcs = bool(getattr(args, "upload_gcs", False))
    _gcs_base = (getattr(args, "gcs_prefix", None) or "").rstrip("/")
    if _upload_gcs and not _gcs_base:
        raise SystemExit("--upload-gcs requires --gcs-prefix <gs://bucket/path>")

    suite_id = getattr(args, "suite", None)
    resume_requested = bool(getattr(args, "resume", False))
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        if not isinstance(resume_from, Path):
            raise SystemExit("--resume-from must be a path.")
        base_out = resume_from.expanduser().resolve()
        # Optional run_id recovery from a prior run (suite runs may omit a root
        # checkpoint; per-task checkpoints still drive resume under suite/<id>/).
        checkpoint_path = base_out / "checkpoint.json"
        if checkpoint_path.is_file():
            try:
                payload = read_checkpoint_json(checkpoint_path)
            except ValueError:
                payload = {}
            raw_run_id = payload.get("run_id")
            if isinstance(raw_run_id, str) and raw_run_id.strip():
                run_id = raw_run_id.strip()
        resume_requested = True
    else:
        base_out = _default_output_dir(args, run_id)

    if config_path is not None:
        from beans_next.config import RunConfigError, load_run_config

        try:
            loaded = load_run_config(config_path)
        except RunConfigError as exc:
            raise SystemExit(str(exc)) from exc

        # If the CLI provided a predict URL override, apply it to all endpoints in the
        # loaded run-config plan. This is required for Slurm workflows where the
        # bundled run-configs use localhost placeholders but inference is against a
        # remote launcher discovered via the URL-file protocol.
        predict_url_override = getattr(args, "predict_url", None)
        if predict_url_override:
            from urllib.parse import urljoin

            from beans_next.config.run_config import ExecutionItem

            override = str(predict_url_override).rstrip("/")
            override_predict = (
                override if override.endswith("/predict") else f"{override}/predict"
            )
            override_info = urljoin(override_predict, "/info")
            override_health = urljoin(override_predict, "/health")

            model_by_name: dict[str, Any] = {}
            for m in loaded.models:
                model_by_name[m.name] = m.model_copy(
                    update={
                        "predict_url": override_predict,
                        "info_url": override_info,
                        "health_url": override_health,
                    }
                )

            loaded_plan: list[ExecutionItem] = [
                ExecutionItem(
                    model=model_by_name[item.model.name],
                    eval_task_id=item.eval_task_id,
                    eval_task=item.eval_task,
                )
                for item in loaded.plan
            ]
        else:
            loaded_plan = loaded.plan

        cfg_run_id = (
            loaded.config.run_id
            or getattr(args, "run_id", None)
            or "beans-next-config"
        )
        cfg_run_id = str(cfg_run_id).strip() or "beans-next-config"
        run_id = cfg_run_id
        if loaded.config.output_dir is not None and loaded.config.output_dir.strip():
            base_out = Path(loaded.config.output_dir).expanduser().resolve()
        else:
            base_out = _default_output_dir(args, run_id)

        effective_limit = (
            getattr(args, "limit", None)
            if getattr(args, "limit", None) is not None
            else loaded.config.limit
        )
        args_for_tasks = Namespace(**vars(args))
        args_for_tasks.limit = effective_limit
        if getattr(args_for_tasks, "data_source", None) is None:
            args_for_tasks.data_source = loaded.config.data_source

        base_out.mkdir(parents=True, exist_ok=True)

        try:
            from beans_next.prompts.renderer import PromptRenderer
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc

        # Group plan items by predict_url so we share one probed client per URL.
        plan_groups: dict[str, list[Any]] = {}
        plan_order: list[str] = []
        for item in loaded_plan:
            url = str(item.model.predict_url)
            if url not in plan_groups:
                plan_groups[url] = []
                plan_order.append(url)
            plan_groups[url].append(item)

        items_out: list[dict[str, Any]] = []
        for url in plan_order:
            group = plan_groups[url]
            first_model = group[0].model
            client_kwargs: dict[str, Any] = {"probe_on_init": True}
            http_timeout_raw = os.environ.get("BEANS_PRO_HTTP_TIMEOUT_SEC", "").strip()
            if http_timeout_raw:
                try:
                    client_kwargs["timeout"] = float(http_timeout_raw)
                except ValueError:
                    raise SystemExit(
                        "Invalid BEANS_PRO_HTTP_TIMEOUT_SEC (must be float seconds): "
                        f"{http_timeout_raw!r}"
                    ) from None
            if first_model.retry_policy is not None:
                client_kwargs["retry_policy"] = dict(first_model.retry_policy)
            with HttpClient(url, **client_kwargs) as client:
                for item in group:
                    model = item.model
                    eval_task_id = item.eval_task_id
                    task_cfg = dict(item.eval_task)
                    task_cfg.setdefault("eval_task_id", eval_task_id)

                    try:
                        examples = _load_examples_for_eval_task(
                            task_cfg, args=args_for_tasks
                        )
                    except ImportError as exc:
                        raise SystemExit(str(exc)) from exc
                    if not examples:
                        raise SystemExit(
                            f"Eval task {eval_task_id!r} loaded zero examples; "
                            "check its HF config/split."
                        )

                    parsers, cleaners = _postprocess_steps_for_examples(
                        examples,
                        task_type=task_cfg.get("task_type"),
                        labels_override=_labels_for_eval_task(task_cfg),
                    )
                    spec = _prompt_spec_from_eval_task(task_cfg, args=args_for_tasks)
                    renderer = PromptRenderer(spec)
                    task_run_id = f"{run_id}__{model.name}__{eval_task_id}"
                    out_dir = (base_out / model.name / eval_task_id).resolve()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cfg = RunnerConfig(
                        output_dir=out_dir,
                        run_id=task_run_id,
                        parser_steps=parsers,
                        cleaner_steps=cleaners,
                        resume=resume_requested,
                        workers=workers,
                        cache_dir=cache_dir,
                        task_type=task_cfg.get("task_type") or None,
                        gcs_upload_prefix=(
                            f"{_gcs_base}/{task_run_id}" if _upload_gcs else None
                        ),
                    )

                    judge = _judge_from_args_and_task(args_for_tasks, task_cfg)
                    runner = BenchmarkRunner(client, renderer, cfg, judge=judge)
                    summary = runner.run(examples)

                    items_out.append(
                        {
                            "model": model.model_dump(),
                            "eval_task_id": eval_task_id,
                            "output_dir": str(out_dir),
                            "summary": summary.model_dump(),
                        }
                    )

        from beans_next.results.store import dumps_canonical

        (base_out / "run_summary.json").write_text(
            dumps_canonical(
                {
                    "schema_version": "beans_next.run_summary_aggregate.v1",
                    "source_config": str(loaded.source_path),
                    "run_id": run_id,
                    "output_dir": str(base_out),
                    "limit": effective_limit if effective_limit is not None else None,
                    "items": items_out,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return

    predict_url = getattr(args, "predict_url", None)
    if not predict_url:
        raise SystemExit(
            "--predict-url is required for beans-next run (unless --config is set)."
        )

    # Resolve suite -> eval tasks (or fall back to a single HF slice run).
    eval_task_ids: list[str] | None = None
    if isinstance(suite_id, str) and suite_id.strip():
        suite_id = suite_id.strip()
        suite_path = _suite_yaml_path(suite_id)
        if not suite_path.is_file():
            msg = (
                f"Suite {suite_id!r} not found at {suite_path}. "
                "Ensure registry content (I4-A) is present."
            )
            raise SystemExit(msg)
        suite_doc = _load_yaml_mapping(suite_path)
        eval_task_ids = _parse_suite_eval_task_ids(suite_doc, suite_id=suite_id)

    try:
        from beans_next.prompts.renderer import PromptRenderer
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    client_kwargs2: dict[str, Any] = {"probe_on_init": True}
    http_timeout_raw2 = os.environ.get("BEANS_PRO_HTTP_TIMEOUT_SEC", "").strip()
    if http_timeout_raw2:
        try:
            client_kwargs2["timeout"] = float(http_timeout_raw2)
        except ValueError:
            raise SystemExit(
                "Invalid BEANS_PRO_HTTP_TIMEOUT_SEC (must be float seconds): "
                f"{http_timeout_raw2!r}"
            ) from None
    with HttpClient(str(predict_url), **client_kwargs2) as client:
        if eval_task_ids is None:
            # Single-task fallback: load the full eval-task YAML when --task-id is
            # given so prompt / task_type / subset fields are respected.  Callers
            # that don't pass --task-id keep the previous behaviour (empty mapping,
            # falls back to HF args + default prompt).
            raw_task_id = getattr(args, "task_id", None)
            if isinstance(raw_task_id, str) and raw_task_id.strip():
                task_yaml_path = _eval_task_yaml_path(raw_task_id.strip())
                if task_yaml_path.is_file():
                    doc = _load_yaml_mapping(task_yaml_path)
                    single_task_cfg = _coerce_eval_task_mapping(
                        doc, source=task_yaml_path
                    )
                else:
                    single_task_cfg = {"eval_task_id": raw_task_id}
            else:
                single_task_cfg = {}
            try:
                examples = _load_examples_for_eval_task(single_task_cfg, args=args)
            except ImportError as exc:
                raise SystemExit(str(exc)) from exc
            if not examples:
                raise SystemExit(
                    "No dataset examples were loaded; check HF parameters."
                )
            parsers, cleaners = _postprocess_steps_for_examples(
                examples,
                task_type=single_task_cfg.get("task_type")
                or getattr(args, "task_type", None),
            )
            spec = _prompt_spec_from_eval_task(single_task_cfg, args=args)

            renderer = PromptRenderer(spec)
            cfg = RunnerConfig(
                output_dir=base_out,
                run_id=run_id,
                parser_steps=parsers,
                cleaner_steps=cleaners,
                resume=resume_requested,
                workers=workers,
                cache_dir=cache_dir,
                task_type=single_task_cfg.get("task_type")
                or getattr(args, "task_type", None)
                or None,
                gcs_upload_prefix=f"{_gcs_base}/{run_id}" if _upload_gcs else None,
            )
            judge = _judge_from_args_and_task(args, single_task_cfg)
            runner = BenchmarkRunner(client, renderer, cfg, judge=judge)
            runner.run(examples)
            return

        # Suite path: run each eval task in its own subdirectory.
        suite_id_str = cast(str, suite_id)
        suite_out = (base_out / "suite" / suite_id_str).resolve()
        suite_out.mkdir(parents=True, exist_ok=True)
        task_summaries: list[dict[str, Any]] = []
        for eval_task_id in eval_task_ids:
            eval_task_path = _eval_task_yaml_path(eval_task_id)
            if not eval_task_path.is_file():
                raise SystemExit(
                    f"Eval task {eval_task_id!r} not found at {eval_task_path}."
                )
            doc = _load_yaml_mapping(eval_task_path)
            task_cfg = _coerce_eval_task_mapping(doc, source=eval_task_path)

            # Load examples (limit applies per task).
            try:
                examples = _load_examples_for_eval_task(task_cfg, args=args)
            except ImportError as exc:
                raise SystemExit(str(exc)) from exc
            if not examples:
                raise SystemExit(
                    f"Eval task {eval_task_id!r} loaded zero examples; "
                    "check its HF config/split."
                )

            parsers, cleaners = _postprocess_steps_for_examples(
                examples,
                task_type=task_cfg.get("task_type"),
                labels_override=_labels_for_eval_task(task_cfg),
            )
            spec = _prompt_spec_from_eval_task(task_cfg, args=args)
            renderer = PromptRenderer(spec)
            task_run_id = f"{run_id}__{eval_task_id}"
            out_dir = (suite_out / eval_task_id).resolve()
            cfg = RunnerConfig(
                output_dir=out_dir,
                run_id=task_run_id,
                parser_steps=parsers,
                cleaner_steps=cleaners,
                resume=resume_requested,
                workers=workers,
                cache_dir=cache_dir,
                task_type=task_cfg.get("task_type") or None,
                gcs_upload_prefix=(
                    f"{_gcs_base}/{task_run_id}" if _upload_gcs else None
                ),
            )
            judge = _judge_from_args_and_task(args, task_cfg)
            runner = BenchmarkRunner(client, renderer, cfg, judge=judge)
            summary = runner.run(examples)
            task_summaries.append(
                {
                    "eval_task_id": eval_task_id,
                    "output_dir": str(out_dir),
                    "summary": summary.model_dump(),
                }
            )

        # Write suite-level aggregate summary.
        from beans_next.results.store import dumps_canonical

        (suite_out / "suite_summary.json").write_text(
            dumps_canonical(
                {
                    "schema_version": "beans_next.suite_summary.v1",
                    "suite_id": suite_id_str,
                    "run_id": run_id,
                    "output_dir": str(suite_out),
                    "eval_tasks": task_summaries,
                }
            )
            + "\n",
            encoding="utf-8",
        )
