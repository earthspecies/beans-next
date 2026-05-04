"""Rescore existing prediction artifacts on CPU.

This module supports the Phase-3 utility CLI:

`beans-next score-from-file <predictions.jsonl>`

When ``--judge-url`` is provided, a second pass calls a judge model served via
the ``predictions_v1`` HTTP predict API and writes separate artifacts prefixed
with ``judge_`` so normal rescorer outputs are never overwritten.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from beans_next.api.types import (
    DatasetExample,
    ModelPrediction,
    RunSummary,
    ScoredPrediction,
)
from beans_next.post_process.pipeline import (
    PostProcessPipelineError,
    PostProcessResult,
    StepSpec,
    run_post_process_pipeline,
)
from beans_next.results.store import dumps_canonical
from beans_next.runner._utils import (
    aggregate_score_means,
    compute_dataset_level_metrics,
)

__all__ = ["rescore_predictions_file", "judge_extract_from_predictions_file"]

_logger = logging.getLogger(__name__)


def _package_version() -> str:
    try:
        return importlib.metadata.version("beans-next")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _read_jsonl(path: Path) -> list[object]:
    out: list[object] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def _collect_label_vocab(targets: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in targets:
        if isinstance(t, str) and t.strip():
            for part in t.split(","):
                tok = part.strip()
                if tok and tok not in seen:
                    seen.add(tok)
                    out.append(tok)
        elif isinstance(t, list):
            for item in t:
                if isinstance(item, str) and item.strip() and item not in seen:
                    seen.add(item)
                    out.append(item)
    return out


def _default_postprocess_steps(
    *,
    targets: Iterable[object],
    task_type: str | None = None,
) -> tuple[tuple[StepSpec, ...], tuple[StepSpec, ...]]:
    cleaners: list[StepSpec] = [
        StepSpec("normalize_whitespace", {}),
        StepSpec("strip_eos", {}),
    ]
    vocab = _collect_label_vocab(targets)
    task_s = (task_type or "").lower()

    # Binary tasks ("Yes"/"No") must be treated as single-label extraction, even
    # when task_type is unknown/None (common in score-from-file usage). Comma
    # splitting turns verbose answers like "Yes, ..." into multiple fragments
    # which then fuzzy-match back to "Yes" and produce "Yes, Yes, ...".
    vocab_lower = {v.lower() for v in vocab if isinstance(v, str)}
    if vocab_lower == {"yes", "no"} and vocab:
        cleaners.append(StepSpec("extract_label_from_text", {"labels": tuple(vocab)}))
        return (), tuple(cleaners)

    # MCQ tasks ("A/B/C/D" choices) must also be treated as single-label extraction
    # even when task_type is unknown/None. Many models enumerate all options
    # ("A: ..., B: ...") and then state a final letter; comma-splitting prose
    # fragments causes repeated letter matches.
    if vocab and all(isinstance(v, str) and len(v.strip()) == 1 for v in vocab):
        mcq = {v.strip().lower() for v in vocab}
        if 2 <= len(mcq) <= 10 and all(t.isalpha() for t in mcq):
            cleaners.append(
                StepSpec(
                    "extract_mcq_choice_from_text",
                    {"labels": tuple(vocab)},
                )
            )
            return (), tuple(cleaners)

    # Hz bucket tasks (e.g. "4010 Hz"): map numeric text to closest bucket and
    # avoid comma-splitting prose (commas appear in normal sentences and in
    # thousands separators like "1,654.75").
    if vocab and all(isinstance(v, str) and v.strip().lower().endswith("hz") for v in vocab):
        # Be strict: require a leading integer to avoid accidentally catching
        # unrelated "Hz" mentions in other tasks.
        if all(v.strip().split()[0].isdigit() for v in vocab):
            cleaners.append(
                StepSpec("extract_hz_bucket_from_text", {"labels": tuple(vocab)})
            )
            return (), tuple(cleaners)

    # Classification: no comma split; use the three-stage extraction cleaner.
    if "classification" in task_s and "detection" not in task_s:
        if vocab:
            cleaners.append(
                StepSpec("extract_label_from_text", {"labels": tuple(vocab)})
            )
        return (), tuple(cleaners)

    # Detection and unknown task types: comma split + fuzzy match.
    parsers = (StepSpec("parse_labels_comma", {}),)
    if vocab:
        cleaners.append(StepSpec("fuzzy_match_to_labels", {"labels": tuple(vocab)}))
    return parsers, tuple(cleaners)


def _score_sample_if_available(
    example: DatasetExample,
    *,
    post: PostProcessResult,
    raw_predictions: list[str],
    task_type: str | None = None,
) -> Mapping[str, float]:
    try:
        from beans_next.metrics import score_sample
    except ImportError:
        return {}
    return score_sample(
        example, post=post, raw_predictions=raw_predictions, task_type=task_type
    )


def rescore_predictions_file(
    predictions_jsonl: Path,
    *,
    output_dir: Path | None = None,
    task_type: str | None = None,
    judge_url: str | None = None,
    judge_extract_url: str | None = None,
) -> RunSummary:
    """Rescore an existing ``predictions.jsonl`` artifact on CPU.

    Two optional judge passes can run after normal scoring.  Both write to
    separate files prefixed with ``judge_`` or ``judge_extracted_`` and never
    overwrite normal rescorer artifacts.  They may be combined in one call.

    Parameters
    ----------
    predictions_jsonl
        Path to a ``predictions.jsonl`` file containing one `predictions_v1`
        `ModelPrediction`-shaped JSON object per line.
    output_dir
        Directory to write artifacts into. Defaults to the parent directory of
        ``predictions_jsonl``.
    task_type : str or None, optional
        Task type string (e.g. ``"classification"``, ``"detection"``).  When
        provided, selects the correct post-processing pipeline and routes
        ``score_sample`` appropriately.  When ``None``, detection-style
        post-processing is used (backward-compatible default).
    judge_url : str or None, optional
        Full URL for ``POST /predict`` on the YES/NO judge model endpoint.
        When set, :class:`~beans_next.judges.predict_v1_judge.PredictV1Judge`
        is called and these artifacts are written:

        * ``judge_scored_predictions.jsonl`` — rows with
          ``scores = {"judge_accuracy": 0.0 or 1.0}``.
        * ``judge_summary.json`` — summary with ``metrics.mean.judge_accuracy``.
        * ``judge_outputs.jsonl`` — raw judge responses.

    judge_extract_url : str or None, optional
        Full URL for ``POST /predict`` on the extractor judge model endpoint.
        When set, :class:`~beans_next.judges.predict_v1_extractor.PredictV1Extractor`
        converts each raw prediction into a structured prediction using
        task-specific templates, then scores the result with the normal pipeline.
        These artifacts are written:

        * ``judge_extracted_scored_predictions.jsonl`` — rows with
          ``processed_prediction`` replaced by the judge-extracted text and
          ``scores`` computed via the normal metrics pipeline.
        * ``judge_extracted_summary.json`` — summary with standard metrics
          (e.g. ``accuracy``, ``f1``) computed on extracted predictions.

    Returns
    -------
    RunSummary
        Summary written to ``summary.json`` in ``output_dir``.

    Raises
    ------
    FileNotFoundError
        If ``predictions_jsonl`` does not exist.
    ValueError
        If the predictions file is empty, or if no targets are available (typically
        because a sibling ``processed_predictions.jsonl`` is absent).
    """
    predictions_jsonl = Path(predictions_jsonl).expanduser().resolve()
    if not predictions_jsonl.is_file():
        raise FileNotFoundError(f"Not found: {predictions_jsonl}")

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else predictions_jsonl.parent.resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_objs = _read_jsonl(predictions_jsonl)
    if not pred_objs:
        raise ValueError(f"No JSONL rows found in {predictions_jsonl}")
    preds: list[ModelPrediction] = [
        ModelPrediction.model_validate(obj) for obj in pred_objs
    ]

    # Targets are required for metric computation; prefer sibling processed rows.
    processed_path = predictions_jsonl.parent / "processed_predictions.jsonl"
    targets_by_id: dict[str, object] = {}
    task_id_by_id: dict[str, str | None] = {}
    if processed_path.is_file():
        for obj in _read_jsonl(processed_path):
            try:
                row = ScoredPrediction.model_validate(obj)
            except Exception:
                continue
            targets_by_id[row.sample_id] = row.targets
            task_id_by_id[row.sample_id] = row.task_id

    if not targets_by_id:
        raise ValueError(
            "Cannot score metrics without targets. Provide a sibling "
            "`processed_predictions.jsonl` containing `targets` for each sample "
            "(typically produced by `beans-next run`)."
        )

    parsers, cleaners = _default_postprocess_steps(
        targets=targets_by_id.values(), task_type=task_type
    )

    processed_out_path = out_dir / "processed_predictions.jsonl"
    scored_out_path = out_dir / "scored_predictions.jsonl"
    score_rows: list[Mapping[str, float]] = []
    dataset_pairs: list[tuple[str, Any]] = []
    n_errors = 0

    # Collect all scored rows and judge-eligible (example, raw_text) pairs.
    all_scored_rows: list[ScoredPrediction] = []
    judge_eligible: list[tuple[DatasetExample, str]] = []

    with contextlib.ExitStack() as stack:
        processed_f = stack.enter_context(
            processed_out_path.open("w", encoding="utf-8")
        )
        scored_f = stack.enter_context(scored_out_path.open("w", encoding="utf-8"))

        for pred in preds:
            raw_text = pred.predictions[0] if pred.predictions else ""
            post_err: str | None = None
            try:
                post = run_post_process_pipeline(
                    raw_text, parser_steps=parsers, cleaner_steps=cleaners
                )
            except PostProcessPipelineError as exc:
                post = PostProcessResult(segments=[], text="", warnings=(str(exc),))
                post_err = str(exc)

            row_err = pred.error or post_err
            sid = pred.sample_id
            targets = targets_by_id.get(sid)
            task_id = task_id_by_id.get(sid)

            processed_row = ScoredPrediction(
                sample_id=sid,
                task_id=task_id,
                predictions=list(pred.predictions),
                processed_prediction=post.text,
                targets=targets,
                scores=None,
                postprocess_version=None,
                error=row_err,
            )
            processed_f.write(
                dumps_canonical(processed_row.model_dump(mode="json")) + "\n"
            )

            if row_err is not None:
                scores: Mapping[str, float] = {}
                n_errors += 1
            elif targets is None:
                _logger.warning(
                    "sample_id=%r has no targets; scores will be empty.", sid
                )
                scores = {}
            else:
                example = DatasetExample(
                    sample_id=sid,
                    task_id=task_id,
                    labels=targets,
                    metadata={},
                )
                scores = _score_sample_if_available(
                    example,
                    post=post,
                    raw_predictions=list(pred.predictions),
                    task_type=task_type,
                )
                dataset_pairs.append((post.text, targets))
                # Collect for judge pass (raw text, not post-processed).
                if judge_url is not None:
                    judge_eligible.append((example, raw_text))

            scored_row = processed_row.model_copy(
                update={"scores": dict(scores) if scores else None}
            )
            scored_f.write(dumps_canonical(scored_row.model_dump(mode="json")) + "\n")
            score_rows.append(scores)
            all_scored_rows.append(scored_row)

    model_identity: dict[str, Any] = {}
    for pred in preds:
        if pred.server_info:
            model_identity = dict(pred.server_info)
            break

    mean_scores = dict(aggregate_score_means(score_rows))
    mean_scores.update(compute_dataset_level_metrics(dataset_pairs, task_type))
    summary = RunSummary(
        run_id="score-from-file",
        library_version=_package_version(),
        code_git_sha=None,
        run_config_hash=None,
        prompt_version=None,
        postprocess_version=None,
        scorer_versions=None,
        model_identity=model_identity,
        seed=None,
        n_samples=len(preds),
        n_errors=n_errors,
        metrics={"mean": mean_scores},
        task_results=None,
    )
    (out_dir / "summary.json").write_text(
        dumps_canonical(summary.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    (out_dir / "model_identity.json").write_text(
        dumps_canonical(model_identity) + "\n", encoding="utf-8"
    )

    if judge_url is not None and judge_eligible:
        _run_judge_pass(
            judge_url=judge_url,
            judge_eligible=judge_eligible,
            all_scored_rows=all_scored_rows,
            out_dir=out_dir,
            model_identity=model_identity,
            n_errors=n_errors,
            library_version=_package_version(),
        )

    if judge_extract_url is not None and judge_eligible:
        _run_judge_extraction_pass(
            judge_extract_url=judge_extract_url,
            judge_eligible=judge_eligible,
            all_scored_rows=all_scored_rows,
            task_type=task_type,
            out_dir=out_dir,
            model_identity=model_identity,
            n_errors=n_errors,
            library_version=_package_version(),
        )

    return summary


def _run_judge_pass(
    *,
    judge_url: str,
    judge_eligible: list[tuple[DatasetExample, str]],
    all_scored_rows: list[ScoredPrediction],
    out_dir: Path,
    model_identity: dict[str, Any],
    n_errors: int,
    library_version: str,
) -> None:
    """Call the judge model and write ``judge_*`` artifacts.

    Writes to ``out_dir``:

    * ``judge_outputs.jsonl`` — raw judge response items.
    * ``judge_scored_predictions.jsonl`` — all scored rows with judge scores.
    * ``judge_summary.json`` — summary with ``judge_accuracy`` metric.

    Parameters
    ----------
    judge_url
        Full URL for ``POST /predict`` on the judge model endpoint.
    judge_eligible
        ``(DatasetExample, raw_text)`` pairs for non-error samples with targets.
    all_scored_rows
        All :class:`~beans_next.api.types.ScoredPrediction` rows in original
        prediction order (including errored rows).
    out_dir
        Output directory (must already exist).
    model_identity
        Server identity dict carried from the primary scoring pass.
    n_errors
        Error count from the primary scoring pass.
    library_version
        Library version string for :class:`~beans_next.api.types.RunSummary`.
    """
    from beans_next.judges.predict_v1_judge import PredictV1Judge

    examples = [ex for ex, _ in judge_eligible]
    raw_texts = [txt for _, txt in judge_eligible]

    _logger.info(
        "Running judge pass: %d samples → %s", len(examples), judge_url
    )
    judge = PredictV1Judge(judge_url)
    judge_results = judge.score_batch(examples, raw_texts)

    judge_scores_by_id: dict[str, float | None] = {
        r.sample_id: r.score for r in judge_results
    }

    # judge_outputs.jsonl — raw judge responses.
    judge_outputs_path = out_dir / "judge_outputs.jsonl"
    with judge_outputs_path.open("w", encoding="utf-8") as f:
        for r in judge_results:
            f.write(dumps_canonical(r.model_dump(mode="json")) + "\n")

    # judge_scored_predictions.jsonl — all rows with judge_accuracy scores.
    judge_scored_path = out_dir / "judge_scored_predictions.jsonl"
    judge_score_rows: list[Mapping[str, float]] = []
    with judge_scored_path.open("w", encoding="utf-8") as f:
        for scored_row in all_scored_rows:
            sid = scored_row.sample_id
            if scored_row.error is not None:
                judge_row_scores: dict[str, float] | None = None
                judge_score_rows.append({})
            else:
                judge_score = judge_scores_by_id.get(sid)
                if judge_score is not None:
                    judge_row_scores = {"judge_accuracy": judge_score}
                    judge_score_rows.append({"judge_accuracy": judge_score})
                else:
                    judge_row_scores = None
                    judge_score_rows.append({})
            judge_row = scored_row.model_copy(update={"scores": judge_row_scores})
            f.write(dumps_canonical(judge_row.model_dump(mode="json")) + "\n")

    # judge_summary.json — aggregate judge metrics.
    judge_summary = RunSummary(
        run_id="judge-score-from-file",
        library_version=library_version,
        code_git_sha=None,
        run_config_hash=None,
        prompt_version=None,
        postprocess_version=None,
        scorer_versions=None,
        model_identity=model_identity,
        seed=None,
        n_samples=len(all_scored_rows),
        n_errors=n_errors,
        metrics={"mean": aggregate_score_means(judge_score_rows)},
        task_results=None,
    )
    (out_dir / "judge_summary.json").write_text(
        dumps_canonical(judge_summary.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )


def _run_judge_extraction_pass(
    *,
    judge_extract_url: str,
    judge_eligible: list[tuple[DatasetExample, str]],
    all_scored_rows: list[ScoredPrediction],
    task_type: str | None,
    out_dir: Path,
    model_identity: dict[str, Any],
    n_errors: int,
    library_version: str,
) -> None:
    """Run the extractor judge and write ``judge_extracted_*`` artifacts.

    The extractor judge converts each raw model output into a structured
    prediction using task-specific templates, then scores the result with the
    normal post-process and metrics pipeline.

    Writes to ``out_dir``:

    * ``judge_extracted_scored_predictions.jsonl`` — all rows with
      ``processed_prediction`` replaced by judge-extracted text and ``scores``
      computed via the standard metrics pipeline.
    * ``judge_extracted_summary.json`` — summary with standard metric means.

    Parameters
    ----------
    judge_extract_url
        Full URL for ``POST /predict`` on the extractor judge endpoint.
    judge_eligible
        ``(DatasetExample, raw_text)`` pairs for non-error samples with targets.
    all_scored_rows
        All :class:`~beans_next.api.types.ScoredPrediction` rows in original
        prediction order (including errored rows).
    task_type
        Task type string used to select the extraction template and scoring path.
    out_dir
        Output directory (must already exist).
    model_identity
        Server identity dict carried from the primary scoring pass.
    n_errors
        Error count from the primary scoring pass.
    library_version
        Library version string for :class:`~beans_next.api.types.RunSummary`.
    """
    from beans_next.judges.predict_v1_extractor import PredictV1Extractor

    examples = [ex for ex, _ in judge_eligible]
    raw_texts = [txt for _, txt in judge_eligible]

    _logger.info(
        "Running extraction judge pass: %d samples → %s",
        len(examples),
        judge_extract_url,
    )
    extractor = PredictV1Extractor(judge_extract_url, task_type=task_type)
    extracted_texts = extractor.extract_batch(examples, raw_texts)
    extracted_by_id = {
        ex.sample_id: txt for ex, txt in zip(examples, extracted_texts, strict=True)
    }

    # Minimal post-process steps for extracted predictions: whitespace + EOS
    # strip only; no label matching (the judge already extracted the right labels).
    min_cleaners: tuple[StepSpec, ...] = (
        StepSpec("normalize_whitespace", {}),
        StepSpec("strip_eos", {}),
    )
    task_s = (task_type or "").lower()
    min_parsers: tuple[StepSpec, ...] = (
        (StepSpec("parse_labels_comma", {}),) if "detection" in task_s else ()
    )

    extraction_scored_path = out_dir / "judge_extracted_scored_predictions.jsonl"
    extraction_score_rows: list[Mapping[str, float]] = []

    with extraction_scored_path.open("w", encoding="utf-8") as f:
        for scored_row in all_scored_rows:
            sid = scored_row.sample_id
            extracted_text = extracted_by_id.get(sid)

            if scored_row.error is not None or not extracted_text:
                extraction_row = scored_row.model_copy(
                    update={"scores": None}
                )
                extraction_score_rows.append({})
            else:
                try:
                    post = run_post_process_pipeline(
                        extracted_text,
                        parser_steps=min_parsers,
                        cleaner_steps=min_cleaners,
                    )
                except PostProcessPipelineError as exc:
                    _logger.warning(
                        "Post-process failed on extracted text for "
                        "sample_id=%r: %s",
                        sid,
                        exc,
                    )
                    extraction_row = scored_row.model_copy(
                        update={"processed_prediction": extracted_text, "scores": None}
                    )
                    extraction_score_rows.append({})
                    f.write(
                        dumps_canonical(extraction_row.model_dump(mode="json")) + "\n"
                    )
                    continue

                example = DatasetExample(
                    sample_id=scored_row.sample_id,
                    task_id=scored_row.task_id,
                    labels=scored_row.targets,
                    metadata={},
                )
                scores = _score_sample_if_available(
                    example,
                    post=post,
                    raw_predictions=[extracted_text],
                    task_type=task_type,
                )
                extraction_row = scored_row.model_copy(
                    update={
                        "processed_prediction": post.text,
                        "scores": dict(scores) if scores else None,
                    }
                )
                extraction_score_rows.append(scores)

            f.write(dumps_canonical(extraction_row.model_dump(mode="json")) + "\n")

    extraction_summary = RunSummary(
        run_id="judge-extract-from-file",
        library_version=library_version,
        code_git_sha=None,
        run_config_hash=None,
        prompt_version=None,
        postprocess_version=None,
        scorer_versions=None,
        model_identity=model_identity,
        seed=None,
        n_samples=len(all_scored_rows),
        n_errors=n_errors,
        metrics={"mean": aggregate_score_means(extraction_score_rows)},
        task_results=None,
    )
    (out_dir / "judge_extracted_summary.json").write_text(
        dumps_canonical(extraction_summary.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )


def judge_extract_from_predictions_file(
    predictions_jsonl: Path,
    judge_extract_url: str,
    *,
    output_dir: Path | None = None,
    task_type: str | None = None,
) -> RunSummary:
    """Run judge extraction on raw predictions and score the results.

    Convenience wrapper for the judge-extraction-only workflow.  Loads raw
    model outputs from ``predictions_jsonl``, uses a judge model to convert
    each output into a structured prediction, then scores the extracted
    predictions with the normal beans-next metrics pipeline.

    Normal rescoring is **not** performed; only the judge-extracted artifacts
    are written.  To combine normal rescoring with extraction, call
    :func:`rescore_predictions_file` with both ``judge_extract_url`` and
    optionally ``judge_url`` set.

    Parameters
    ----------
    predictions_jsonl
        Path to a ``predictions.jsonl`` file produced by ``beans-next run``.
    judge_extract_url
        Full URL for ``POST /predict`` on the extractor judge model endpoint
        (e.g. ``http://localhost:8010/predict``).
    output_dir
        Directory to write artifacts into.  Defaults to the parent directory
        of ``predictions_jsonl``.
    task_type : str or None, optional
        Task type string (e.g. ``"classification"``, ``"detection"``,
        ``"captioning"``).  Selects the extraction template and scoring path.

    Returns
    -------
    RunSummary
        Summary written to ``judge_extracted_summary.json`` in ``output_dir``.
    """
    return rescore_predictions_file(
        predictions_jsonl,
        output_dir=output_dir,
        task_type=task_type,
        judge_extract_url=judge_extract_url,
    )
