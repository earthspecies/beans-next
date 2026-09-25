"""Rescore existing prediction artifacts on CPU.

This module supports the Phase-3 utility CLI:

`beans-next score-from-file <predictions.jsonl>`

"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from beans_next.api.types import (
    DatasetExample,
    ModelPrediction,
    RunSummary,
    ScoredPrediction,
)
from beans_next.post_process.answers import SCORING_VERSION
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

__all__ = ["rescore_predictions_file"]

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


def _canonical_mcq_vocab(labels: Iterable[str]) -> tuple[str, ...] | None:
    """Return canonical MCQ letters when all labels are option tokens.

    Parameters
    ----------
    labels : iterable of str
        Candidate label vocabulary.

    Returns
    -------
    tuple of str or None
        Canonical labels, preserving the first-seen target casing, when the
        vocabulary is MCQ-like; otherwise ``None``.
    """
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        match = re.fullmatch(r"\s*[\(\[]?\s*([A-Za-z])\s*[\)\]\.:]?\s*", label)
        if match is None:
            return None
        token = match.group(1)
        key = token.lower()
        if key not in seen:
            seen.add(key)
            out.append(token)
    if 2 <= len(out) <= 10:
        return tuple(out)
    return None


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
    task_s = (task_type or "").lower()

    # Open-ended tasks preserve free text; label parsing would corrupt it and,
    # for captioning specifically, would silently turn every reference caption
    # into a comma-split "label vocabulary" and fuzzy-match (Levenshtein) each
    # prediction against all of it — O(n^2) over the whole corpus and never
    # semantically meaningful, since CIDEr (not label matching) scores these.
    # Mirrors the live runner's `_postprocess_steps_for_examples`.
    from beans_next.post_process.answers import FREE_TEXT_TASKS

    if task_s in FREE_TEXT_TASKS:
        return (), tuple(cleaners)

    vocab = _collect_label_vocab(targets)

    # Regression metrics parse numbers from the processed text, so avoid
    # snapping F0/SNR outputs onto a closed label vocabulary before scoring.
    if "regression" in task_s:
        return (), tuple(cleaners)

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
    mcq_vocab = _canonical_mcq_vocab(vocab)
    if mcq_vocab is not None:
        cleaners.append(
            StepSpec(
                "extract_mcq_choice_from_text",
                {"labels": mcq_vocab},
            )
        )
        return (), tuple(cleaners)

    # Hz bucket tasks (e.g. "4010 Hz"): map numeric text to closest bucket and
    # avoid comma-splitting prose (commas appear in normal sentences and in
    # thousands separators like "1,654.75").
    if vocab and all(
        isinstance(v, str) and v.strip().lower().endswith("hz") for v in vocab
    ):
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
) -> RunSummary:
    """Rescore an existing ``predictions.jsonl`` artifact on CPU.

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
    question_by_id: dict[str, str] = {}
    if processed_path.is_file():
        for obj in _read_jsonl(processed_path):
            try:
                row = ScoredPrediction.model_validate(obj)
            except Exception:
                continue
            targets_by_id[row.sample_id] = row.targets
            task_id_by_id[row.sample_id] = row.task_id
            if row.question:
                question_by_id[row.sample_id] = row.question

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
            metadata = (
                {"instruction": question_by_id[sid]} if sid in question_by_id else {}
            )
            from beans_next.post_process.answers import (
                FREE_TEXT_TASKS,
                normalize_task_answer,
                question_options,
            )

            if task_type in FREE_TEXT_TASKS or question_options(metadata):
                text = normalize_task_answer(raw_text, task_type, metadata)
                post = PostProcessResult(segments=[text] if text else [], text=text)

            processed_row = ScoredPrediction(
                sample_id=sid,
                task_id=task_id,
                predictions=list(pred.predictions),
                processed_prediction=post.text,
                targets=targets,
                question=question_by_id.get(sid),
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
                    metadata=metadata,
                )
                scores = _score_sample_if_available(
                    example,
                    post=post,
                    raw_predictions=list(pred.predictions),
                    task_type=task_type,
                )
                dataset_pairs.append((post.text, targets))

            scored_row = processed_row.model_copy(
                update={"scores": dict(scores) if scores else None}
            )
            scored_f.write(dumps_canonical(scored_row.model_dump(mode="json")) + "\n")
            score_rows.append(scores)

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
        scorer_versions={"deterministic": SCORING_VERSION},
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

    return summary
