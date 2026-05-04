"""Post-processing pipeline engine (parser phase, then cleaner phase).

Runs all configured parser steps on the working ``segments`` list, then all
cleaner steps. Parser steps should perform structural extraction only; cleaner
steps perform text normalization and label alignment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "PostProcessContext",
    "PostProcessPipelineError",
    "PostProcessResult",
    "StepSpec",
    "builtin_cleaner_steps",
    "builtin_parser_steps",
    "run_post_process_pipeline",
]


class PostProcessPipelineError(ValueError):
    """Raised when a post-process pipeline cannot be executed as configured.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    step_name : str | None
        Name of the step that failed, when known.
    """

    def __init__(self, message: str, *, step_name: str | None = None) -> None:
        detail = f" (step={step_name!r})" if step_name else ""
        super().__init__(f"{message}{detail}")
        self.step_name = step_name


@dataclass(frozen=True)
class PostProcessContext:
    """Mutable-through-``replace`` working state for one prediction string.

    Attributes
    ----------
    segments : list[str]
        Label fragments or free-text segments. The pipeline starts with a
        single segment containing the raw model text.
    warnings : tuple[str, ...]
        Non-fatal diagnostics accumulated while running steps.
    """

    segments: list[str]
    warnings: tuple[str, ...] = ()

    def with_warnings(self, *messages: str) -> PostProcessContext:
        """Return a copy with additional warning messages appended.

        Parameters
        ----------
        *messages : str
            Warning strings to append.

        Returns
        -------
        PostProcessContext
            New context including the extra warnings.
        """
        if not messages:
            return self
        return replace(self, warnings=self.warnings + messages)


@dataclass(frozen=True)
class PostProcessResult:
    """Final output of a post-process pipeline run.

    Attributes
    ----------
    segments : list[str]
        Processed segments (for example matched labels).
    text : str
        Segments joined with `", "` (comma + space), matching BEANS-Zero style
        multi-label strings.
    warnings : tuple[str, ...]
        Non-fatal diagnostics emitted by steps.
    """

    segments: list[str]
    text: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepSpec:
    """One configured pipeline step.

    Attributes
    ----------
    name : str
        Builtin step name (for example ``\"parse_labels_comma\"``).
    params : Mapping[str, Any]
        Keyword arguments forwarded to the step implementation.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


def builtin_parser_steps() -> dict[str, Callable[..., PostProcessContext]]:
    """Return the mapping of builtin parser step names to callables.

    Returns
    -------
    dict[str, Callable[..., PostProcessContext]]
        Name to parser function. Imported lazily to avoid import cycles.
    """
    from beans_next.post_process import parsers as parsers_mod

    return {
        "parse_labels_comma": parsers_mod.apply_parse_labels_comma,
    }


def builtin_cleaner_steps() -> dict[str, Callable[..., PostProcessContext]]:
    """Return the mapping of builtin cleaner step names to callables.

    Returns
    -------
    dict[str, Callable[..., PostProcessContext]]
        Name to cleaner function.
    """
    from beans_next.post_process import cleaners as cleaners_mod

    return {
        "normalize_whitespace": cleaners_mod.apply_normalize_whitespace,
        "strip_eos": cleaners_mod.apply_strip_eos,
        "fuzzy_match_to_labels": cleaners_mod.apply_fuzzy_match_to_labels,
        "conservative_match_to_labels": cleaners_mod.apply_conservative_match_to_labels,
        "extract_label_from_text": cleaners_mod.apply_extract_label_from_text,
        "extract_scientific_name_from_text": (
            cleaners_mod.apply_extract_scientific_name_from_text
        ),
        "extract_mcq_choice_from_text": cleaners_mod.apply_extract_mcq_choice_from_text,
        "extract_hz_bucket_from_text": cleaners_mod.apply_extract_hz_bucket_from_text,
    }


def run_post_process_pipeline(
    raw_text: str,
    *,
    parser_steps: Sequence[StepSpec] = (),
    cleaner_steps: Sequence[StepSpec] = (),
) -> PostProcessResult:
    """Execute parser steps, then cleaner steps, on one raw prediction string.

    The initial context is ``segments=[raw_text]``. Parser steps typically
    expand or split segments; cleaners normalize each segment.

    Parameters
    ----------
    raw_text : str
        Raw model output for one sample (possibly multi-label).
    parser_steps : Sequence[StepSpec]
        Parser phase steps, run in order.
    cleaner_steps : Sequence[StepSpec]
        Cleaner phase steps, run after all parsers, in order.

    Returns
    -------
    PostProcessResult
        Processed segments, joined text, and any warnings.

    Raises
    ------
    PostProcessPipelineError
        If a step name is unknown or a step raises an unexpected error.
    TypeError
        If ``raw_text`` is not a string.

    Notes
    -----
    Steps should only mutate context by returning a new :class:`PostProcessContext`
    via :func:`dataclasses.replace` for determinism and traceability.
    """
    if not isinstance(raw_text, str):
        msg = f"raw_text must be str, got {type(raw_text).__name__}"
        raise TypeError(msg)

    parsers = builtin_parser_steps()
    cleaners = builtin_cleaner_steps()

    ctx = PostProcessContext(segments=[raw_text])

    for spec in parser_steps:
        fn = parsers.get(spec.name)
        if fn is None:
            raise PostProcessPipelineError(
                f"Unknown parser step {spec.name!r}",
                step_name=spec.name,
            )
        try:
            ctx = fn(ctx, **dict(spec.params))
        except PostProcessPipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced with context
            raise PostProcessPipelineError(
                f"Parser step failed: {exc}",
                step_name=spec.name,
            ) from exc

    for spec in cleaner_steps:
        fn = cleaners.get(spec.name)
        if fn is None:
            raise PostProcessPipelineError(
                f"Unknown cleaner step {spec.name!r}",
                step_name=spec.name,
            )
        try:
            ctx = fn(ctx, **dict(spec.params))
        except PostProcessPipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PostProcessPipelineError(
                f"Cleaner step failed: {exc}",
                step_name=spec.name,
            ) from exc

    joined = ", ".join(ctx.segments)
    return PostProcessResult(
        segments=list(ctx.segments),
        text=joined,
        warnings=ctx.warnings,
    )
