"""Parser-phase post-process steps (structural extraction only)."""

from __future__ import annotations

from dataclasses import replace

from beans_next.post_process.pipeline import (
    PostProcessContext,
    PostProcessPipelineError,
)

__all__ = ["apply_parse_labels_comma"]


def apply_parse_labels_comma(
    ctx: PostProcessContext,
    *,
    separator: str = ",",
    keep_empty: bool = False,
) -> PostProcessContext:
    """Split each segment on a comma-like separator into separate segments.

    Mirrors BEANS-Zero detection parsing: comma-separated labels become
    individual segments for downstream per-label cleaners.

    Parameters
    ----------
    ctx : PostProcessContext
        Current pipeline context.
    separator : str, optional
        Separator used to split segments. The default is a single comma.
    keep_empty : bool, optional
        When ``False`` (default), empty fragments after stripping are dropped.

    Returns
    -------
    PostProcessContext
        New context whose ``segments`` list is the flattened split of all
        input segments.

    Raises
    ------
    PostProcessPipelineError
        If ``separator`` is empty.
    """
    if not separator:
        raise PostProcessPipelineError(
            "separator must be a non-empty string",
            step_name="parse_labels_comma",
        )

    def _split_commas_not_in_numbers(text: str) -> list[str]:
        # Avoid splitting thousands separators like "1,654.75" (digit-comma-digit).
        parts: list[str] = []
        buf: list[str] = []
        for i, ch in enumerate(text):
            if ch == separator:
                prev = text[i - 1] if i > 0 else ""
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if prev.isdigit() and nxt.isdigit():
                    buf.append(ch)
                    continue
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        parts.append("".join(buf))
        return parts

    out: list[str] = []
    for piece in ctx.segments:
        # Fast path: only customize splitting for comma separators.
        raw_parts = (
            _split_commas_not_in_numbers(piece)
            if separator == ","
            else piece.split(separator)
        )
        for part in raw_parts:
            stripped = part.strip()
            if stripped:
                out.append(stripped)
            elif keep_empty:
                out.append(stripped)

    warnings: tuple[str, ...] = ctx.warnings
    if not out and ctx.segments:
        msg = (
            "parse_labels_comma produced no segments from non-empty input; "
            f"raw segment count was {len(ctx.segments)}"
        )
        warnings = warnings + (msg,)

    return replace(ctx, segments=out, warnings=warnings)
