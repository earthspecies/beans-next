"""Cleaner-phase post-process steps (normalization and label alignment)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

from beans_next.post_process.pipeline import (
    PostProcessContext,
    PostProcessPipelineError,
)

__all__ = [
    "apply_conservative_match_to_labels",
    "apply_extract_mcq_choice_from_text",
    "apply_extract_hz_bucket_from_text",
    "apply_extract_label_from_text",
    "apply_extract_scientific_name_from_text",
    "apply_fuzzy_match_to_labels",
    "apply_normalize_whitespace",
    "apply_strip_eos",
]

_WS_RE = re.compile(r"\s+")
_MCQ_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Strongest: explicit "answer"/"final answer"/"correct option" style.
    re.compile(
        r"(?im)(?:^|\b)(?:final\s+answer|answer|correct\s+(?:option|choice|description)|"
        r"selected\s+option|i\s+choose|i\s+pick)\s*(?:is|:|-)?\s*(?:option\s*)?"
        r"[*_`\(\[]*(?P<label>[a-z])\b"
    ),
    # Common: "the correct description is **B**"
    re.compile(
        r"(?im)(?:^|\b)(?:the\s+)?(?:correct|best)\s+(?:match|option|choice|description)\s*"
        r"(?:is|:|-)\s*[*_`\(\[]*(?P<label>[a-z])\b"
    ),
    # Fallback: last standalone letter token (often appears as "**B**" on its own line).
    re.compile(r"(?im)(?:^|\b)[*_`\(\[]*(?P<label>[a-z])\b[*_`\)\]]*\s*$"),
)

# Secondary patterns for MCQ outputs that refer to "option X" without explicit
# "Answer:"/final marker. We only accept uppercase labels to avoid matching the
# English article "a".
_MCQ_OPTION_REF_RE = re.compile(
    r"(?im)\b(?:option|choice|description)\s*(?P<label>[A-Z])\b"
)

_HZ_LABEL_RE = re.compile(r"(?im)^\s*(?P<hz>\d+)\s*hz\s*$")
_NUMBER_RE = re.compile(
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)
_HZ_RANGE_RE = re.compile(
    r"(?im)(?P<a>\d{3,6})\s*-\s*(?P<b>\d{3,6})\s*hz\b"
)

_BIRDSET_TIME_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*#?\s*\d+(?:\.\d+)?\s*s?\s*-\s*\d+(?:\.\d+)?\s*s?\s*#?\s*:\s*"
)
_MARKDOWN_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[*_`]")
_NON_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9\s\-\.]")


def _normalize_for_scientific_match(text: str) -> str:
    text2 = _BIRDSET_TIME_PREFIX_RE.sub("", text)
    text2 = _MARKDOWN_PUNCT_RE.sub(" ", text2)
    text2 = text2.lower()
    text2 = _NON_WORD_RE.sub(" ", text2)
    text2 = _WS_RE.sub(" ", text2).strip()
    return text2


def _scientific_parts(label: str) -> list[str]:
    parts = [p.strip().lower() for p in label.split() if p.strip()]
    return [p for p in parts if p]


def _score_scientific_label(norm_text: str, parts: list[str]) -> float:
    if not parts:
        return 0.0
    tokens = norm_text.split()
    tok_set = set(tokens)
    matched = sum(1 for p in parts if p in tok_set)
    frac = matched / len(parts)
    # Bonus for contiguous phrase match (e.g. "turdus philomelos" appears).
    phrase = " ".join(parts)
    bonus = 0.25 if phrase and phrase in norm_text else 0.0
    return frac + bonus


def apply_extract_scientific_name_from_text(
    ctx: PostProcessContext,
    *,
    labels: Sequence[str],
    min_score: float = 1.0,
) -> PostProcessContext:
    """Extract the best-matching scientific name from free-form model output.

    This is intended for BirdSet-style open-set outputs where models may emit
    timestamps, prose, or mixed casing. It matches scientific labels by
    checking whether each lowercased token-part of the label appears in the
    normalized answer, preferring contiguous matches.

    Parameters
    ----------
    ctx
        Current pipeline context.
    labels
        Candidate scientific labels (e.g. ``"Turdus philomelos"``).
    min_score
        Minimum score required to replace a segment. A score of ``1.0`` means
        all label parts were found (or a contiguous phrase match adds a bonus).

    Returns
    -------
    PostProcessContext
        Context where each segment is replaced by the best label when a match
        is confident; otherwise segments are preserved.

    Raises
    ------
    PostProcessPipelineError
        If ``labels`` is empty or ``min_score`` is not positive.
    """
    if not labels:
        raise PostProcessPipelineError(
            "labels must be a non-empty sequence",
            step_name="extract_scientific_name_from_text",
        )
    if min_score <= 0:
        raise PostProcessPipelineError(
            f"min_score must be positive, got {min_score}",
            step_name="extract_scientific_name_from_text",
        )

    label_parts: list[tuple[str, list[str]]] = [
        (lab, _scientific_parts(str(lab))) for lab in labels if isinstance(lab, str)
    ]
    mapped: list[str] = []
    for seg in ctx.segments:
        norm = _normalize_for_scientific_match(str(seg))
        best_label = str(seg)
        best_score = 0.0
        best_len = 0
        for lab, parts in label_parts:
            score = _score_scientific_label(norm, parts)
            if score > best_score or (score == best_score and len(parts) > best_len):
                best_label = lab
                best_score = score
                best_len = len(parts)
        mapped.append(best_label if best_score >= min_score else str(seg).strip())
    return replace(ctx, segments=mapped)


def apply_normalize_whitespace(ctx: PostProcessContext) -> PostProcessContext:
    """Collapse internal runs of whitespace and strip each segment.

    Parameters
    ----------
    ctx : PostProcessContext
        Current pipeline context.

    Returns
    -------
    PostProcessContext
        Context with each segment normalized independently.
    """
    normalized = [_WS_RE.sub(" ", s).strip() for s in ctx.segments]
    return replace(ctx, segments=normalized)


def apply_strip_eos(
    ctx: PostProcessContext,
    *,
    eos_token: str = "<|end_of_text|>",
) -> PostProcessContext:
    """Remove text from the first end-of-sequence marker onward (per segment).

    Follows BEANS-Zero :meth:`PredictionPostProcessor._remove_eos_token` by
    splitting on ``eos_token`` and keeping the leading fragment only.

    Parameters
    ----------
    ctx : PostProcessContext
        Current pipeline context.
    eos_token : str, optional
        Marker substring to truncate on. Use ``\"<|UNUSED|>\"``-style tokens
        for models that never emit EOS in decoded text.

    Returns
    -------
    PostProcessContext
        Truncated segments.

    Raises
    ------
    PostProcessPipelineError
        If ``eos_token`` is empty (would make truncation ambiguous).
    """
    if eos_token == "":
        raise PostProcessPipelineError(
            "eos_token must not be empty",
            step_name="strip_eos",
        )

    stripped = [s.split(eos_token, maxsplit=1)[0] for s in ctx.segments]
    return replace(ctx, segments=stripped)


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    int
        Minimum number of single-character edits (insert, delete, substitute)
        to transform *a* into *b*.
    """
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m
    row = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(
                row[j] + 1, row[j - 1] + 1, prev + (ca != cb)
            )
    return row[n]


def _best_label_by_distance(
    text: str,
    labels: Sequence[str],
) -> tuple[str, int]:
    """Pick the label with minimum Levenshtein edit distance.

    Ties break lexicographically on the label string (ascending) for a
    deterministic choice with a fixed vocabulary.

    Parameters
    ----------
    text : str
        Candidate fragment to match.
    labels : Sequence[str]
        Allowed labels (duplicates de-duplicated, preserving first occurrence).

    Returns
    -------
    tuple[str, int]
        Chosen label and its Levenshtein distance from *text*.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            unique.append(lab)

    best_label = ""
    best_dist = 10**9
    for lab in unique:
        dist = _levenshtein_distance(text, lab)
        if dist < best_dist or (dist == best_dist and lab < best_label):
            best_dist = dist
            best_label = lab
    return best_label, best_dist


def apply_extract_label_from_text(
    ctx: PostProcessContext,
    *,
    labels: Sequence[str],
    apply_threshold: bool = False,
    max_distance: int = 5,
    default_label: str = "None",
) -> PostProcessContext:
    """Extract the best-matching label from each segment via a three-stage strategy.

    For each segment, attempts in order:

    1. **Exact match** (case-insensitive, after stripping whitespace) — returns
       the matching vocabulary label preserving original case.
    2. **Substring scan** (case-insensitive) — returns the longest vocabulary
       label found as a substring within the segment.
    3. **Levenshtein fuzzy match** — falls back to minimum edit-distance with
       optional threshold rejection, same behaviour as
       :func:`apply_fuzzy_match_to_labels`.

    Stages 1 and 2 both use case-insensitive comparison so ``"Dog"`` matches
    the label ``"dog"`` without requiring the model to produce exact casing.

    Parameters
    ----------
    ctx : PostProcessContext
        Current pipeline context.
    labels : Sequence[str]
        Allowed target labels. Must be non-empty.
    apply_threshold : bool, optional
        If ``True``, segments that reach stage 3 and whose nearest label
        exceeds ``max_distance`` are mapped to ``default_label`` instead.
    max_distance : int, optional
        Maximum Levenshtein distance for threshold rejection (stage 3 only).
    default_label : str, optional
        Replacement label when threshold rejection fires.

    Returns
    -------
    PostProcessContext
        Context with each segment replaced by a vocabulary label (or
        ``default_label`` when threshold rejects it).

    Raises
    ------
    PostProcessPipelineError
        If ``labels`` is empty, ``max_distance`` is negative, or
        ``default_label`` is empty when threshold rejection would occur.
    """
    if not labels:
        raise PostProcessPipelineError(
            "labels must be a non-empty sequence",
            step_name="extract_label_from_text",
        )
    if max_distance < 0:
        raise PostProcessPipelineError(
            f"max_distance must be non-negative, got {max_distance}",
            step_name="extract_label_from_text",
        )

    # Build case-insensitive → original-case mapping (first occurrence wins).
    lower_to_orig: dict[str, str] = {}
    for lab in labels:
        low = lab.lower()
        if low not in lower_to_orig:
            lower_to_orig[low] = lab

    # Sorted longest-first for substring scan: prefer the most specific match.
    labels_by_len = sorted(lower_to_orig.keys(), key=lambda x: -len(x))

    mapped: list[str] = []
    extra: list[str] = []

    def _contains_label(seg_lower: str, low_lab: str) -> bool:
        """Check if ``low_lab`` appears as a standalone label in ``seg_lower``.

        For very short labels (notably 1-character MCQ choices like ``"A"``),
        naive substring matching is unsafe because the character may occur inside
        normal words (e.g. ``"a"`` in ``"based"``). In that case we require
        non-alphanumeric boundaries around the label.

        Returns
        -------
        bool
            ``True`` when the label is present, else ``False``.
        """
        if len(low_lab) <= 2:
            # Accept boundaries like whitespace, punctuation, markdown (**C**), etc.
            # Disallow letters/digits/underscore adjacent to the label.
            return (
                re.search(
                    rf"(?<![0-9a-z_]){re.escape(low_lab)}(?![0-9a-z_])",
                    seg_lower,
                )
                is not None
            )
        return low_lab in seg_lower

    for seg in ctx.segments:
        seg_lower = seg.lower().strip()

        # Stage 1: exact match (case-insensitive).
        if seg_lower in lower_to_orig:
            mapped.append(lower_to_orig[seg_lower])
            continue

        # Stage 2: substring scan (longest label wins).
        found: str | None = None
        for low_lab in labels_by_len:
            if _contains_label(seg_lower, low_lab):
                found = lower_to_orig[low_lab]
                break
        if found is not None:
            mapped.append(found)
            continue

        # Stage 3: Levenshtein on lowercased strings; map result back to
        # original case via lower_to_orig.
        lower_vocab = list(lower_to_orig.keys())
        best_lower, dist = _best_label_by_distance(seg_lower, lower_vocab)
        best = lower_to_orig.get(best_lower, best_lower)

        if apply_threshold and dist > max_distance:
            if default_label == "":
                raise PostProcessPipelineError(
                    "default_label must be non-empty when apply_threshold is True",
                    step_name="extract_label_from_text",
                )
            extra.append(
                f"extract_label_from_text: segment {seg!r} exceeded "
                f"max_distance={max_distance} "
                f"(best={best!r}, distance={dist}); "
                f"using default_label={default_label!r}"
            )
            mapped.append(default_label)
        else:
            mapped.append(best)

    warnings = ctx.warnings + tuple(extra) if extra else ctx.warnings
    return replace(ctx, segments=mapped, warnings=warnings)


def apply_extract_mcq_choice_from_text(
    ctx: PostProcessContext,
    *,
    labels: Sequence[str],
) -> PostProcessContext:
    """Extract the chosen MCQ label from verbose enumerations.

    This targets multiple-choice tasks where labels are single-letter options
    like ``["A", "B", "C", "D"]``. Many LLMs respond by enumerating all options
    before stating a final choice; naive substring/Levenshtein extraction will
    often pick the first mentioned option (usually "A") rather than the final
    selected answer.

    Strategy (per segment):
    - Look for explicit answer markers ("final answer", "answer:", "correct option is").
    - Otherwise, take the last standalone letter token in the segment (common
      in markdown outputs like ``**B**`` on its own line).
    - Fall back to :func:`apply_extract_label_from_text` if no marker matches.

    Parameters
    ----------
    ctx : PostProcessContext
        Current pipeline context.
    labels : Sequence[str]
        Allowed MCQ labels (typically ``["A", "B", "C", "D"]``). Must be non-empty.

    Returns
    -------
    PostProcessContext
        Context with each segment mapped to a single MCQ label.

    Raises
    ------
    PostProcessPipelineError
        If ``labels`` is empty.
    """
    if not labels:
        raise PostProcessPipelineError(
            "labels must be a non-empty sequence",
            step_name="extract_mcq_choice_from_text",
        )

    # Map case-insensitive label tokens to canonical casing from the vocab.
    lower_to_orig: dict[str, str] = {}
    for lab in labels:
        low = str(lab).lower()
        if low not in lower_to_orig:
            lower_to_orig[low] = str(lab)

    allowed: set[str] = set(lower_to_orig.keys())

    def _extract_one(seg: str) -> str | None:
        # If the model output is essentially just a single letter (possibly wrapped
        # in whitespace/markdown/punctuation), accept lowercase too. This handles
        # short answers like "b" without letting the English article "a" inside
        # prose trigger label A.
        seg_stripped = seg.strip()
        # Remove common wrappers. Keep this conservative: only used for short outputs.
        seg_core = re.sub(
            r"^[\s\*\_`\(\[\{<]+|[\s\*\_`\)\]\}>\.!\?:;,'\"]+$",
            "",
            seg_stripped,
        )
        if 1 <= len(seg_core) <= 3:
            low = seg_core.lower()
            if low in allowed and low.isalpha() and len(low) == 1:
                return lower_to_orig[low]

        for pat in _MCQ_MARKER_PATTERNS:
            matches = list(pat.finditer(seg))
            for m in reversed(matches):
                token = (m.group("label") or "").lower()
                if token in allowed:
                    return lower_to_orig[token]

        # Next: "option C" / "choice B" references (uppercase only).
        opt_matches = list(_MCQ_OPTION_REF_RE.finditer(seg))
        for m in reversed(opt_matches):
            token = (m.group("label") or "").lower()
            if token in allowed:
                return lower_to_orig[token]

        # Finally: any standalone uppercase letter token on its own (or in markdown).
        # This intentionally avoids lowercase "a" to prevent the English article
        # from triggering label "A".
        standalone = re.findall(r"(?m)(?<![0-9A-Za-z_])([A-Z])(?![0-9A-Za-z_])", seg)
        for tok in reversed(standalone):
            low = tok.lower()
            if low in allowed:
                return lower_to_orig[low]
        return None

    mapped: list[str] = []
    fallback_ctx = ctx
    for seg in ctx.segments:
        chosen = _extract_one(seg)
        if chosen is None:
            # Fall back to the general extractor (exact → substring → Levenshtein).
            fallback_ctx = apply_extract_label_from_text(
                PostProcessContext(segments=[seg], warnings=fallback_ctx.warnings),
                labels=list(labels),
            )
            mapped.append(fallback_ctx.segments[0])
        else:
            mapped.append(chosen)

    return replace(ctx, segments=mapped)


def apply_extract_hz_bucket_from_text(
    ctx: PostProcessContext,
    *,
    labels: Sequence[str],
) -> PostProcessContext:
    """Extract a numeric frequency and map it to the closest ``"<N> Hz"`` bucket.

    Intended for BEANS-Next F0 mean bucket tasks where the ground truth is a
    discrete bucket label like ``"4010 Hz"``. Some models emit a raw numeric
    estimate like ``"1,654.75"`` (including thousands separators). Others emit
    prose containing commas; naive comma-splitting produces repeated buckets.

    Strategy (per segment):
    - Parse all numbers in the segment (supports thousands separators).
    - Use the **last** number (often appears at the end in NatureLM-style outputs).
    - Map to the closest bucket by absolute difference.
    - If no number is present, fall back to :func:`apply_extract_label_from_text`
      to pick a single bucket label without comma-splitting.

    Parameters
    ----------
    ctx : PostProcessContext
        Current pipeline context.
    labels : Sequence[str]
        Bucket labels. Expected to look like ``"<int> Hz"``.

    Returns
    -------
    PostProcessContext
        Context with each segment mapped to a single bucket label.

    Raises
    ------
    PostProcessPipelineError
        If ``labels`` is empty.
    """
    if not labels:
        raise PostProcessPipelineError(
            "labels must be a non-empty sequence",
            step_name="extract_hz_bucket_from_text",
        )

    buckets: list[tuple[float, str]] = []
    for lab in labels:
        if not isinstance(lab, str):
            continue
        m = _HZ_LABEL_RE.match(lab)
        if m is None:
            continue
        buckets.append((float(m.group("hz")), lab.strip()))
    if not buckets:
        # Nothing looks like "<N> Hz" — delegate to generic extractor.
        return apply_extract_label_from_text(ctx, labels=list(labels))

    buckets.sort(key=lambda x: x[0])

    def _nearest_bucket(value: float) -> str:
        best_label = buckets[0][1]
        best_dist = abs(value - buckets[0][0])
        for hz, lab in buckets[1:]:
            dist = abs(value - hz)
            if dist < best_dist:
                best_dist = dist
                best_label = lab
        return best_label

    mapped: list[str] = []
    for seg in ctx.segments:
        # Special-case: NatureLM sometimes emits a range like "2131-4440 Hz"
        # which is best interpreted as deci-Hz (213.1–444.0 Hz). Use the midpoint.
        m_range = _HZ_RANGE_RE.search(seg)
        if m_range is not None:
            try:
                a = float(m_range.group("a"))
                b = float(m_range.group("b"))
                # Heuristic: if both ends are "too large" for plausible F0 (Hz),
                # interpret as deci-Hz. This matches observed NatureLM outputs.
                scale = 0.1 if (a >= 1000 and b >= 1000 and max(a, b) <= 50000) else 1.0
                mapped.append(_nearest_bucket(((a + b) / 2.0) * scale))
                continue
            except ValueError:
                # Fall through to generic numeric parsing.
                pass

        nums = [m.group("num") for m in _NUMBER_RE.finditer(seg)]
        if not nums:
            out = apply_extract_label_from_text(
                PostProcessContext(segments=[seg], warnings=ctx.warnings),
                labels=list(labels),
            )
            mapped.append(out.segments[0])
            continue
        raw_num = nums[-1].replace(",", "")
        try:
            val = float(raw_num)
        except ValueError:
            out = apply_extract_label_from_text(
                PostProcessContext(segments=[seg], warnings=ctx.warnings),
                labels=list(labels),
            )
            mapped.append(out.segments[0])
            continue
        mapped.append(_nearest_bucket(val))

    return replace(ctx, segments=mapped)


def apply_fuzzy_match_to_labels(
    ctx: PostProcessContext,
    *,
    labels: Sequence[str],
    apply_threshold: bool = False,
    max_distance: int = 5,
    default_label: str = "None",
) -> PostProcessContext:
    """Map each segment to the closest string in ``labels`` via Levenshtein distance.

    Mirrors BEANS-Zero's ``PredictionPostProcessor._get_nearest_label``: finds
    the label with minimum edit distance, then optionally rejects low-confidence
    matches (detection style) by returning ``default_label`` when
    ``levenshtein_distance > max_distance``.

    Parameters
    ----------
    ctx : PostProcessContext
        Current pipeline context.
    labels : Sequence[str]
        Allowed target labels. Must be non-empty.
    apply_threshold : bool, optional
        If ``True``, segments whose nearest label exceeds ``max_distance``
        are mapped to ``default_label`` instead (detection style).
    max_distance : int, optional
        Maximum Levenshtein distance for ``apply_threshold`` mode. Matches
        with ``distance > max_distance`` are rejected.
    default_label : str, optional
        Replacement label used when a segment exceeds the distance threshold.

    Returns
    -------
    PostProcessContext
        Context with each segment replaced by a vocabulary label (or
        ``default_label``).

    Raises
    ------
    PostProcessPipelineError
        If ``labels`` is empty, ``max_distance`` is negative, or
        ``default_label`` is empty when a threshold rejection would occur.
    """
    if not labels:
        raise PostProcessPipelineError(
            "labels must be a non-empty sequence",
            step_name="fuzzy_match_to_labels",
        )
    if max_distance < 0:
        raise PostProcessPipelineError(
            f"max_distance must be non-negative, got {max_distance}",
            step_name="fuzzy_match_to_labels",
        )

    mapped: list[str] = []
    extra: list[str] = []
    for seg in ctx.segments:
        best, dist = _best_label_by_distance(seg, labels)
        if apply_threshold and dist > max_distance:
            if default_label == "":
                raise PostProcessPipelineError(
                    "default_label must be non-empty when apply_threshold is True",
                    step_name="fuzzy_match_to_labels",
                )
            extra.append(
                f"fuzzy_match_to_labels: segment {seg!r} exceeded "
                f"max_distance={max_distance} "
                f"(best={best!r}, distance={dist}); "
                f"using default_label={default_label!r}"
            )
            mapped.append(default_label)
        else:
            mapped.append(best)

    warnings = ctx.warnings + tuple(extra) if extra else ctx.warnings
    return replace(ctx, segments=mapped, warnings=warnings)


def apply_conservative_match_to_labels(
    ctx: PostProcessContext,
    *,
    labels: Sequence[str],
    max_distance: int = 2,
    allow_fuzzy: bool = True,
    drop_unmatched: bool = True,
) -> PostProcessContext:
    """Conservatively align each segment to a label vocabulary.

    This is intended for multi-label detection answers where we would rather
    *drop* unrecognized fragments than force them into the nearest label.

    Matching strategy (per segment):
    1. Case-insensitive exact match.
    2. Case-insensitive substring scan (longest label wins).
    3. Optional Levenshtein match (lowercased) accepted only when
       ``distance <= max_distance``.

    Parameters
    ----------
    ctx
        Current pipeline context (typically already comma-split).
    labels
        Allowed label vocabulary. Must be non-empty.
    max_distance
        Maximum Levenshtein edit distance accepted for fuzzy matches.
    allow_fuzzy
        Whether to attempt stage-3 Levenshtein matching.
    drop_unmatched
        When ``True`` (default), segments that cannot be matched confidently
        are removed. When ``False``, unmatched segments are preserved verbatim.

    Returns
    -------
    PostProcessContext
        Context with segments replaced by matched vocabulary labels. Unmatched
        segments are dropped (or preserved) depending on ``drop_unmatched``.

    Raises
    ------
    PostProcessPipelineError
        If ``labels`` is empty or contains no non-empty strings, or if
        ``max_distance`` is negative.
    """
    if not labels:
        raise PostProcessPipelineError(
            "labels must be a non-empty sequence",
            step_name="conservative_match_to_labels",
        )
    if max_distance < 0:
        raise PostProcessPipelineError(
            f"max_distance must be non-negative, got {max_distance}",
            step_name="conservative_match_to_labels",
        )

    # Build case-insensitive → original-case mapping (first occurrence wins).
    lower_to_orig: dict[str, str] = {}
    for lab in labels:
        if not isinstance(lab, str):
            continue
        low = lab.strip().lower()
        if low and low not in lower_to_orig:
            lower_to_orig[low] = lab.strip()

    if not lower_to_orig:
        raise PostProcessPipelineError(
            "labels must contain at least one non-empty string",
            step_name="conservative_match_to_labels",
        )

    labels_by_len = sorted(lower_to_orig.keys(), key=lambda x: -len(x))

    def _contains_label(seg_lower: str, low_lab: str) -> bool:
        # Use the same boundary logic as extract_label_from_text for short labels.
        if len(low_lab) <= 2:
            return (
                re.search(
                    rf"(?<![0-9a-z_]){re.escape(low_lab)}(?![0-9a-z_])",
                    seg_lower,
                )
                is not None
            )
        return low_lab in seg_lower

    mapped: list[str] = []
    extra: list[str] = []
    stopwords: set[str] = {
        "a",
        "an",
        "the",
        "definitely",
        "probably",
        "likely",
        "maybe",
        "it",
        "is",
        "sounds",
        "sound",
        "like",
        "detected",
        "detect",
        "i",
        "hear",
        "hearing",
    }
    for seg in ctx.segments:
        seg_s = str(seg).strip()
        if not seg_s:
            continue
        seg_lower = seg_s.lower()
        # Normalize to a label-ish core phrase:
        # - remove punctuation that frequently surrounds labels
        # - drop common lead-in words to make fuzzy matching usable
        seg_norm = re.sub(r"[^0-9a-z_\s\-]", " ", seg_lower)
        seg_norm = _WS_RE.sub(" ", seg_norm).strip()
        toks = seg_norm.split()
        while len(toks) >= 2 and toks[0] in stopwords:
            toks = toks[1:]
        seg_norm = " ".join(toks)

        # Stage 1: exact match.
        if seg_norm in lower_to_orig:
            mapped.append(lower_to_orig[seg_norm])
            continue

        # Stage 2: substring scan.
        found: str | None = None
        for low_lab in labels_by_len:
            if _contains_label(seg_norm, low_lab):
                found = lower_to_orig[low_lab]
                break
        if found is not None:
            mapped.append(found)
            continue

        # Stage 3: bounded fuzzy match.
        if allow_fuzzy:
            best_lower, dist = _best_label_by_distance(seg_norm, labels_by_len)
            if dist <= max_distance:
                mapped.append(lower_to_orig.get(best_lower, best_lower))
                continue
            extra.append(
                f"conservative_match_to_labels: dropped segment {seg_s!r} "
                f"(best={best_lower!r}, distance={dist} > max_distance={max_distance})"
            )

        if not drop_unmatched:
            mapped.append(seg_s)

    warnings = ctx.warnings + tuple(extra) if extra else ctx.warnings
    return replace(ctx, segments=mapped, warnings=warnings)
