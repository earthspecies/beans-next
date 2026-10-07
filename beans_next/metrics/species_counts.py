"""Target-independent parsing of per-species vocalization counts."""

from __future__ import annotations

import re

from beans_next.metrics.base import MetricsError
from beans_next.metrics.regression import extract_numeric_value
from beans_next.post_process.answers import clean_answer

_COUNT = r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
_COLON = re.compile(
    r"(?P<name>[^\n,:;]+?)(?:\s*:\s*|\s+-\s+)"
    r"(?:(?:there (?:is|are) )?(?:only )?)"
    r"(?P<count>" + _COUNT + r")\b",
    re.I,
)
_PROSE = re.compile(
    r"(?P<count>" + _COUNT + r")\s+"
    r"(?:(?:distinct|individual|separate)\s+)?"
    r"(?:calls?|songs?|vocalizations?)\s+"
    r"(?:from\s+|heard[^\n:]*:\s*(?:that of\s+)?)"
    r"(?P<name>[^\n,;.]+?)(?=\s+and\s+|[,.;\n]|$)",
    re.I,
)


def _name(raw: str) -> str | None:
    raw = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw.strip())
    # Use a scientific name explicitly supplied alongside a common name.
    parenthesis = re.search(r"\(([A-Z][a-z]+\s+[a-z][a-z-]+)\)", raw)
    if parenthesis:
        raw = parenthesis.group(1)
    else:
        raw = re.sub(r"\s*\([^)]*\)", "", raw)
        raw = re.sub(
            r"\s+in (?:this|the) (?:recording|clip|audio).*$", "", raw, flags=re.I
        )
        raw = re.sub(r"^(?:a|an|the)\s+", "", raw, flags=re.I)
    raw = raw.strip(" *\"'").casefold()
    if (
        not re.fullmatch(r"[a-z][a-z '\-]*", raw)
        or len(raw.split()) > 5
        or raw
        in {
            "answer",
            "count",
            "total",
            "species",
            "bird species",
            "bird",
            "calls",
            "vocalizations",
            "scientific name",
        }
        or raw.startswith(("there ", "based ", "i ", "here "))
    ):
        return None
    return raw


def parse_species_counts(text: str) -> dict[str, float] | None:
    """Return counts, an explicit empty prediction {}, or None for invalid text.

    Missing species in a valid mapping remain scoring errors. Parsing never
    consults target species or counts. Conflicting duplicate counts and ranges
    are rejected instead of selecting an interpretation using the target.

    Returns
    -------
    dict[str, float] | None
        Named counts, an empty mapping for explicit absence, or invalidity.
    """
    text = clean_answer(text).replace("*", "")
    plain = " ".join(text.casefold().split()).strip(" .!\"'")
    if plain in {
        "none",
        "no calls",
        "no vocalizations",
        "no species",
        "{}",
        "[]",
    } or re.fullmatch(
        r"(?:there (?:are|were) )?no (?:calls|vocalizations|species)"
        r"(?: from any species)?(?: (?:are |were )?(?:detected|audible))?"
        r"(?: in (?:this|the) (?:recording|audio|clip))?",
        plain,
    ):
        return {}
    result: dict[str, float] = {}
    for pattern in (_COLON, _PROSE):
        for match in pattern.finditer(text):
            name = _name(match.group("name"))
            if name is None:
                continue
            prefix = text[max(0, match.start() - 12) : match.start()]
            tail = text[match.end("count") :]
            if re.search(
                r"(?:at least|at most|more than|less than)\s*$", prefix, re.I
            ) or re.match(r"\s*(?:-|–|to\b|or\b)", tail):
                return None
            try:
                count = extract_numeric_value(match.group("count"))
            except MetricsError:
                continue
            if name in result and result[name] != count:
                return None
            result[name] = count
    return result or None
