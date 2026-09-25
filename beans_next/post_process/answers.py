"""Target-independent answer extraction for the reported evaluation tasks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

INVALID_ANSWER = "__invalid_answer__"
SCORING_VERSION = "2026-09-25-v5"
FREE_TEXT_TASKS = frozenset(
    {
        "captioning",
        "qa",
        "open_ended",
        "counting",
        "regression",
        "species_name",
        "species_order",
        "species_set",
        "species_count_dict",
        "species_summary",
        "species_freq_range",
        "frequency_range",
        "multilabel_accuracy",
        "multilabel_detection",
        "multilabel_classification",
    }
)
CALL_TYPES = ("alarm call", "flight call", "begging call", "song", "call")


def clean_answer(text: str) -> str:
    """Remove formatting and known terminal tokens without choosing an answer.

    Returns
    -------
    str
        Cleaned answer text.
    """
    text = re.split(r"<\|(?:end_of_text|eot_id|im_end)\|>", text, maxsplit=1)[0]
    text = re.sub(r"#\s*\d+(?:\.\d+)?s?\s*[-–]\s*\d+(?:\.\d+)?s?\s*#\s*:\s*", "", text)
    return text.replace("**", "").replace("`", "").strip()


def _plain(text: str) -> str:
    return " ".join(clean_answer(text).casefold().split()).strip(" .!\"'")


def explicit_empty(text: str) -> bool:
    """Accept explicit absence, not arbitrary text that failed extraction.

    Returns
    -------
    bool
        Whether the answer explicitly states absence.
    """
    return _plain(text) in {
        "none",
        "none of the above",
        "no species",
        "no species present",
        "no species are present",
        "no identifiable species",
        "no identifiable species are present",
        "no identifiable species are present in this audio",
        "no vocalizations",
        "no vocalizations detected",
        "no sounds",
        "[]",
        "{}",
    }


def question_text(metadata: Mapping[str, Any]) -> str:
    """Read user input only; assistant messages and labels never supply options.

    Returns
    -------
    str
        User question text, or an empty string.
    """
    for key in ("instruction", "conversation", "question", "prompt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = metadata.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            m.get("content", "")
            for m in messages
            if isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
        )
    return ""


def question_options(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Extract lettered choices from the question, preserving per-row mappings.

    Returns
    -------
    dict[str, str]
        Per-example letter-to-option mapping.
    """
    text = question_text(metadata)
    # Formats: Options: a) ..., b) ...; newline A: ...; (A) ... .
    pattern = r"(?:^|\n|[,;]\s*|Options?:\s*)\s*\(?([A-Ha-h])[).:]\s*"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    result: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip()
        value = re.split(r"\n\s*(?:Answer|Respond|Which|What)\b", value, maxsplit=1)[0]
        key = match.group(1).lower()
        if key in result:
            return {}  # Conflicting/repeated option blocks need explicit repair.
        result[key] = value
    return result if len(result) >= 2 else {}


def _unwrap(text: str) -> str:
    text = clean_answer(text).strip()
    return re.sub(
        r"^(?:(?:the\s+)?(?:correct\s+|final\s+)?(?:answer|option|choice|species)"
        r"\s*(?:is\s*|:|=)|I\s+(?:choose|select)\s+)",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def parse_choice_set(
    text: str,
    options: Mapping[str, str] | None = None,
    *,
    multiple: bool = False,
) -> set[str] | None:
    """Parse an explicit selection; reject enumeration, negation and conflicts.

    No correct answer is accepted as input. Without the original question,
    only explicit letters are recoverable, never guessed option text.

    Returns
    -------
    set[str] | None
        Selected letters, or None for an ambiguous answer.
    """
    text = _unwrap(text).strip().rstrip(".! ")
    audio_only = bool(options) and all(
        "<AudioHere>" in value for value in options.values()
    )
    # Selection statements may be followed by reasoning that discusses all options.
    # Extract only explicit statements, never arbitrary option mentions.
    letter_set = r"[A-H](?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)[A-H])*"
    selections = []
    patterns = [
        r"(?:final answer|answer)\s*:\s*",
        r"the (?:correct|best) (?:answer|match|option|choice) is\s*:?\s*",
        r"(?:therefore,?\s*)?(?:the )?sounds? present in "
        r"(?:this|the) recording (?:is|are)\s*:?\s*",
    ]
    if audio_only:
        patterns += [
            r"(?:best |closely |most closely )?matches?\s+",
            r"(?:recording|audio) (?:contains|corresponds to|is most likely from)\s+"
            r"(?:(?:sounds from|elements of)\s+)?",
            r"(?:therefore,?\s*)only\s+",
            r"(?:species|call type|sound) that best matches the recording is\s*:?\s*",
            r"matches best with\s*:?\s*",
        ]
    for lead in patterns:
        pattern = (
            lead
            + r"(?:>\s*)?(?:sound |species |option |call type )?("
            + letter_set
            + r")(?=\s*(?:[.,:—]|$)|\n)"
        )
        selections.extend(m.group(1) for m in re.finditer(pattern, text, re.I))
    if audio_only:
        reverse = (
            r"\b(" + letter_set + r") (?:is|would be) the (?:correct answer|best match)"
        )
        selections.extend(m.group(1) for m in re.finditer(reverse, text, re.I))
        short = re.fullmatch(
            r"(?:species|sound|call type) (" + letter_set + r")", text, re.I
        )
        if short:
            selections.append(short.group(1))
    if multiple and audio_only:
        all_sounds = re.search(
            r"all (?:two|three|four|five|six|[2-6]) sounds? \(([^)]+)\) are present",
            text,
            re.I,
        )
        if all_sounds:
            selections.append(all_sounds.group(1))
        listed = re.search(
            r"(?:following sounds are present|sounds present in "
            r"(?:this|the) recording are):\s*\n(.+)",
            text,
            re.I | re.S,
        )
        if listed:
            block = listed.group(1).split("\n\nTherefore")[0]
            bullets = re.findall(
                r"^\s*[-*]\s*(?:Sound )?([A-H])(?:\s*[:.]|\s*$)", block, re.I | re.M
            )
            if bullets and not re.search(r"\b(?:not|absent|except)\b", block, re.I):
                selections.append(", ".join(bullets))
    if selections:
        sets = {frozenset(re.findall(r"\b[a-h]\b", x.lower())) for x in selections}
        if len(sets) != 1:
            return None
        text = ", ".join(sorted(next(iter(sets))))
    if multiple and re.match(
        r"^None of (?:these|the (?:given|above|provided)) sounds\b", text, re.I
    ):
        # A direct absence statement; later justification is not another selection.
        if not re.search(r"\b(?:but|however|except)\b", text, re.I):
            return set()
    list_match = re.search(
        r"sounds? present in (?:this|the) recording (?:is|are):\s*\n"
        r"((?:\s*[-*]\s*(?:Sound )?[A-H]\s*\n?)+)",
        text,
        re.I,
    )
    if list_match:
        text = ", ".join(
            re.findall(r"[-*]\s*(?:Sound )?([A-H])", list_match.group(1), re.I)
        )
    if explicit_empty(text):
        return set() if multiple else None
    allowed = set(options) if options else set("abcdefgh")
    # Bare letter(s), including brackets, comma/and-separated sets.
    bare = text.strip("[]():. ").lower()
    if re.fullmatch(r"[a-h](?:\s*(?:,|\band\b|&)\s*[a-h])*", bare):
        chosen = set(re.findall(r"[a-h]", re.sub(r"\band\b", ",", bare)))
        return chosen if chosen <= allowed and (multiple or len(chosen) == 1) else None
    prefix_only = re.match(r"^\(?([A-Ha-h])\)?[).:]\s*(.+)$", text, re.DOTALL)
    audio_only = bool(options) and all(
        "<AudioHere>" in value for value in options.values()
    )
    if not options or audio_only:
        # Audio choices have no textual species labels to contradict. With no
        # saved question, retain only explicit letters and flag provenance upstream.
        if prefix_only and not re.search(
            r"(?:^|\n|[,;])\s*\(?[A-Ha-h][).:]", prefix_only.group(2)
        ):
            return (
                {prefix_only.group(1).lower()}
                if prefix_only.group(1).lower() in allowed
                else None
            )
        return None
    if re.search(r"\b(?:not|neither|except|isn't|isnt)\b", text, re.IGNORECASE):
        return None
    embedded = list(re.finditer(r"(?<!\w)\(?([A-Ha-h])\)?[).:]\s+(.+)", text))
    # One explicit letter+text answer inside prose; repeated options are ambiguous.
    if len(embedded) == 1:
        text = text[embedded[0].start() :]
    prefixed = re.match(r"^\(?([A-Ha-h])\)?[).:\s]+(.+)$", text, re.DOTALL)
    content = prefixed.group(2) if prefixed else text
    # The fixed alias lexicon is independent of this example's answer key.
    from beans_next.metrics import _t3_species_names

    def equivalent(candidate: str, option: str) -> bool:
        candidate, option = _plain(candidate), _plain(option)
        if candidate == option:
            return True
        # Permit explanation after an exact selected option, not a substring
        # somewhere in the response. Do not ignore an alternative selection.
        if prefixed and candidate.startswith(option):
            tail = candidate[len(option) :]
            if re.match(r"^[.,;\n]", tail) and not re.search(
                r"\b(?:or|not|instead)\b", tail
            ):
                return True
        matches = [
            scientific
            for scientific, commons in _t3_species_names().items()
            if candidate in {scientific, *commons}
        ]
        return len(matches) == 1 and option in {
            matches[0],
            *_t3_species_names()[matches[0]],
        }

    hits = {key for key, value in options.items() if equivalent(content, value)}
    if len(hits) != 1:
        return None
    if prefixed and prefixed.group(1).lower() not in hits:
        return None
    return hits


def parse_call_types(text: str) -> set[str] | None:
    """Parse the five advertised call types without treating 'alarm call' as 'call'.

    Returns
    -------
    set[str] | None
        Recognized labels with an invalid-item marker, or None.
    """
    if explicit_empty(text):
        return set()
    text = _plain(_unwrap(text))
    text = re.sub(r"^(?:the following (?:are present|can be heard):\s*)", "", text)
    parts = [p.strip(" .") for p in re.split(r"[,;\n]|\band\b", text) if p.strip()]
    known = set(parts) & set(CALL_TYPES)
    if not known:
        return None
    # A malformed extra item must not erase the other explicitly listed labels.
    # Preserve its presence for parse coverage and exact-set accuracy.
    return known | (
        {INVALID_ANSWER} if any(p not in CALL_TYPES for p in parts) else set()
    )


def normalize_task_answer(
    raw: str, task_type: str | None, metadata: Mapping[str, Any]
) -> str:
    """Normalize semantic answers before artifact writing and aggregation.

    Returns
    -------
    str
        Normalized text suitable for storage and aggregation.
    """
    kind = (task_type or "").lower()
    if kind == "multilabel_classification":
        parsed = parse_call_types(raw)
        return INVALID_ANSWER if parsed is None else ", ".join(sorted(parsed)) or "None"
    options = question_options(metadata)
    if kind in {"multilabel_accuracy", "multilabel_detection"} or (
        kind == "classification" and options
    ):
        parsed = parse_choice_set(
            raw,
            options,
            multiple=kind in {"multilabel_accuracy", "multilabel_detection"},
        )
        return INVALID_ANSWER if parsed is None else ", ".join(sorted(parsed)) or "None"
    return clean_answer(raw)
