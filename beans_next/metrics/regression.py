"""Regression metrics for numeric open-ended answers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from beans_next.metrics.base import MetricsError, register_scorer, validate_equal_length

__all__ = [
    "extract_frequency_range",
    "extract_numeric_value",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "root_mean_squared_error",
]

_NUMBER_RE = re.compile(r"(?P<num>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_RANGE_RE = re.compile(
    r"(?P<a>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?:-|to|–|—)\s*"
    r"(?P<b>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<unit>khz|hz|db)?\b",
    flags=re.IGNORECASE,
)


def _coerce_float(raw: object) -> float:
    """Convert a scalar-like value to float.

    Parameters
    ----------
    raw
        Numeric value or numeric string.

    Returns
    -------
    float
        Parsed numeric value.

    Raises
    ------
    MetricsError
        If ``raw`` cannot be parsed as a finite float.
    """
    if isinstance(raw, bool):
        raise MetricsError("Boolean values are not valid regression targets.")
    if isinstance(raw, (int, float)):
        val = float(raw)
    elif isinstance(raw, str):
        val = float(raw.replace(",", ""))
    else:
        raise MetricsError(f"Unsupported numeric value type: {type(raw).__name__}.")
    if not math.isfinite(val):
        raise MetricsError("Regression values must be finite.")
    return val


def _unit_hint(text: str) -> str | None:
    low = text.lower()
    if "khz" in low:
        return "khz"
    if "hz" in low:
        return "hz"
    if "db" in low:
        return "db"
    return None


def _convert_to_target_unit(
    value: float,
    *,
    source_unit: str | None,
    target_unit: str | None,
) -> float:
    if source_unit == "khz" and target_unit == "hz":
        return value * 1000.0
    return value


def extract_numeric_value(
    value: object,
    *,
    target_value: float | None = None,
    unit: str | None = None,
) -> float:
    """Extract an unambiguous scalar or range midpoint, independently of targets.

    ``target_value`` is retained for API compatibility and intentionally ignored.
    Explicit kHz values are converted to Hz. Multiple distinct measurements,
    incompatible units, and unrelated numbers are rejected rather than guessed.

    Returns
    -------
    float
        Numeric answer in the requested units.

    Raises
    ------
    MetricsError
        If no unambiguous finite measurement can be extracted.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            raise MetricsError("Expected exactly one numeric answer.")
        value = value[0]
    if not isinstance(value, str):
        return _coerce_float(value)
    text = re.sub(r"\bf_?0\b", "fundamental frequency", value.strip(), flags=re.I)
    # NatureLM timestamps describe the clip, not the requested measurement.
    text = re.sub(r"\#[^#]*\#\s*:\s*", "", text)
    text = re.sub(r"\[\d+(?:\.\d+)?\s*(?:-|–)\s*\d+(?:\.\d+)?\]", "", text)
    target_unit = (unit or "").lower() or None

    # Decimal commas with one/two fractional digits and an explicit unit are
    # unambiguous here; preserve thousands separators such as 1,200 Hz.
    text = re.sub(
        r"(?<![\d,])([+-]?\d+),(\d{1,2})(?=\s*(?:dB|kHz|Hz)\b)",
        r"\1.\2",
        text,
        flags=re.I,
    )
    text = text.replace("**", "").replace("`", "")
    for spelling, abbreviation in [
        ("kilohertz", "kHz"),
        ("hertz", "Hz"),
        ("decibels?", "dB"),
    ]:
        text = re.sub(r"\b" + spelling + r"\b", abbreviation, text, flags=re.I)
    if target_unit:
        labeled_scalar = re.fullmatch(
            r"\s*(?:SNR|fundamental frequency)\s*(?:is|:|=)?\s*"
            r"([+-]?\d+(?:\.\d+)?)(?:\s*\([^\d]*\))?\s*",
            text,
            re.I,
        )
        if labeled_scalar:
            return _coerce_float(labeled_scalar.group(1))
    # A stated answer precedes supporting discussion/list numbering. Extract
    # only an explicit answer clause, never whichever number fits the target.
    if target_unit in {"hz", "khz", "db"}:
        quantity = (
            r"(?:mean\s+)?fundamental frequency"
            if target_unit != "db"
            else r"(?:signal[- ]to[- ]noise ratio(?:\s*\(SNR\))?|SNR)"
        )
        primary = re.match(
            r"^\s*(?:The\s+)?(?:estimated\s+)?"
            + quantity
            + r"(?:\s+(?:of|for)\s+[^\n:.!?]{0,100}?)?\s*(?:is|:|=)\s*"
            + r"(?:approximately\s*|about\s*|around\s*|~\s*)?"
            + r"([+-]?[\d,]+(?:\.\d+)?\s*(?:kHz|Hz|dB))(?=[.,;\s]|$)",
            text,
            re.I,
        )
        if primary:
            # Do not turn the first endpoint of a range into a point estimate.
            tail = text[primary.end() :]
            if not re.match(r"\s*(?:-|–|—|to\b|or\b|and\b)", tail):
                return extract_numeric_value(primary.group(1), unit=target_unit)
    else:
        words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
        }
        for ten, value10 in {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }.items():
            words[ten] = value10
            for digit in [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]:
                for sep in ["", " ", "-"]:
                    words[ten + sep + digit] = value10 + words[digit]
        number_words = "|".join(
            re.escape(w) for w in sorted(words, key=len, reverse=True)
        )
        count_pattern = r"(?:\d+(?:\.\d+)?|" + number_words + r"|no)"
        primary = re.match(
            r"^\s*(?:(?:Based on|In) [^,\n]+,\s*)?"
            r"(?:There (?:are|is|appear to be|appears to be)|"
            r"I (?:can )?(?:hear|detect|count|identify))\s+"
            r"(?:(?:only|approximately|about|at least|a total of)\s+)?"
            r"(?P<count>" + count_pattern + r")\s+(?:(?:instance|type) of\s+)?"
            r"(?:(?:different|distinct|individual|separate|bird|animal|flight|audible|crowing)\s+)*"
            r"(?:species|birds?|calls?|vocalizations?|events?)\b",
            text,
            re.I,
        )
        # 'at least' is a lower bound, not an exact count.
        if primary and "at least" not in primary.group().lower():
            token = primary.group("count").lower()
            first_sentence_tail = re.split(
                r"[.!?\n]", text[primary.end() :], maxsplit=1
            )[0]
            if not re.search(
                r"\b(?:or|but|and)\b.*(?:\d|" + number_words + r")",
                first_sentence_tail,
                re.I,
            ):
                return float(
                    0 if token == "no" else words[token] if token in words else token
                )
        bare_word = re.fullmatch(r"\s*(" + number_words + r")[.!]?\s*", text, re.I)
        if bare_word:
            return float(words[bare_word.group(1).lower()])
    if re.search(
        r"\b(?:at least|at most|above|below|more than|less than)\s+[+-]?\d", text, re.I
    ):
        raise MetricsError("A one-sided bound is not a numerical estimate.")
    if re.search(r"\b(?:to|or|and)\s*$", text, re.I):
        raise MetricsError("Truncated numerical answer.")
    number = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    ranges = list(
        re.finditer(
            rf"(?P<a>{number})\s*(?P<ua>khz|hz|db)?\s*(?:-|to|–|—)\s*"
            rf"(?P<b>{number})\s*(?P<ub>khz|hz|db)?\b",
            text,
            re.IGNORECASE,
        )
    )

    def convert(raw: str, source: str | None) -> float:
        source = source.lower() if source else None
        if target_unit and source and ((target_unit == "db") != (source == "db")):
            raise MetricsError("Answer has incompatible units.")
        result = _coerce_float(raw)
        if source == "khz" and target_unit != "khz":
            result *= 1000
        elif source == "hz" and target_unit == "khz":
            result /= 1000
        return result

    if ranges:
        if len(ranges) != 1:
            raise MetricsError("Multiple numeric ranges are ambiguous.")
        m = ranges[0]
        if _NUMBER_RE.search(text[: m.start()] + text[m.end() :]):
            raise MetricsError("Range plus additional numbers is ambiguous.")
        ua = m.group("ua") or m.group("ub") or target_unit
        ub = m.group("ub") or m.group("ua") or target_unit
        return (convert(m.group("a"), ua) + convert(m.group("b"), ub)) / 2
    if re.search(r"\brange(?:s)? (?:anywhere )?(?:from|between)\b", text, re.I):
        raise MetricsError("Incomplete numerical range.")
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        word_match = re.fullmatch(
            r"\s*(?:there (?:are|is)\s+)?("
            + "|".join(words)
            + r")(?:\s+(?:species|calls?|vocalizations?))?[.!]?\s*",
            text,
            re.IGNORECASE,
        )
        if word_match and target_unit is None:
            return float(words[word_match.group(1).lower()])
        raise MetricsError(f"No numeric answer found in {value!r}.")
    values = []
    for match in matches:
        suffix = text[match.end() :]
        unit_match = re.match(r"\s*(khz|hz|db)\b", suffix, re.IGNORECASE)
        # Numbers in numbered instructions or formulas (e.g. 10 * log10)
        # are not physical measurements. Unitless answers must be bare scalars.
        if (
            target_unit
            and not unit_match
            and not re.fullmatch(number + r"[.!]?", text.strip())
        ):
            continue
        # A time value is never a count/F0/SNR answer.
        if not unit_match and re.match(
            r"\s*(?:s|sec|seconds?)\b", suffix, re.IGNORECASE
        ):
            continue
        source = unit_match.group(1) if unit_match else target_unit
        values.append(convert(match.group("num"), source))
    if not values or len(set(values)) != 1:
        raise MetricsError("Expected one unambiguous numeric answer.")
    return values[0]


def extract_frequency_range(text: str) -> tuple[float, float]:
    """Extract a ``(low_hz, high_hz)`` frequency range from text.

    When the text contains a range pattern such as ``"200-8000 Hz"`` or
    ``"1.5 to 4 kHz"``, both bounds are returned after unit normalisation to
    Hz.  When only a single numeric value is found (degenerate point range),
    both elements of the tuple are equal to that value.

    Parameters
    ----------
    text
        Raw label or model prediction string.

    Returns
    -------
    tuple[float, float]
        ``(low_hz, high_hz)`` with ``low_hz <= high_hz``.

    Raises
    ------
    MetricsError
        If no numeric value can be extracted from ``text``.
    """
    if not isinstance(text, str) or not text.strip():
        raise MetricsError(
            "Cannot extract a frequency range from empty or non-string input."
        )
    range_match = _RANGE_RE.search(text)
    if range_match is not None:
        a = _coerce_float(range_match.group("a"))
        b = _coerce_float(range_match.group("b"))
        unit = (range_match.group("unit") or "").lower()
        if unit == "khz":
            a *= 1000.0
            b *= 1000.0
        return min(a, b), max(a, b)
    val = extract_numeric_value(text, unit="hz")
    return val, val


def _paired_numeric_values(
    predictions: Sequence[Any], targets: Sequence[Any]
) -> tuple[list[float], list[float]]:
    validate_equal_length(predictions, targets)
    pred_vals: list[float] = []
    target_vals: list[float] = []
    for pred, target in zip(predictions, targets, strict=True):
        target_val = extract_numeric_value(target)
        target_unit = _unit_hint(str(target))
        pred_val = extract_numeric_value(
            pred,
            target_value=target_val,
            unit=target_unit,
        )
        pred_vals.append(pred_val)
        target_vals.append(target_val)
    return pred_vals, target_vals


@register_scorer
def mean_absolute_error(predictions: Sequence[Any], targets: Sequence[Any]) -> float:
    """Mean absolute error for numeric predictions.

    Parameters
    ----------
    predictions
        Predicted numeric values or text containing a numeric value.
    targets
        Target numeric values or text containing a numeric value.

    Returns
    -------
    float
        Mean absolute error in the target units.
    """
    pred_vals, target_vals = _paired_numeric_values(predictions, targets)
    return sum(abs(p - t) for p, t in zip(pred_vals, target_vals, strict=True)) / float(
        len(pred_vals)
    )


@register_scorer
def mean_absolute_percentage_error(
    predictions: Sequence[Any], targets: Sequence[Any]
) -> float:
    """Mean absolute percentage error for numeric predictions.

    The returned value is a fraction, not a percentage-point value. For example,
    a return value of ``0.25`` corresponds to ``25%``.

    Parameters
    ----------
    predictions
        Predicted numeric values or text containing a numeric value.
    targets
        Target numeric values or text containing a numeric value.

    Returns
    -------
    float
        Mean absolute percentage error as a fraction.

    Raises
    ------
    MetricsError
        If inputs are empty, lengths differ, any value cannot be parsed, or any
        target value is zero.
    """
    pred_vals, target_vals = _paired_numeric_values(predictions, targets)
    if any(t == 0.0 for t in target_vals):
        raise MetricsError("MAPE is undefined for zero-valued regression targets.")
    return sum(
        abs((p - t) / t) for p, t in zip(pred_vals, target_vals, strict=True)
    ) / float(len(pred_vals))


@register_scorer
def mean_squared_error(predictions: Sequence[Any], targets: Sequence[Any]) -> float:
    """Mean squared error for numeric predictions.

    Parameters
    ----------
    predictions
        Predicted numeric values or text containing a numeric value.
    targets
        Target numeric values or text containing a numeric value.

    Returns
    -------
    float
        Mean squared error in squared target units.
    """
    pred_vals, target_vals = _paired_numeric_values(predictions, targets)
    return sum(
        (p - t) ** 2 for p, t in zip(pred_vals, target_vals, strict=True)
    ) / float(len(pred_vals))


@register_scorer
def root_mean_squared_error(
    predictions: Sequence[Any], targets: Sequence[Any]
) -> float:
    """Root mean squared error for numeric predictions.

    Parameters
    ----------
    predictions
        Predicted numeric values or text containing a numeric value.
    targets
        Target numeric values or text containing a numeric value.

    Returns
    -------
    float
        Root mean squared error in the target units.
    """
    return math.sqrt(mean_squared_error(predictions, targets))
