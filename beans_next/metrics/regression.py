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


def _maybe_deci_hz(
    value: float,
    *,
    target_value: float | None,
    unit: str | None,
) -> float:
    if unit != "hz" or target_value is None:
        return value
    scaled = value * 0.1
    if target_value < 1000.0 and value >= 1000.0:
        if abs(scaled - target_value) < abs(value - target_value):
            return scaled
    return value


def extract_numeric_value(
    value: object,
    *,
    target_value: float | None = None,
    unit: str | None = None,
) -> float:
    """Extract one numeric value from model text or a target label.

    Parameters
    ----------
    value
        Numeric scalar, string, or single-item sequence containing text such as
        ``"12.5 dB"`` or ``"3140 Hz"``.
    target_value
        Optional target value used only for an F0-specific deci-Hz correction on
        range outputs like ``"2131-4440 Hz"`` when the target is hundreds of Hz.
    unit
        Optional target-unit hint (currently ``"hz"`` or ``"db"``). Predictions
        in ``kHz`` are converted to Hz when the target unit is ``"hz"``.

    Returns
    -------
    float
        Extracted numeric value.

    Raises
    ------
    MetricsError
        If no finite numeric value can be extracted.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise MetricsError("Cannot extract a numeric value from an empty sequence.")
        value = value[0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _coerce_float(value)
    if not isinstance(value, str):
        return _coerce_float(value)

    text = value.strip()
    if not text:
        raise MetricsError("Cannot extract a numeric value from empty text.")
    target_unit = (unit or "").lower() or None
    inferred_unit = (_unit_hint(text) or target_unit or "").lower() or None

    range_match = _RANGE_RE.search(text)
    if range_match is not None:
        a = _coerce_float(range_match.group("a"))
        b = _coerce_float(range_match.group("b"))
        range_unit = (
            range_match.group("unit") or inferred_unit or ""
        ).lower() or None
        midpoint = (a + b) / 2.0
        midpoint = _convert_to_target_unit(
            midpoint,
            source_unit=range_unit,
            target_unit=target_unit,
        )
        return _maybe_deci_hz(midpoint, target_value=target_value, unit=range_unit)

    matches = list(_NUMBER_RE.finditer(text))
    nums = [m.group("num") for m in matches]
    if not nums:
        raise MetricsError(f"No numeric value found in {value!r}.")
    last_match = matches[-1]
    parsed = _coerce_float(nums[-1])
    suffix = text[last_match.end() : last_match.end() + 12].lower().lstrip()
    source_unit = "khz" if suffix.startswith("khz") else inferred_unit
    parsed = _convert_to_target_unit(
        parsed,
        source_unit=source_unit,
        target_unit=target_unit,
    )
    return parsed


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
