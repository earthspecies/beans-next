"""Judge template registry and public result types."""

from __future__ import annotations

from beans_next.judges.bioacoustic_classification_v1 import (
    BIOACOUSTIC_CLASSIFICATION_V1_ID,
    BIOACOUSTIC_CLASSIFICATION_V1_TEMPLATE,
)
from beans_next.judges.bioacoustic_counting_v1 import (
    BIOACOUSTIC_COUNTING_V1_ID,
    BIOACOUSTIC_COUNTING_V1_TEMPLATE,
)
from beans_next.judges.bioacoustic_open_qa_v1 import (
    BIOACOUSTIC_OPEN_QA_V1_ID,
    BIOACOUSTIC_OPEN_QA_V1_TEMPLATE,
)
from beans_next.judges.extraction_templates import (
    EXTRACTION_CAPTIONING_V1_ID,
    EXTRACTION_CAPTIONING_V1_TEMPLATE,
    EXTRACTION_CLASSIFICATION_V1_ID,
    EXTRACTION_CLASSIFICATION_V1_TEMPLATE,
    EXTRACTION_DETECTION_V1_ID,
    EXTRACTION_DETECTION_V1_TEMPLATE,
)
from beans_next.judges.http_schemas import JudgeScoresV1ResponseItem

__all__ = [
    "JudgeError",
    "JudgeOutput",
    "get_judge_template",
    "list_judge_templates",
    "register_judge_template",
]

_TEMPLATES: dict[str, str] = {
    BIOACOUSTIC_CLASSIFICATION_V1_ID: BIOACOUSTIC_CLASSIFICATION_V1_TEMPLATE,
    BIOACOUSTIC_COUNTING_V1_ID: BIOACOUSTIC_COUNTING_V1_TEMPLATE,
    BIOACOUSTIC_OPEN_QA_V1_ID: BIOACOUSTIC_OPEN_QA_V1_TEMPLATE,
    EXTRACTION_CAPTIONING_V1_ID: EXTRACTION_CAPTIONING_V1_TEMPLATE,
    EXTRACTION_CLASSIFICATION_V1_ID: EXTRACTION_CLASSIFICATION_V1_TEMPLATE,
    EXTRACTION_DETECTION_V1_ID: EXTRACTION_DETECTION_V1_TEMPLATE,
}


class JudgeError(ValueError):
    """Raised for invalid judge configuration, rendering, or HTTP payloads."""


JudgeOutput = JudgeScoresV1ResponseItem


def register_judge_template(name: str, source: str) -> None:
    """Register a Jinja2 template string under ``name``.

    Parameters
    ----------
    name
        Identifier used by :class:`~beans_next.judges.scorer.JudgeScorer`.
    source
        Jinja2 template body.

    Raises
    ------
    JudgeError
        If ``name`` is empty or already registered.
    """
    if not name.strip():
        msg = "Template name must be non-empty."
        raise JudgeError(msg)
    if name in _TEMPLATES:
        msg = f"Judge template {name!r} is already registered."
        raise JudgeError(msg)
    _TEMPLATES[name] = source


def get_judge_template(name: str) -> str:
    """Return a registered Jinja2 template string.

    Parameters
    ----------
    name
        Registered template id.

    Returns
    -------
    str
        Template source.

    Raises
    ------
    LookupError
        If ``name`` is unknown.
    """
    try:
        return _TEMPLATES[name]
    except KeyError as exc:
        msg = f"Judge template {name!r} not found."
        raise LookupError(msg) from exc


def list_judge_templates() -> list[str]:
    """Return sorted registered template ids.

    Returns
    -------
    list of str
        Template names.
    """
    return sorted(_TEMPLATES.keys())
