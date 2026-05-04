"""Extraction prompt templates for the judge-as-structured-extractor path.

Each template pair (system + user Jinja2 body) teaches the judge model to
convert a raw, free-form model response into a structured prediction that can
be scored by the normal beans-next metrics pipeline.

Three task-specific templates are provided:

* ``extraction_classification_v1`` — single label from a provided vocabulary.
* ``extraction_detection_v1`` — comma-separated labels from a vocabulary.
* ``extraction_captioning_v1`` — cleaned single-sentence audio description.

Template variables available in user-message Jinja2 bodies:

* ``reference_text`` — ground truth labels joined as a string (empty for
  captioning tasks where no label vocabulary applies).
* ``candidate_text`` — raw model response to structure.
* ``sample_id`` — sample identifier (informational).
* ``task_id`` — task identifier (informational).
"""

from __future__ import annotations

__all__ = [
    "EXTRACTION_CAPTIONING_V1_ID",
    "EXTRACTION_CAPTIONING_V1_SYSTEM",
    "EXTRACTION_CAPTIONING_V1_TEMPLATE",
    "EXTRACTION_CLASSIFICATION_V1_ID",
    "EXTRACTION_CLASSIFICATION_V1_SYSTEM",
    "EXTRACTION_CLASSIFICATION_V1_TEMPLATE",
    "EXTRACTION_DETECTION_V1_ID",
    "EXTRACTION_DETECTION_V1_SYSTEM",
    "EXTRACTION_DETECTION_V1_TEMPLATE",
    "select_extraction_template",
]

# ---------------------------------------------------------------------------
# Classification — single label
# ---------------------------------------------------------------------------

EXTRACTION_CLASSIFICATION_V1_ID = "extraction_classification_v1"

EXTRACTION_CLASSIFICATION_V1_SYSTEM = (
    "You are a structured data extraction assistant. "
    "Given a raw audio model response below, extract the single predicted class label. "
    "Return only the exact label name from the provided vocabulary list — "
    "no punctuation, no explanation, nothing else."
)

EXTRACTION_CLASSIFICATION_V1_TEMPLATE = (
    "Valid labels:\n"
    "{{ reference_text }}\n"
    "\n"
    "Model response to structure:\n"
    "{{ candidate_text }}\n"
    "\n"
    "Return the single most likely label from the list above:"
)

# ---------------------------------------------------------------------------
# Detection — comma-separated labels
# ---------------------------------------------------------------------------

EXTRACTION_DETECTION_V1_ID = "extraction_detection_v1"

EXTRACTION_DETECTION_V1_SYSTEM = (
    "You are a structured data extraction assistant. "
    "Given a raw audio model response below, extract all detected sound event labels. "
    "Return only a comma-separated list of label names from the provided vocabulary — "
    "no explanation, no extra text."
)

EXTRACTION_DETECTION_V1_TEMPLATE = (
    "Valid labels:\n"
    "{{ reference_text }}\n"
    "\n"
    "Model response to structure:\n"
    "{{ candidate_text }}\n"
    "\n"
    "Return all detected labels from the list above (comma-separated):"
)

# ---------------------------------------------------------------------------
# Captioning — clean single-sentence description
# ---------------------------------------------------------------------------

EXTRACTION_CAPTIONING_V1_ID = "extraction_captioning_v1"

EXTRACTION_CAPTIONING_V1_SYSTEM = (
    "You are a structured data extraction assistant. "
    "Given a raw audio model response below, extract and clean the audio description. "
    "Return only a single concise sentence describing the audio — "
    "no hedging, no meta-commentary, no extra text."
)

EXTRACTION_CAPTIONING_V1_TEMPLATE = (
    "Model response to structure:\n"
    "{{ candidate_text }}\n"
    "\n"
    "Cleaned audio description:"
)


def select_extraction_template(task_type: str | None) -> tuple[str, str]:
    """Return ``(template_id, system_prompt)`` for the given task type.

    Parameters
    ----------
    task_type
        Task type string (e.g. ``"classification"``, ``"detection"``,
        ``"captioning"``). When ``None`` or unrecognised, defaults to the
        classification template.

    Returns
    -------
    tuple[str, str]
        ``(template_id, system_prompt)`` to pass to
        :class:`~beans_next.judges.predict_v1_extractor.PredictV1Extractor`.
    """
    task_s = (task_type or "").lower()
    if "caption" in task_s:
        return EXTRACTION_CAPTIONING_V1_ID, EXTRACTION_CAPTIONING_V1_SYSTEM
    if "detection" in task_s:
        return EXTRACTION_DETECTION_V1_ID, EXTRACTION_DETECTION_V1_SYSTEM
    return EXTRACTION_CLASSIFICATION_V1_ID, EXTRACTION_CLASSIFICATION_V1_SYSTEM
