"""Built-in bioacoustics-only judge prompt template (Jinja2 source).

This template is intended for **open-ended bioacoustic captioning or short
answer** items where deterministic metrics are insufficient. It does not
reference non-bioacoustic benchmarks (speech ASR, music, general audio QA).

The rendered string is sent as the ``rubric`` field on
:class:`~beans_next.judges.http_schemas.JudgeScoresV1RequestItem`; the remote
judge service interprets it together with ``reference_text`` and
``candidate_text``.
"""

from __future__ import annotations

__all__ = ["BIOACOUSTIC_OPEN_QA_V1_ID", "BIOACOUSTIC_OPEN_QA_V1_TEMPLATE"]

BIOACOUSTIC_OPEN_QA_V1_ID = "bioacoustic_open_qa_v1"

BIOACOUSTIC_OPEN_QA_V1_TEMPLATE = (
    "You are a strict evaluator for bioacoustic captioning and short open answers.\n"
    "\n"
    "Task identifier: {{ task_id }}\n"
    "Sample identifier: {{ sample_id }}\n"
    "\n"
    "Ground-truth reference (bioacoustics):\n"
    "{{ reference_text }}\n"
    "\n"
    "Model candidate output:\n"
    "{{ candidate_text }}\n"
    "\n"
    "Evaluation criteria (bioacoustics-only):\n"
    "- Reward correct species, call types, behaviors, habitat, temporal patterns, "
    "and other sound-ecology facts aligned with the reference.\n"
    "- Penalize contradictions, invented species or events not supported by the "
    "reference, and off-topic generic text.\n"
    "- Ignore stylistic differences when the bioacoustic facts match.\n"
    "- Do not credit answers about speech, music, or unrelated environmental audio "
    "genres unless the reference itself concerns them.\n"
    "\n"
    "Your downstream judge service must map this rubric to a single scalar score in "
    "[0.0, 1.0] returned in the HTTP response (see ``judge_scores_v1`` in "
    "``beans_next.judges.http_schemas``).\n"
).strip()
