"""Built-in judge prompt template for bioacoustic counting tasks.

This rubric targets tasks where the desired output is a mapping of
**scientific species names** to **integer vocalization counts**.

It is designed to be robust to variable formatting (tables, bullet lists,
sentences) and to explicitly penalize refusals and hallucinated species.
"""

from __future__ import annotations

__all__ = ["BIOACOUSTIC_COUNTING_V1_ID", "BIOACOUSTIC_COUNTING_V1_TEMPLATE"]

BIOACOUSTIC_COUNTING_V1_ID = "bioacoustic_counting_v1"

BIOACOUSTIC_COUNTING_V1_TEMPLATE = (
    "You are a strict evaluator for bioacoustic counting tasks.\n"
    "\n"
    "Task identifier: {{ task_id }}\n"
    "Sample identifier: {{ sample_id }}\n"
    "\n"
    "Ground-truth reference (species -> vocalization count):\n"
    "{{ reference_text }}\n"
    "\n"
    "Model candidate output:\n"
    "{{ candidate_text }}\n"
    "\n"
    "Goal:\n"
    "Score the candidate in [0.0, 1.0] based on how well it reports the correct\n"
    "number of vocalizations per species present in the reference.\n"
    "\n"
    "Evaluation rules:\n"
    "- Treat the reference as the source of truth. The candidate should match it.\n"
    "- Accept flexible formatting (JSON, YAML, bullet list, table, prose), as long\n"
    "  as species names and counts are unambiguous.\n"
    "- Prefer scientific names (binomials), but accept common names when they\n"
    "  unambiguously identify a species present in the reference.\n"
    "- Count values must be non-negative integers.\n"
    "\n"
    "Scoring guidance:\n"
    "- Start from 1.0.\n"
    "- Deduct for each incorrect or missing species, and for each incorrect count.\n"
    "- Heavily penalize hallucinated species not present in the reference.\n"
    "- If the candidate refuses, claims inability, or provides no usable "
    "species->count\n"
    "  information, score 0.0.\n"
    "- If the candidate is partially correct, return an intermediate score reflecting\n"
    "  the fraction of correct species and correct counts.\n"
    "- Ignore extra narrative text if the species->count content is otherwise "
    "correct.\n"
    "\n"
    "Your downstream judge service must map this rubric to a single scalar score in\n"
    "[0.0, 1.0] returned in the HTTP response (see ``judge_scores_v1`` in\n"
    "``beans_next.judges.http_schemas``).\n"
).strip()
