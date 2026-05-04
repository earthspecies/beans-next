"""Built-in judge prompt template for classification and detection tasks.

This template is used by :class:`~beans_next.judges.predict_v1_judge.PredictV1Judge`
when the judge model is served via the ``predictions_v1`` HTTP predict API.
The rendered string becomes the ``user`` message content in a two-turn chat;
the judge model is expected to reply YES or NO only.
"""

from __future__ import annotations

__all__ = [
    "BIOACOUSTIC_CLASSIFICATION_V1_ID",
    "BIOACOUSTIC_CLASSIFICATION_V1_TEMPLATE",
    "BIOACOUSTIC_CLASSIFICATION_V1_SYSTEM",
]

BIOACOUSTIC_CLASSIFICATION_V1_ID = "bioacoustic_classification_v1"

BIOACOUSTIC_CLASSIFICATION_V1_SYSTEM = (
    "You are a strict evaluator for audio classification and detection tasks. "
    "You will be given ground truth labels and a model prediction. "
    "Reply only with YES if the prediction correctly identifies at least one ground "
    "truth label, or NO if it does not. No explanation."
)

BIOACOUSTIC_CLASSIFICATION_V1_TEMPLATE = (
    "Ground truth labels: {{ reference_text }}\n"
    'Model output: "{{ candidate_text }}"\n'
    "\n"
    "Does the model output correctly identify at least one of the ground truth labels? "
    "Reply YES or NO only."
)
