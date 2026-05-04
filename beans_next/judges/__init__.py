"""HTTP-only judge helpers (separate wire schema from ``predictions_v1``)."""

from __future__ import annotations

from beans_next.judges.base import (
    JudgeError,
    JudgeOutput,
    get_judge_template,
    list_judge_templates,
    register_judge_template,
)
from beans_next.judges.bioacoustic_classification_v1 import (
    BIOACOUSTIC_CLASSIFICATION_V1_ID,
    BIOACOUSTIC_CLASSIFICATION_V1_SYSTEM,
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
from beans_next.judges.client import JudgeHttpError, post_judge_scores
from beans_next.judges.extraction_templates import (
    EXTRACTION_CAPTIONING_V1_ID,
    EXTRACTION_CAPTIONING_V1_SYSTEM,
    EXTRACTION_CAPTIONING_V1_TEMPLATE,
    EXTRACTION_CLASSIFICATION_V1_ID,
    EXTRACTION_CLASSIFICATION_V1_SYSTEM,
    EXTRACTION_CLASSIFICATION_V1_TEMPLATE,
    EXTRACTION_DETECTION_V1_ID,
    EXTRACTION_DETECTION_V1_SYSTEM,
    EXTRACTION_DETECTION_V1_TEMPLATE,
    select_extraction_template,
)
from beans_next.judges.http_schemas import (
    JUDGE_SCORES_V1,
    JudgeScoresV1Request,
    JudgeScoresV1RequestItem,
    JudgeScoresV1Response,
    JudgeScoresV1ResponseItem,
)
from beans_next.judges.predict_v1_extractor import PredictV1Extractor
from beans_next.judges.predict_v1_judge import PredictV1Judge
from beans_next.judges.scorer import JudgeScorer

__all__ = [
    "BIOACOUSTIC_CLASSIFICATION_V1_ID",
    "BIOACOUSTIC_CLASSIFICATION_V1_SYSTEM",
    "BIOACOUSTIC_CLASSIFICATION_V1_TEMPLATE",
    "BIOACOUSTIC_COUNTING_V1_ID",
    "BIOACOUSTIC_COUNTING_V1_TEMPLATE",
    "BIOACOUSTIC_OPEN_QA_V1_ID",
    "BIOACOUSTIC_OPEN_QA_V1_TEMPLATE",
    "EXTRACTION_CAPTIONING_V1_ID",
    "EXTRACTION_CAPTIONING_V1_SYSTEM",
    "EXTRACTION_CAPTIONING_V1_TEMPLATE",
    "EXTRACTION_CLASSIFICATION_V1_ID",
    "EXTRACTION_CLASSIFICATION_V1_SYSTEM",
    "EXTRACTION_CLASSIFICATION_V1_TEMPLATE",
    "EXTRACTION_DETECTION_V1_ID",
    "EXTRACTION_DETECTION_V1_SYSTEM",
    "EXTRACTION_DETECTION_V1_TEMPLATE",
    "JUDGE_SCORES_V1",
    "JudgeError",
    "JudgeHttpError",
    "JudgeOutput",
    "JudgeScoresV1Request",
    "JudgeScoresV1RequestItem",
    "JudgeScoresV1Response",
    "JudgeScoresV1ResponseItem",
    "JudgeScorer",
    "PredictV1Extractor",
    "PredictV1Judge",
    "get_judge_template",
    "list_judge_templates",
    "post_judge_scores",
    "register_judge_template",
    "select_extraction_template",
]
