"""Wire schemas for the judge HTTP API (separate from ``predictions_v1``).

Benchmark **models** must use ``predictions_v1`` on ``POST /predict`` only.
Optional **judge** services used by :class:`~beans_next.judges.scorer.JudgeScorer`
may implement a lighter batch JSON contract documented here so judge endpoints
do not need to masquerade as full multimodal launchers.

Contract summary
----------------
**Endpoint:** caller-provided absolute URL (for example
``http://localhost:8010/judge``). The library performs a single ``POST`` with
``Content-Type: application/json``.

**Request body (object):**

- ``schema_version`` — literal ``\"judge_scores_v1\"``.
- ``items`` — non-empty list of objects, each with:

  - ``sample_id`` (string, unique within the batch)
  - ``rubric`` (string; rendered evaluation instructions, often from a Jinja
    template)
  - ``reference_text`` (string; ground-truth or reference answer text)
  - ``candidate_text`` (string; model output to score)

**Response body (object):**

- ``schema_version`` — literal ``\"judge_scores_v1\"``.
- ``items`` — list with **exactly** the same ``sample_id`` set as the request
  (order may differ). Each object may include:

  - ``sample_id`` (string)
  - ``score`` (number in ``[0.0, 1.0]`` when present and ``error`` is null)
  - ``rationale`` (optional string)
  - ``error`` (optional string; per-item failure without failing the HTTP call)

Servers should return HTTP ``200`` for well-formed batches; transport-level
errors follow the same retry policy as
:func:`~beans_next.judges.client.post_judge_scores`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JUDGE_SCORES_V1: Literal["judge_scores_v1"] = "judge_scores_v1"


class JudgeScoresV1RequestItem(BaseModel):
    """One row in a ``judge_scores_v1`` request batch."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    rubric: str = Field(
        ...,
        description="Rendered judge instructions (bioacoustics-only templates "
        "ship in ``beans_next.judges``).",
    )
    reference_text: str = Field(default="", description="Reference or gold text.")
    candidate_text: str = Field(default="", description="Model output to score.")


class JudgeScoresV1Request(BaseModel):
    """Top-level ``POST`` JSON body for a judge service."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["judge_scores_v1"] = JUDGE_SCORES_V1
    items: list[JudgeScoresV1RequestItem]


class JudgeScoresV1ResponseItem(BaseModel):
    """One row in a ``judge_scores_v1`` response batch."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    score: float | None = None
    rationale: str | None = None
    error: str | None = None


class JudgeScoresV1Response(BaseModel):
    """Top-level JSON body returned by a judge service."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["judge_scores_v1"] = JUDGE_SCORES_V1
    items: list[JudgeScoresV1ResponseItem]


__all__ = [
    "JUDGE_SCORES_V1",
    "JudgeScoresV1Request",
    "JudgeScoresV1RequestItem",
    "JudgeScoresV1Response",
    "JudgeScoresV1ResponseItem",
]
