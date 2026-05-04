"""LLM-as-judge via the ``predictions_v1`` HTTP predict endpoint (text-only).

Sends text-only chat messages to a model served on the same ``predictions_v1``
wire format as benchmark models (no audio inputs). Parses YES/NO responses to
binary scores compatible with ``JudgeScoresV1ResponseItem``.

Designed for ``cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`` (Gemma 4 26B A4B MoE)
but compatible with any model that accepts ``predictions_v1`` requests and
produces YES/NO text output.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment

from beans_next.api.http_schemas import (
    HttpChatMessage,
    HttpGenerationConfig,
    PredictionsV1Request,
    PredictionsV1RequestItem,
)
from beans_next.api.types import DatasetExample
from beans_next.judges.base import JudgeError, get_judge_template
from beans_next.judges.bioacoustic_classification_v1 import (
    BIOACOUSTIC_CLASSIFICATION_V1_ID,
    BIOACOUSTIC_CLASSIFICATION_V1_SYSTEM,
)
from beans_next.judges.http_schemas import JudgeScoresV1ResponseItem
from beans_next.models.http import HttpClient

__all__ = ["PredictV1Judge"]

_logger = logging.getLogger(__name__)


def _reference_text(example: DatasetExample) -> str:
    labels = example.labels
    if isinstance(labels, str):
        return labels
    if isinstance(labels, list):
        return ", ".join(str(x) for x in labels if isinstance(x, str) and x.strip())
    return ""


def _parse_yes_no(text: str) -> tuple[float | None, str | None]:
    """Parse a YES/NO judge response to ``(score, error)``.

    Parameters
    ----------
    text
        Raw model output text.

    Returns
    -------
    tuple[float | None, str | None]
        ``(1.0, None)`` for YES, ``(0.0, None)`` for NO,
        ``(None, error_message)`` when the response is unrecognised.
    """
    clean = text.strip().upper()
    if clean.startswith("YES"):
        return 1.0, None
    if clean.startswith("NO"):
        return 0.0, None
    return None, f"Unexpected judge response: {text[:120]!r}"


class PredictV1Judge:
    """LLM-as-judge using the ``predictions_v1`` HTTP predict endpoint.

    Unlike :class:`~beans_next.judges.scorer.JudgeScorer`, which sends batches
    to a dedicated ``judge_scores_v1`` endpoint, this class sends text-only chat
    messages to a model served on the standard ``predictions_v1`` wire format.

    The judge model receives:

    * A ``system`` message instructing it to reply YES or NO.
    * A ``user`` message rendered from a Jinja2 template containing the ground
      truth labels and the model output to evaluate.

    Responses are parsed to binary scores (1.0 for YES, 0.0 for NO) and
    returned as :class:`~beans_next.judges.http_schemas.JudgeScoresV1ResponseItem`
    objects so downstream code can treat both judge implementations uniformly.

    Parameters
    ----------
    judge_url
        Full URL for ``POST /predict`` on the judge model endpoint (for example
        ``http://localhost:8010/predict``).
    template_id
        Registered Jinja2 template name used to build the ``user`` message
        content.  Defaults to the built-in
        ``bioacoustic_classification_v1`` template.
    system_prompt
        ``system`` message sent to the judge model.  Defaults to the standard
        YES/NO evaluator prompt from the built-in template module.
    max_tokens
        Maximum tokens the judge model may generate (default 16; YES/NO only).
    timeout
        Per-request socket timeout in seconds.
    max_attempts
        HTTP retry budget for transient failures.

    Raises
    ------
    JudgeError
        If ``examples`` and ``candidate_texts`` lengths differ, or if template
        rendering fails for any sample.
    """

    def __init__(
        self,
        judge_url: str,
        *,
        template_id: str = BIOACOUSTIC_CLASSIFICATION_V1_ID,
        system_prompt: str | None = None,
        max_tokens: int = 16,
        timeout: float = 120.0,
        max_attempts: int = 3,
    ) -> None:
        self._judge_url = judge_url
        self._template_id = template_id
        self._system_prompt = system_prompt or BIOACOUSTIC_CLASSIFICATION_V1_SYSTEM
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._jinja = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)

    @property
    def judge_url(self) -> str:
        """URL used for judge ``POST /predict`` requests."""
        return self._judge_url

    @property
    def template_id(self) -> str:
        """Registered template name used to build the ``user`` message content."""
        return self._template_id

    def _render_user_content(self, example: DatasetExample, candidate_text: str) -> str:
        template_src = get_judge_template(self._template_id)
        try:
            tpl = self._jinja.from_string(template_src)
            return tpl.render(
                sample_id=example.sample_id,
                task_id=example.task_id,
                reference_text=_reference_text(example),
                candidate_text=candidate_text,
            )
        except TemplateError as exc:
            msg = (
                f"Judge template render failed for sample_id={example.sample_id!r}: "
                f"{exc}"
            )
            raise JudgeError(msg) from exc

    def score_batch(
        self,
        examples: Sequence[DatasetExample],
        candidate_texts: Sequence[str],
    ) -> list[JudgeScoresV1ResponseItem]:
        """Send judge requests and return parsed YES/NO scores.

        Parameters
        ----------
        examples
            One dataset row per item.
        candidate_texts
            Raw model outputs aligned with ``examples`` (not post-processed).

        Returns
        -------
        list of JudgeScoresV1ResponseItem
            Same order as ``examples``.  Items where the model response cannot
            be parsed carry ``score=None`` and a non-null ``error`` string.

        Raises
        ------
        JudgeError
            If ``examples`` and ``candidate_texts`` lengths differ, or if
            template rendering fails.
        """
        if len(examples) != len(candidate_texts):
            msg = (
                "`examples` and `candidate_texts` must have the same length "
                f"({len(examples)} vs {len(candidate_texts)})."
            )
            raise JudgeError(msg)

        if not examples:
            return []

        request_items: list[PredictionsV1RequestItem] = []
        for ex, cand in zip(examples, candidate_texts, strict=True):
            user_content = self._render_user_content(ex, cand)
            messages = [
                HttpChatMessage(role="system", content=self._system_prompt),
                HttpChatMessage(role="user", content=user_content),
            ]
            request_items.append(
                PredictionsV1RequestItem(
                    sample_id=ex.sample_id,
                    messages=messages,
                    audio_inputs=[],
                    generation_config=HttpGenerationConfig(
                        max_tokens=self._max_tokens,
                        temperature=0.0,
                    ),
                )
            )

        request = PredictionsV1Request(requests=request_items)
        with HttpClient(
            self._judge_url,
            timeout=self._timeout,
            max_attempts=self._max_attempts,
            probe_on_init=False,
        ) as client:
            response = client.generate(request)

        by_id = {r.sample_id: r for r in response.responses}
        results: list[JudgeScoresV1ResponseItem] = []
        for ex in examples:
            resp_item = by_id.get(ex.sample_id)
            if resp_item is None:
                results.append(
                    JudgeScoresV1ResponseItem(
                        sample_id=ex.sample_id,
                        score=None,
                        error=f"No response for sample_id={ex.sample_id!r}",
                    )
                )
                continue
            if resp_item.error:
                results.append(
                    JudgeScoresV1ResponseItem(
                        sample_id=ex.sample_id,
                        score=None,
                        error=resp_item.error,
                    )
                )
                continue
            raw_text = resp_item.predictions[0] if resp_item.predictions else ""
            score, err = _parse_yes_no(raw_text)
            results.append(
                JudgeScoresV1ResponseItem(
                    sample_id=ex.sample_id,
                    score=score,
                    rationale=raw_text if not err else None,
                    error=err,
                )
            )
        return results
