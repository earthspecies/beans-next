"""Judge-as-extractor via the ``predictions_v1`` HTTP predict endpoint.

Unlike :class:`~beans_next.judges.predict_v1_judge.PredictV1Judge` which asks
the judge model to output YES/NO, :class:`PredictV1Extractor` asks it to
produce a *structured prediction* from a raw, free-form model response.

The two-step evaluation flow this enables::

    Raw model output (verbose / noisy)
        → PredictV1Extractor (judge generates structured prediction)
        → structured prediction (e.g. "cat" or "dog, rain")
        → normal post-process + scoring pipeline

This path is an alternative to the regex / fuzzy-matching post-process
pipeline for models whose outputs do not conform to label format expectations.
Task-specific templates are selected automatically from ``task_type``; see
:mod:`beans_next.judges.extraction_templates` for template content.
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
from beans_next.judges.extraction_templates import select_extraction_template
from beans_next.models.http import HttpClient

__all__ = ["PredictV1Extractor"]

_logger = logging.getLogger(__name__)


def _reference_text(example: DatasetExample) -> str:
    labels = example.labels
    if isinstance(labels, str):
        return labels
    if isinstance(labels, list):
        return ", ".join(str(x) for x in labels if isinstance(x, str) and x.strip())
    return ""


class PredictV1Extractor:
    """Judge-as-extractor: converts raw model outputs to structured predictions.

    Sends a two-turn chat to a judge model served on the ``predictions_v1``
    HTTP predict API.  The system message describes the extraction task; the
    user message contains the ground truth label vocabulary (where applicable)
    and the raw model output to structure.

    The judge responds with a structured prediction — a single label name for
    classification tasks, a comma-separated label list for detection tasks, or
    a clean one-sentence description for captioning tasks.  The returned strings
    are ready to feed into the normal beans-next post-process and scoring pipeline.

    Designed for ``cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`` (Gemma 4 26B A4B
    MoE) but works with any ``predictions_v1``-compatible text-only endpoint.

    Parameters
    ----------
    judge_url
        Full URL for ``POST /predict`` on the judge model endpoint.
    task_type
        Task type string used to select the extraction template automatically
        (e.g. ``"classification"``, ``"detection"``, ``"captioning"``).
        When ``None``, the classification template is used as default.
    template_id
        Override the auto-selected template by name.  When provided, takes
        precedence over ``task_type``-based selection.
    system_prompt
        Override the auto-selected system message.  When provided, takes
        precedence over the template-derived system message.
    max_tokens
        Maximum tokens the judge may generate (default 128; enough for a label
        list or short caption).
    timeout
        Per-request socket timeout in seconds.
    max_attempts
        HTTP retry budget for transient failures.

    Raises
    ------
    JudgeError
        If ``examples`` and ``raw_texts`` lengths differ, or if template
        rendering fails for any sample.
    """

    def __init__(
        self,
        judge_url: str,
        *,
        task_type: str | None = None,
        template_id: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        timeout: float = 120.0,
        max_attempts: int = 3,
    ) -> None:
        self._judge_url = judge_url
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._jinja = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)

        auto_template_id, auto_system = select_extraction_template(task_type)
        self._template_id = (
            template_id if template_id is not None else auto_template_id
        )
        self._system_prompt = (
            system_prompt if system_prompt is not None else auto_system
        )

    @property
    def judge_url(self) -> str:
        """URL used for extractor ``POST /predict`` requests."""
        return self._judge_url

    @property
    def template_id(self) -> str:
        """Registered template name used to build the ``user`` message content."""
        return self._template_id

    def _render_user_content(self, example: DatasetExample, raw_text: str) -> str:
        template_src = get_judge_template(self._template_id)
        try:
            tpl = self._jinja.from_string(template_src)
            return tpl.render(
                sample_id=example.sample_id,
                task_id=example.task_id,
                reference_text=_reference_text(example),
                candidate_text=raw_text,
            )
        except TemplateError as exc:
            msg = (
                f"Extraction template render failed for "
                f"sample_id={example.sample_id!r}: {exc}"
            )
            raise JudgeError(msg) from exc

    def extract_batch(
        self,
        examples: Sequence[DatasetExample],
        raw_texts: Sequence[str],
    ) -> list[str]:
        """Extract structured predictions from raw model outputs.

        Sends one ``predictions_v1`` request per batch to the judge endpoint.
        The judge model returns a structured string for each sample:

        * Classification: a single label name.
        * Detection: a comma-separated list of label names.
        * Captioning: a cleaned one-sentence description.

        Empty string is returned for any sample whose judge response is empty.

        Parameters
        ----------
        examples
            One :class:`~beans_next.api.types.DatasetExample` per item,
            providing ``labels`` (ground truth vocabulary) and identifiers.
        raw_texts
            Raw model outputs aligned with ``examples`` (not post-processed).

        Returns
        -------
        list of str
            Structured extracted predictions, one per example, in input order.
            Each string has leading/trailing whitespace stripped.

        Raises
        ------
        JudgeError
            If ``examples`` and ``raw_texts`` lengths differ, or if template
            rendering fails.
        """
        if len(examples) != len(raw_texts):
            msg = (
                "`examples` and `raw_texts` must have the same length "
                f"({len(examples)} vs {len(raw_texts)})."
            )
            raise JudgeError(msg)

        if not examples:
            return []

        request_items: list[PredictionsV1RequestItem] = []
        for ex, raw in zip(examples, raw_texts, strict=True):
            user_content = self._render_user_content(ex, raw)
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
        results: list[str] = []
        for ex in examples:
            resp_item = by_id.get(ex.sample_id)
            if resp_item is None or resp_item.error:
                err = (
                    resp_item.error
                    if resp_item
                    else f"missing sample_id={ex.sample_id!r}"
                )
                _logger.warning(
                    "Extractor error for sample_id=%r: %s", ex.sample_id, err
                )
                results.append("")
                continue
            raw_out = resp_item.predictions[0] if resp_item.predictions else ""
            results.append(raw_out.strip())

        return results
