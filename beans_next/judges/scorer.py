"""LLM-as-judge scoring over HTTP (``judge_scores_v1`` wire format)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment

from beans_next.api.types import DatasetExample
from beans_next.judges.base import JudgeError, get_judge_template
from beans_next.judges.client import post_judge_scores
from beans_next.judges.http_schemas import (
    JudgeScoresV1Request,
    JudgeScoresV1RequestItem,
    JudgeScoresV1ResponseItem,
)

__all__ = ["JudgeScorer"]


def _reference_text(example: DatasetExample) -> str:
    labels = example.labels
    if isinstance(labels, str):
        return labels
    if isinstance(labels, list):
        parts = [str(x) for x in labels if isinstance(x, str) and x.strip()]
        return ", ".join(parts)
    return ""


class JudgeScorer:
    """Batch judge calls using the ``judge_scores_v1`` HTTP contract.

    Model inference remains ``predictions_v1`` on the benchmark launcher.
    This class only talks to a **separate** judge HTTP endpoint documented in
    :mod:`beans_next.judges.http_schemas`.

    Parameters
    ----------
    judge_url
        Full URL for ``POST`` (for example ``http://127.0.0.1:8010/judge``).
    template_id
        Registered Jinja template name (default: built-in
        ``bioacoustic_open_qa_v1``).
    headers
        Optional HTTP headers for the judge endpoint.
    timeout
        Per-request socket timeout in seconds.
    max_attempts
        Retry budget for transient HTTP failures.

    Raises
    ------
    JudgeError
        If ``examples`` and ``candidate_texts`` lengths differ or rendering fails.
    """

    def __init__(
        self,
        judge_url: str,
        *,
        template_id: str = "bioacoustic_open_qa_v1",
        headers: Mapping[str, str] | None = None,
        timeout: float = 120.0,
        max_attempts: int = 3,
    ) -> None:
        self._judge_url = judge_url
        self._template_id = template_id
        self._headers = dict(headers) if headers else None
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._jinja = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)

    @property
    def judge_url(self) -> str:
        """URL used for judge ``POST`` requests."""
        return self._judge_url

    @property
    def template_id(self) -> str:
        """Registered template name used to build ``rubric`` text."""
        return self._template_id

    def _render_rubric(self, example: DatasetExample, candidate_text: str) -> str:
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

    def build_request(
        self,
        examples: Sequence[DatasetExample],
        candidate_texts: Sequence[str],
    ) -> JudgeScoresV1Request:
        """Build a ``judge_scores_v1`` request (render rubrics, no HTTP).

        Parameters
        ----------
        examples
            One dataset row per item.
        candidate_texts
            Model outputs aligned with ``examples``.

        Returns
        -------
        JudgeScoresV1Request
            Outbound payload.

        Raises
        ------
        JudgeError
            If lengths differ or template rendering fails.
        """
        if len(examples) != len(candidate_texts):
            msg = (
                "`examples` and `candidate_texts` must have the same length "
                f"({len(examples)} vs {len(candidate_texts)})."
            )
            raise JudgeError(msg)
        items: list[JudgeScoresV1RequestItem] = []
        for ex, cand in zip(examples, candidate_texts, strict=True):
            rubric = self._render_rubric(ex, cand)
            items.append(
                JudgeScoresV1RequestItem(
                    sample_id=ex.sample_id,
                    rubric=rubric,
                    reference_text=_reference_text(ex),
                    candidate_text=cand,
                ),
            )
        return JudgeScoresV1Request(items=items)

    def score_batch(
        self,
        examples: Sequence[DatasetExample],
        candidate_texts: Sequence[str],
    ) -> list[JudgeScoresV1ResponseItem]:
        """Render rubrics, ``POST`` ``judge_scores_v1``, return ordered response rows.

        Parameters
        ----------
        examples
            One dataset row per item.
        candidate_texts
            Model outputs aligned with ``examples``.

        Returns
        -------
        list of JudgeScoresV1ResponseItem
            Same order as ``examples``.

        Notes
        -----
        :meth:`build_request` may raise :exc:`JudgeError`. HTTP and response
        validation errors surface as :exc:`beans_next.judges.client.JudgeHttpError`
        from :func:`~beans_next.judges.client.post_judge_scores`.
        """
        req = self.build_request(examples, candidate_texts)
        if not req.items:
            return []
        resp = post_judge_scores(
            self._judge_url,
            req,
            headers=self._headers,
            timeout=self._timeout,
            max_attempts=self._max_attempts,
        )
        return list(resp.items)
