"""Minimal HTTP client for ``judge_scores_v1`` (stdlib only)."""

from __future__ import annotations

import errno
import json
import random
import time
from collections.abc import Mapping
from http.client import RemoteDisconnected
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from beans_next.judges.http_schemas import (
    JUDGE_SCORES_V1,
    JudgeScoresV1Request,
    JudgeScoresV1Response,
)

_DEFAULT_MAX_ATTEMPTS: Final[int] = 3
_DEFAULT_BACKOFF_INITIAL: Final[float] = 1.0
_DEFAULT_BACKOFF_MAX: Final[float] = 30.0


class JudgeHttpError(RuntimeError):
    """Raised for judge HTTP contract violations or exhausted retries."""


def _is_transient_http_status(code: int, *, retry_on_429: bool) -> bool:
    if code >= 500:
        return True
    if code == 408:
        return True
    return code == 429 and retry_on_429


def _is_transient_url_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, RemoteDisconnected):
        return True
    if isinstance(exc, OSError):
        return exc.errno in (
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.EPIPE,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
        )
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            return True
        if isinstance(reason, OSError):
            return _is_transient_url_error(reason)
        if isinstance(reason, BaseException):
            return _is_transient_url_error(reason)
    return False


def _merge_headers(
    base: dict[str, str],
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    if not extra:
        return dict(base)
    merged = dict(base)
    merged.update(dict(extra))
    return merged


def _read_json(body: bytes, *, context: str) -> object:
    if not body.strip():
        msg = f"{context}: empty response body"
        raise JudgeHttpError(msg)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{context}: invalid JSON ({exc})"
        raise JudgeHttpError(msg) from exc


def _validate_response_ids(
    request_ids: list[str],
    response: JudgeScoresV1Response,
) -> JudgeScoresV1Response:
    req_set = set(request_ids)
    by_id = {item.sample_id: item for item in response.items}
    if len(by_id) != len(response.items):
        msg = "Judge response contains duplicate `sample_id` values"
        raise JudgeHttpError(msg)
    if set(by_id) != req_set:
        missing = sorted(req_set - set(by_id))
        extra = sorted(set(by_id) - req_set)
        msg = (
            "`sample_id` mismatch between judge request and response: "
            f"missing={missing!r} extra={extra!r}"
        )
        raise JudgeHttpError(msg)
    ordered = [by_id[sid] for sid in request_ids]
    return JudgeScoresV1Response(schema_version=JUDGE_SCORES_V1, items=ordered)


def post_judge_scores(
    judge_url: str,
    request: JudgeScoresV1Request,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 120.0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_initial: float = _DEFAULT_BACKOFF_INITIAL,
    backoff_max: float = _DEFAULT_BACKOFF_MAX,
    retry_on_429: bool = True,
) -> JudgeScoresV1Response:
    """POST a ``judge_scores_v1`` batch to ``judge_url`` and validate the response.

    Parameters
    ----------
    judge_url
        Full URL for the judge ``POST`` endpoint.
    request
        Outbound payload.
    headers
        Optional HTTP headers (for example auth).
    timeout
        Socket timeout in seconds per HTTP attempt.
    max_attempts
        Maximum attempts including the initial POST.
    backoff_initial
        Lower bound for exponential backoff before the first retry (seconds).
    backoff_max
        Upper bound for each sleep (seconds).
    retry_on_429
        Whether HTTP 429 is treated as transient.

    Returns
    -------
    JudgeScoresV1Response
        Parsed response with ``items`` ordered to match ``request.items``.

    Raises
    ------
    JudgeHttpError
        On schema mismatch, ``sample_id`` mismatch, exhausted retries, or
        non-transient HTTP failures.
    """
    request_ids = [it.sample_id for it in request.items]
    if len(request_ids) != len(set(request_ids)):
        msg = "Duplicate `sample_id` values in judge batch"
        raise JudgeHttpError(msg)

    hdrs = _merge_headers({"Accept": "application/json"}, headers)
    hdrs = _merge_headers(hdrs, {"Content-Type": "application/json"})
    body_bytes = json.dumps(request.model_dump(mode="json")).encode("utf-8")
    max_tries = max(1, int(max_attempts))

    def sleep_backoff(attempt_index: int) -> None:
        base = min(backoff_max, backoff_initial * (2**attempt_index))
        jitter = random.uniform(0.0, 0.25 * base)
        time.sleep(min(backoff_max, base + jitter))

    last_exc: BaseException | None = None
    for attempt in range(max_tries):
        req = Request(judge_url, data=body_bytes, headers=hdrs, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200))
                raw = resp.read()
        except HTTPError as exc:
            raw = exc.read()
            status = int(exc.code)
            if _is_transient_http_status(status, retry_on_429=retry_on_429):
                if attempt + 1 == max_tries:
                    msg = (
                        f"POST judge: HTTP {status} persisted after "
                        f"{max_tries} attempt(s): {raw[:512]!r}"
                    )
                    raise JudgeHttpError(msg) from exc
                sleep_backoff(attempt)
                last_exc = exc
                continue
            msg = f"POST judge: HTTP {status}: {raw[:2048]!r}"
            raise JudgeHttpError(msg) from exc
        except (URLError, OSError, RemoteDisconnected, TimeoutError) as exc:
            if not _is_transient_url_error(exc):
                msg = f"POST judge: {exc!r}"
                raise JudgeHttpError(msg) from exc
            if attempt + 1 == max_tries:
                msg = (
                    f"POST judge: transient network failure persisted after "
                    f"{max_tries} attempt(s): {exc!r}"
                )
                raise JudgeHttpError(msg) from exc
            sleep_backoff(attempt)
            last_exc = exc
            continue

        if status != 200:
            msg = f"POST judge: expected HTTP 200, got {status}: {raw[:2048]!r}"
            raise JudgeHttpError(msg)

        payload = _read_json(raw, context="POST judge")
        if not isinstance(payload, dict):
            msg = "POST judge: top-level JSON must be an object"
            raise JudgeHttpError(msg)
        if payload.get("schema_version") != JUDGE_SCORES_V1:
            msg = (
                "POST judge: top-level `schema_version` must be "
                f"{JUDGE_SCORES_V1!r}, got {payload.get('schema_version')!r}"
            )
            raise JudgeHttpError(msg)
        try:
            parsed = JudgeScoresV1Response.model_validate(payload)
        except ValidationError as exc:
            msg = f"POST judge: response failed schema validation: {exc}"
            raise JudgeHttpError(msg) from exc
        if parsed.schema_version != JUDGE_SCORES_V1:
            msg = "POST judge: parsed `schema_version` is inconsistent"
            raise JudgeHttpError(msg)
        for row in parsed.items:
            if row.error:
                continue
            if row.score is None:
                msg = (
                    f"POST judge: item {row.sample_id!r} missing `score` "
                    "without `error`"
                )
                raise JudgeHttpError(msg)
            if not (0.0 <= float(row.score) <= 1.0):
                msg = (
                    f"POST judge: item {row.sample_id!r} has score out of "
                    f"[0, 1]: {row.score!r}"
                )
                raise JudgeHttpError(msg)
        return _validate_response_ids(request_ids, parsed)

    msg = f"POST judge: exhausted {max_tries} attempt(s): {last_exc!r}"
    raise JudgeHttpError(msg)


__all__ = ["JudgeHttpError", "post_judge_scores"]
