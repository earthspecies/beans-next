"""HTTP inference client for the ``predictions_v1`` launcher contract."""

from __future__ import annotations

import errno
import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.client import RemoteDisconnected
from typing import Any, Final, Self
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pydantic import ValidationError

from beans_next.api.http_schemas import (
    PREDICTIONS_V1,
    PredictionsV1Request,
    PredictionsV1Response,
)

_DEFAULT_MAX_ATTEMPTS: Final[int] = 12
_DEFAULT_BACKOFF_INITIAL: Final[float] = 0.5
_DEFAULT_BACKOFF_MAX: Final[float] = 30.0
_DEFAULT_JITTER_FRACTION: Final[float] = 0.25
_DEFAULT_RETRYABLE_HTTP_STATUSES: Final[tuple[int, ...]] = (408, 429)
_DEFAULT_RETRYABLE_OS_ERRNOS: Final[tuple[int, ...]] = (
    errno.ECONNRESET,
    errno.ECONNREFUSED,
    errno.EPIPE,
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry policy for :class:`HttpClient`.

    The default values match the current hardcoded behavior:

    - 3 attempts
    - exponential backoff from 1s to 30s
    - uniform jitter in ``[0, 0.25 * backoff]``
    - retry on HTTP 5xx, 408, 429
    - retry on transient network errors (timeouts, resets, etc.)

    Parameters
    ----------
    max_attempts
        Maximum attempts per logical operation (initial try plus retries).
    backoff_initial
        Initial backoff lower bound in seconds before the first retry.
    backoff_max
        Maximum backoff in seconds (each sleep is capped at this value).
    jitter_fraction
        Jitter factor as a fraction of the computed backoff, applied as an
        additive uniform random term in ``[0, jitter_fraction * backoff]``.
    retry_on_5xx
        When ``True``, HTTP 5xx responses are treated as transient and retried.
    retry_http_statuses
        Additional HTTP statuses treated as transient (commonly 408 and 429).
    retry_on_network_errors
        When ``True``, transient network failures are retried.
    retryable_os_errnos
        When retrying network failures, the set of ``errno`` values treated as
        transient for :class:`OSError` cases.
    """

    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    backoff_initial: float = _DEFAULT_BACKOFF_INITIAL
    backoff_max: float = _DEFAULT_BACKOFF_MAX
    jitter_fraction: float = _DEFAULT_JITTER_FRACTION

    retry_on_5xx: bool = True
    retry_http_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset(_DEFAULT_RETRYABLE_HTTP_STATUSES)
    )

    retry_on_network_errors: bool = True
    retryable_os_errnos: frozenset[int] = field(
        default_factory=lambda: frozenset(_DEFAULT_RETRYABLE_OS_ERRNOS)
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("`retry_policy.max_attempts` must be >= 1")
        if self.backoff_initial < 0:
            raise ValueError("`retry_policy.backoff_initial` must be >= 0")
        if self.backoff_max < 0:
            raise ValueError("`retry_policy.backoff_max` must be >= 0")
        if self.backoff_max < self.backoff_initial:
            raise ValueError(
                "`retry_policy.backoff_max` must be >= `retry_policy.backoff_initial`"
            )
        if self.jitter_fraction < 0:
            raise ValueError("`retry_policy.jitter_fraction` must be >= 0")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Build a :class:`RetryPolicy` from a JSON/YAML-friendly mapping.

        Parameters
        ----------
        data
            Mapping with keys matching :class:`RetryPolicy` fields.

        Returns
        -------
        RetryPolicy
            Parsed retry policy.

        Raises
        ------
        ValueError
            If the mapping is invalid (unknown keys or invalid value types).
        """
        allowed = {
            "max_attempts",
            "backoff_initial",
            "backoff_max",
            "jitter_fraction",
            "retry_on_5xx",
            "retry_http_statuses",
            "retry_on_network_errors",
            "retryable_os_errnos",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown retry_policy key(s): {unknown!r}")

        def _as_int_set(v: object, *, key: str) -> frozenset[int]:
            if v is None:
                return frozenset()
            if isinstance(v, (list, tuple, set, frozenset)) and all(
                isinstance(x, int) for x in v
            ):
                return frozenset(int(x) for x in v)
            raise ValueError(f"`retry_policy.{key}` must be a list of ints")

        kwargs: dict[str, Any] = dict(data)
        if "retry_http_statuses" in kwargs:
            kwargs["retry_http_statuses"] = _as_int_set(
                kwargs["retry_http_statuses"], key="retry_http_statuses"
            )
        if "retryable_os_errnos" in kwargs:
            kwargs["retryable_os_errnos"] = _as_int_set(
                kwargs["retryable_os_errnos"], key="retryable_os_errnos"
            )
        return cls(**kwargs)


class HttpClientFatalError(RuntimeError):
    """Raised for contract violations and non-retriable HTTP failures.

    Examples include missing ``predictions_v1`` in ``/info``, ``sample_id``
    mismatches between request and response, and client-side HTTP 4xx codes
    other than those treated as transient (see :class:`HttpClient`).
    """


def _info_url(predict_url: str) -> str:
    return urljoin(predict_url, "/info")


def _health_url(predict_url: str) -> str:
    return urljoin(predict_url, "/health")


def _merge_headers(
    base: dict[str, str],
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    if not extra:
        return dict(base)
    merged = dict(base)
    merged.update(dict(extra))
    return merged


def _is_transient_http_status(code: int, *, policy: RetryPolicy) -> bool:
    if policy.retry_on_5xx and code >= 500:
        return True
    return code in policy.retry_http_statuses


def _is_transient_url_error(exc: BaseException, *, policy: RetryPolicy) -> bool:
    if not policy.retry_on_network_errors:
        return False
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, RemoteDisconnected):
        return True
    if isinstance(exc, OSError):
        return exc.errno in policy.retryable_os_errnos
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            return True
        if isinstance(reason, URLError):
            return False
        if isinstance(reason, OSError):
            return _is_transient_url_error(reason, policy=policy)
        if isinstance(reason, BaseException):
            return _is_transient_url_error(reason, policy=policy)
    return False


def _read_json_response(body: bytes, *, context: str) -> object:
    if not body.strip():
        msg = f"{context}: empty response body"
        raise HttpClientFatalError(msg)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{context}: invalid JSON ({exc})"
        raise HttpClientFatalError(msg) from exc


def _validate_info_advertises_predictions_v1(info: Mapping[str, Any]) -> None:
    versions = info.get("schema_versions")
    if not isinstance(versions, list) or not all(isinstance(v, str) for v in versions):
        msg = "/info: `schema_versions` must be a list of strings"
        raise HttpClientFatalError(msg)
    if PREDICTIONS_V1 not in versions:
        msg = f"/info: launcher must advertise `{PREDICTIONS_V1}` in `schema_versions`"
        raise HttpClientFatalError(msg)


def _validate_request_sample_ids_unique(request: PredictionsV1Request) -> list[str]:
    ids = [item.sample_id for item in request.requests]
    if len(ids) != len(set(ids)):
        msg = "Duplicate `sample_id` values in `PredictionsV1Request.requests`"
        raise HttpClientFatalError(msg)
    return ids


def _validate_response_sample_ids(
    request_ids: list[str],
    response: PredictionsV1Response,
) -> PredictionsV1Response:
    req_set = set(request_ids)
    response_by_id = {item.sample_id: item for item in response.responses}
    if len(response_by_id) != len(response.responses):
        msg = "Response contains duplicate `sample_id` values"
        raise HttpClientFatalError(msg)
    if set(response_by_id) != req_set:
        missing = sorted(req_set - set(response_by_id))
        extra = sorted(set(response_by_id) - req_set)
        msg = (
            "`sample_id` mismatch between request and response: "
            f"missing={missing!r} extra={extra!r}"
        )
        raise HttpClientFatalError(msg)
    ordered = [response_by_id[sid] for sid in request_ids]
    return PredictionsV1Response(schema_version=PREDICTIONS_V1, responses=ordered)


class HttpClient:
    """``predictions_v1`` HTTP client (stdlib ``urllib``; no extra HTTP deps).

    Parameters
    ----------
    predict_url
        Full URL for ``POST /predict`` (for example ``http://localhost:8000/predict``).
        ``GET /info`` and ``GET /health`` are resolved via :func:`urllib.parse.urljoin`
        against the launcher's host (paths ``/info`` and ``/health``).
    headers
        Optional extra headers merged into every request (for example auth).
    timeout
        Per-request socket timeout in seconds (applies to each HTTP call,
        including retries).
    max_attempts
        Maximum attempts per logical operation (initial try plus retries).
        Default ``3`` matches DESIGN §4.4 ("3 attempts").
    backoff_initial
        Initial backoff lower bound in seconds before the first retry.
    backoff_max
        Maximum backoff in seconds (each sleep is capped at this value).
    retry_on_429
        When ``True``, HTTP 429 is treated as transient and retried.
    retry_policy
        Optional :class:`RetryPolicy` (or mapping accepted by
        :meth:`RetryPolicy.from_mapping`) to fully control retry decisions and
        backoff. When provided, it takes precedence over the legacy
        ``max_attempts``/backoff/retry flags.
    probe_on_init
        When ``True``, :meth:`probe_info` runs during construction so the client
        fails fast if the launcher does not advertise ``predictions_v1``.

    Raises
    ------
    HttpClientFatalError
        If ``probe_on_init`` is ``True`` and ``/info`` is missing, invalid, or
        does not list ``predictions_v1``.
    """

    def __init__(
        self,
        predict_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 120.0,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        backoff_initial: float = _DEFAULT_BACKOFF_INITIAL,
        backoff_max: float = _DEFAULT_BACKOFF_MAX,
        retry_on_429: bool = True,
        retry_policy: RetryPolicy | Mapping[str, Any] | None = None,
        probe_on_init: bool = True,
    ) -> None:
        self._predict_url = predict_url
        self._headers = dict(headers) if headers else {}
        self._timeout = timeout
        if retry_policy is None:
            statuses = set(_DEFAULT_RETRYABLE_HTTP_STATUSES)
            if not retry_on_429:
                statuses.discard(429)
            self._retry_policy = RetryPolicy(
                max_attempts=max(1, int(max_attempts)),
                backoff_initial=backoff_initial,
                backoff_max=backoff_max,
                retry_http_statuses=frozenset(statuses),
            )
        elif isinstance(retry_policy, RetryPolicy):
            self._retry_policy = retry_policy
        else:
            self._retry_policy = RetryPolicy.from_mapping(retry_policy)
        self._server_info: dict[str, Any] | None = None
        if probe_on_init:
            self.probe_info()

    @property
    def predict_url(self) -> str:
        """URL used for ``POST /predict``."""
        return self._predict_url

    @property
    def server_info(self) -> dict[str, Any] | None:
        """Last successful ``GET /info`` payload, or ``None`` if not probed yet."""
        return self._server_info

    def close(self) -> None:
        """Release resources (noop for stdlib client; symmetry for context managers)."""

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _sleep_before_retry(self, attempt_index: int) -> None:
        # attempt_index is 0 after first failure, 1 after second, ...
        base = min(
            self._retry_policy.backoff_max,
            self._retry_policy.backoff_initial * (2**attempt_index),
        )
        jitter = random.uniform(0.0, self._retry_policy.jitter_fraction * base)
        time.sleep(min(self._retry_policy.backoff_max, base + jitter))

    def _build_urllib_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None,
    ) -> Request:
        hdrs = _merge_headers({"Accept": "application/json"}, self._headers)
        if json_body is not None:
            hdrs = _merge_headers(hdrs, {"Content-Type": "application/json"})
            data = json.dumps(json_body).encode("utf-8")
        else:
            data = None
        return Request(url, data=data, headers=hdrs, method=method)

    def _handle_http_error(
        self,
        exc: HTTPError,
        body: bytes,
        attempt: int,
        context: str,
    ) -> None:
        code = exc.code
        if _is_transient_http_status(code, policy=self._retry_policy):
            if attempt + 1 == self._retry_policy.max_attempts:
                msg = (
                    f"{context}: HTTP {code} persisted after "
                    f"{self._retry_policy.max_attempts} attempt(s): {body[:512]!r}"
                )
                raise HttpClientFatalError(msg) from exc
            self._sleep_before_retry(attempt)
            return
        msg = f"{context}: HTTP {code}: {body[:2048]!r}"
        raise HttpClientFatalError(msg) from exc

    def _handle_network_error(
        self,
        exc: BaseException,
        attempt: int,
        context: str,
    ) -> None:
        if not _is_transient_url_error(exc, policy=self._retry_policy):
            msg = f"{context}: {exc!r}"
            raise HttpClientFatalError(msg) from exc
        if attempt + 1 == self._retry_policy.max_attempts:
            msg = (
                f"{context}: transient network failure persisted after "
                f"{self._retry_policy.max_attempts} attempt(s): {exc!r}"
            )
            raise HttpClientFatalError(msg) from exc
        self._sleep_before_retry(attempt)

    def _request_bytes(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        context: str,
    ) -> tuple[int, bytes]:
        for attempt in range(self._retry_policy.max_attempts):
            req = self._build_urllib_request(method, url, json_body=json_body)
            try:
                with urlopen(req, timeout=self._timeout) as resp:
                    status = getattr(resp, "status", 200)
                    body = resp.read()
                    return int(status), body
            except HTTPError as exc:
                body = exc.read()
                self._handle_http_error(exc, body, attempt, context)
            except (URLError, OSError, RemoteDisconnected, TimeoutError) as exc:
                self._handle_network_error(exc, attempt, context)
        msg = f"{context}: exhausted {self._retry_policy.max_attempts} attempt(s)"
        raise HttpClientFatalError(msg)

    def health(self) -> None:
        """GET ``/health``; raises if the status is not HTTP 200.

        Raises
        ------
        HttpClientFatalError
            On non-200 responses or repeated transient failures.
        """
        url = _health_url(self._predict_url)
        status, body = self._request_bytes(
            "GET",
            url,
            json_body=None,
            context="GET /health",
        )
        if status != 200:
            msg = f"GET /health: expected HTTP 200, got {status}: {body[:512]!r}"
            raise HttpClientFatalError(msg)

    def probe_info(self) -> dict[str, Any]:
        """GET ``/info`` and validate ``predictions_v1`` support.

        Returns
        -------
        dict[str, Any]
            Parsed ``/info`` JSON object.

        Raises
        ------
        HttpClientFatalError
            If the payload is invalid or does not advertise ``predictions_v1``.
        """
        url = _info_url(self._predict_url)
        status, body = self._request_bytes(
            "GET",
            url,
            json_body=None,
            context="GET /info",
        )
        if status != 200:
            msg = f"GET /info: expected HTTP 200, got {status}: {body[:512]!r}"
            raise HttpClientFatalError(msg)
        payload = _read_json_response(body, context="GET /info")
        if not isinstance(payload, dict):
            msg = "GET /info: top-level JSON must be an object"
            raise HttpClientFatalError(msg)
        _validate_info_advertises_predictions_v1(payload)
        self._server_info = dict(payload)
        return self._server_info

    def generate(self, request: PredictionsV1Request) -> PredictionsV1Response:
        """POST ``/predict`` with a ``predictions_v1`` batch envelope.

        Per-sample failures use the wire-level ``error`` field; this method does
        not raise solely because an item's ``error`` is non-null.

        Parameters
        ----------
        request
            Outbound ``predictions_v1`` request body.

        Returns
        -------
        PredictionsV1Response
            Validated response with ``responses`` ordered to match ``request.requests``.

        Raises
        ------
        HttpClientFatalError
            On duplicate ``sample_id`` values in the request, top-level schema mismatch,
            ``sample_id`` set mismatches between request and response, or non-transient
            HTTP failures.
        """
        request_ids = _validate_request_sample_ids_unique(request)
        body = request.model_dump(mode="json")
        status, raw = self._request_bytes(
            "POST",
            self._predict_url,
            json_body=body,
            context="POST /predict",
        )
        if status != 200:
            msg = f"POST /predict: expected HTTP 200, got {status}: {raw[:2048]!r}"
            raise HttpClientFatalError(msg)
        payload = _read_json_response(raw, context="POST /predict")
        if not isinstance(payload, dict):
            msg = "POST /predict: top-level JSON must be an object"
            raise HttpClientFatalError(msg)
        if payload.get("schema_version") != PREDICTIONS_V1:
            msg = (
                "POST /predict: top-level `schema_version` must be "
                f"{PREDICTIONS_V1!r}, got {payload.get('schema_version')!r}"
            )
            raise HttpClientFatalError(msg)
        try:
            parsed = PredictionsV1Response.model_validate(payload)
        except ValidationError as exc:
            msg = f"POST /predict: response failed schema validation: {exc}"
            raise HttpClientFatalError(msg) from exc
        if parsed.schema_version != PREDICTIONS_V1:
            msg = (
                "POST /predict: parsed `schema_version` must be "
                f"{PREDICTIONS_V1!r}, got {parsed.schema_version!r}"
            )
            raise HttpClientFatalError(msg)
        return _validate_response_sample_ids(request_ids, parsed)


__all__ = ["HttpClient", "HttpClientFatalError", "RetryPolicy"]
