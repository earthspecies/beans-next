"""Pre-run HTTP launcher probes (``/health`` and ``/info``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from beans_next.models.http import HttpClient

__all__ = ["ensure_launcher_ready", "probe_health"]


def probe_health(
    predict_url: str,
    *,
    timeout: float = 30.0,
    headers: Mapping[str, str] | None = None,
) -> None:
    """Send ``GET /health`` for the launcher behind ``predict_url``.

    Opens a short-lived :class:`~beans_next.models.http.HttpClient` without
    calling ``GET /info``.

    Parameters
    ----------
    predict_url
        Same URL you would pass to :class:`~beans_next.models.http.HttpClient`
        for ``POST /predict``.
    timeout
        Socket timeout in seconds for the probe.
    headers
        Optional headers merged into the probe request (for example auth).

    Notes
    -----
    Contract failures and non-retriable HTTP errors surface as
    :exc:`beans_next.models.http.HttpClientFatalError` from :meth:`HttpClient.health`.
    """
    client = HttpClient(
        predict_url,
        headers=headers,
        timeout=timeout,
        probe_on_init=False,
    )
    try:
        client.health()
    finally:
        client.close()


def ensure_launcher_ready(
    predict_url: str,
    *,
    timeout: float = 30.0,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Probe ``GET /health`` then ``GET /info`` for a ``predictions_v1`` launcher.

    Parameters
    ----------
    predict_url
        Same URL you would pass to :class:`~beans_next.models.http.HttpClient`
        for ``POST /predict``.
    timeout
        Socket timeout in seconds for each HTTP call.
    headers
        Optional headers merged into each request (for example auth).

    Returns
    -------
    dict[str, Any]
        Parsed ``/info`` JSON (includes ``schema_versions``).

    Notes
    -----
    Failures from ``/health`` or ``/info`` surface as
    :exc:`beans_next.models.http.HttpClientFatalError`.
    """
    client = HttpClient(
        predict_url,
        headers=headers,
        timeout=timeout,
        probe_on_init=False,
    )
    try:
        client.health()
        return client.probe_info()
    finally:
        client.close()
