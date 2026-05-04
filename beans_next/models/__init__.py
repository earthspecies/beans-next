"""Model adapters and protocols."""

from beans_next.models.base import InferenceModel
from beans_next.models.http import HttpClient, HttpClientFatalError

__all__ = ["HttpClient", "HttpClientFatalError", "InferenceModel"]
