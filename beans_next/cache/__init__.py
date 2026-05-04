"""Optional benchmark caches (inference + scoring)."""

from beans_next.cache.sqlite_kv import SqliteJsonStore
from beans_next.cache.two_layer import (
    TwoLayerRunCache,
    inference_cache_key,
    scoring_cache_key,
)

__all__ = [
    "SqliteJsonStore",
    "TwoLayerRunCache",
    "inference_cache_key",
    "scoring_cache_key",
]
