"""Thread-safe SQLite key/value store for JSON string payloads."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class SqliteJsonStore:
    """Append-only style string store with ``get`` / ``put`` on a single table.

    Parameters
    ----------
    path
        SQLite database file path (parent directories should exist).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
            isolation_level=None,
        )
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
            )

    def get(self, key: str) -> str | None:
        """Return the stored JSON string for ``key``, or ``None`` when missing.

        Parameters
        ----------
        key
            Primary key string.

        Returns
        -------
        str or None
            Stored value or ``None``.
        """
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        """Insert or replace ``key`` with ``value``.

        Parameters
        ----------
        key
            Primary key string.
        value
            JSON text to persist.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                (key, value),
            )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    @property
    def path(self) -> Path:
        """Filesystem path of the backing SQLite file."""
        return self._path


__all__ = ["SqliteJsonStore"]
