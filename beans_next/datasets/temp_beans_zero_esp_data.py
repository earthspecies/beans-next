"""Temporary compatibility shim for removed dataset backend.

BEANS-Next is a clean-room extraction and must not import or reference private
research code or packages. Earlier iterations included an opt-in dataset loader
path that relied on a private backend; that path has been removed.
"""

from __future__ import annotations

__all__ = ["register_temp_beans_zero"]


def register_temp_beans_zero() -> type:
    """Return a dataset class for a removed temporary backend.

    Raises
    ------
    ImportError
        Always raised. Use HuggingFace-backed dataset loaders instead.
    """
    raise ImportError(
        "This temporary dataset backend was removed. Use the HuggingFace dataset "
        "loader paths configured via registry YAML instead."
    )
