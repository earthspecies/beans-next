"""SPICE metric implementation (requires Java 8+ and Stanford CoreNLP JARs)."""

from __future__ import annotations

from beans_next.metrics._spice.spice import (
    Spice,
    SpiceUnavailableError,
    check_spice_available,
)

__all__ = ["Spice", "SpiceUnavailableError", "check_spice_available"]
