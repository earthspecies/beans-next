"""Tests for NatureLM v1.0 prompt normalization in the serve launcher."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def _load_serve_module() -> object:
    serve_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "servers"
        / "naturelm-v1.0"
        / "serve.py"
    )
    # Force stub mode so importing the module doesn't require torch/GPU deps.
    os.environ.setdefault("NATURELM_V1_0_STUB", "1")
    spec = importlib.util.spec_from_file_location(
        "beans_next_naturelm_v1_0_serve", serve_path
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is present in sys.modules so dataclasses can resolve
    # postponed annotations during import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_prompt_normalization_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_serve_module()
    monkeypatch.delenv("NATURELM_V1_0_NORMALIZE_PROMPTS", raising=False)
    q = (
        "<Audio><AudioHere></Audio> The objective is to classify the sound into one of "
        "the following categories: dog, cat"
    )
    assert mod._normalize_query_for_v1_0(q) == q


def test_prompt_normalization_rewrites_closed_set_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_serve_module()
    monkeypatch.setenv("NATURELM_V1_0_NORMALIZE_PROMPTS", "1")
    q = (
        "<Audio><AudioHere></Audio> The objective is to classify the sound into one of "
        "the following categories: dog, cat"
    )
    assert mod._normalize_query_for_v1_0(q) == q
