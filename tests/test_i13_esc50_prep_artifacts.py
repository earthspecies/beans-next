"""Increment 13 prep artifact presence + consistency checks.

These tests are intentionally fast and offline. They validate that Increment 13
prep artifacts (config/docs/script) are present and internally consistent, while
skipping gracefully when earlier Increment 13 tasks have not yet produced the
expected artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _maybe_parse_yaml(text: str) -> object:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(text)


def test_i13_c2_script_exists_and_is_sane() -> None:
    script = _REPO_ROOT / "scripts" / "run_esc50_official_openai_proxy_stub.sh"
    if not script.exists():
        pytest.skip(
            "I13-C2 script missing: scripts/run_esc50_official_openai_proxy_stub.sh"
        )

    body = _read_text(script)
    assert body.startswith("#!"), "script should start with a shebang"
    assert "set -euo pipefail" in body, "script should use `set -euo pipefail`"
    assert "scripts/with_uvicorn.py" in body, (
        "script should use scripts/with_uvicorn.py"
    )
    assert "19085" in body, "script should use fixed port 19085"
    assert "OPENAI_PROXY_STUB=1" in body, "script should enable stub mode"
    assert "beans_zero_esc50_official" in body, "script should run official ESC-50 task"

    # Prefer executable bit, but accept environments where checkout drops it.
    if os.name == "posix":
        assert os.access(script, os.X_OK) or body.startswith(
            "#!/usr/bin/env bash"
        ) or body.startswith(
            "#!/bin/bash"
        ), "script should be executable or clearly a bash script"


def test_i13_c1_docs_mentions_fixed_port_and_with_uvicorn_when_linked() -> None:
    """If evaluation guide links I13 ESC-50 config, it must mention key details."""

    docs = _REPO_ROOT / "docs" / "evaluation_guide.md"
    if not docs.exists():
        pytest.skip("docs missing: docs/evaluation_guide.md")

    cfg = (
        _REPO_ROOT
        / "configs"
        / "benchmarks"
        / "esc50_official_openai_proxy_stub.yaml"
    )
    if not cfg.exists():
        pytest.skip(
            "I13-C1 benchmark config missing: "
            "configs/benchmarks/esc50_official_openai_proxy_stub.yaml"
        )

    body = _read_text(docs)
    if "esc50_official_openai_proxy_stub.yaml" not in body:
        pytest.skip(
            "evaluation guide does not reference the I13 ESC-50 stub config; "
            "skipping doc linkage checks."
        )

    assert "19085" in body, "docs should mention fixed port 19085"
    assert "scripts/with_uvicorn.py" in body, (
        "docs should reference scripts/with_uvicorn.py"
    )


def test_i13_yaml_config_parses_and_references_expected_strings() -> None:
    cfg = (
        _REPO_ROOT
        / "configs"
        / "benchmarks"
        / "esc50_official_openai_proxy_stub.yaml"
    )
    if not cfg.exists():
        pytest.skip(
            "I13-C1 benchmark config missing: "
            "configs/benchmarks/esc50_official_openai_proxy_stub.yaml"
        )

    text = _read_text(cfg)
    parsed = _maybe_parse_yaml(text)
    assert parsed is not None, "benchmark YAML should parse to a non-null document"

    # Robust checks: require key strings, not a strict schema.
    assert "beans_zero_esc50_official" in text, (
        "config should reference official ESC-50 task"
    )
    assert "/predict" in text, "config should reference a predict endpoint"
    assert "19085" in text, "config should reference fixed port 19085"


def test_i13_paper_scaffold_all_models_yaml_parses() -> None:
    cfg = _REPO_ROOT / "configs" / "paper" / "beans_zero_all_models.yaml"
    if not cfg.exists():
        pytest.skip("missing: configs/paper/beans_zero_all_models.yaml")

    text = _read_text(cfg)
    parsed = _maybe_parse_yaml(text)
    assert isinstance(parsed, dict), "YAML should parse to a mapping"

    assert parsed.get("paper_id") == "beans_zero_all_models"
    assert parsed.get("suite"), "expected a top-level suite"
    assert "models" in parsed, "expected a top-level models block"
