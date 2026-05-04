"""Regression tests for Increment-7 ingestion helper scripts.

These tests execute the real bash scripts with tiny vendored fixtures to keep the
offline ingestion/rescoring workflow stable while waiting for collaborator
cluster artifacts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fixtures_root() -> Path:
    return _repo_root() / "tests" / "fixtures" / "ingestion_run_dir_v1"


def _copy_fixture_dir(tmp_path: Path, *, name: str) -> Path:
    src = _fixtures_root() / name
    if not src.is_dir():
        raise RuntimeError(f"Missing fixture dir: {src}")
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def _run_bash(script_rel: str, *args: str) -> subprocess.CompletedProcess[str]:
    root = _repo_root()
    script = (root / script_rel).resolve()
    if not script.is_file():
        raise RuntimeError(f"Missing script: {script}")
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=str(root),
        text=True,
        capture_output=True,
    )


def test_validate_run_dir_good_output_leaf_passes(tmp_path: Path) -> None:
    run_dir = _copy_fixture_dir(tmp_path, name="good_output_leaf")
    res = _run_bash("scripts/validate_run_dir.sh", str(run_dir))
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "PASS: artifacts directory looks usable" in res.stdout


def test_validate_run_dir_good_input_leaf_passes(tmp_path: Path) -> None:
    run_dir = _copy_fixture_dir(tmp_path, name="good_input_leaf")
    res = _run_bash("scripts/validate_run_dir.sh", str(run_dir))
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "PASS: artifacts directory looks usable" in res.stdout


@pytest.mark.parametrize(
    ("fixture_name", "expected_stderr_substrings"),
    [
        (
            "bad_missing_sidecar",
            (
                "FAIL:",
                "missing required file:",
                "processed_predictions.jsonl (required for offline rescoring",
            ),
        ),
        (
            "bad_missing_predictions_field",
            (
                "FAIL:",
                'at least one row with `"predictions": [...]`',
            ),
        ),
    ],
)
def test_validate_run_dir_failure_cases_fail_fast(
    tmp_path: Path, fixture_name: str, expected_stderr_substrings: tuple[str, ...]
) -> None:
    run_dir = _copy_fixture_dir(tmp_path, name=fixture_name)
    res = _run_bash("scripts/validate_run_dir.sh", str(run_dir))
    assert res.returncode != 0
    for s in expected_stderr_substrings:
        assert s in res.stderr


def test_ingest_and_rescore_input_leaf_can_skip_rescore(tmp_path: Path) -> None:
    run_dir = _copy_fixture_dir(tmp_path, name="good_input_leaf")
    root = _repo_root()
    script = (root / "scripts" / "ingest_and_rescore.sh").resolve()
    res = subprocess.run(
        ["bash", str(script), str(run_dir)],
        cwd=str(root),
        text=True,
        capture_output=True,
        env={**os.environ, "BEANS_PRO_SKIP_RESCORE": "1"},
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert (
        "rescore skipped" in res.stdout.lower()
        or "rescore skipped" in res.stderr.lower()
    )


def test_ingest_and_rescore_missing_sidecar_fails_before_uv(tmp_path: Path) -> None:
    run_dir = _copy_fixture_dir(tmp_path, name="bad_missing_sidecar")
    res = _run_bash("scripts/ingest_and_rescore.sh", str(run_dir))
    assert res.returncode != 0
    assert "FAIL:" in res.stderr
    assert "missing required file" in res.stderr
    assert (
        "processed_predictions.jsonl (required targets sidecar for offline rescoring"
        in res.stderr
    )
