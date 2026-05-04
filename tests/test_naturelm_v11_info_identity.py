"""Tests for NatureLM v1.1 `/info` identity behavior.

These tests validate that the launcher reports the correct model identity for both:

- HuggingFace-backed weights (default)
- GCS checkpoint overrides via `NATURELM_GCS_CHECKPOINT_URI`
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _load_naturelm_v11_serve_module(
    *,
    monkeypatch: pytest.MonkeyPatch,
    stub_mode: bool,
    gcs_checkpoint_uri: str | None = None,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
) -> object:
    if stub_mode:
        monkeypatch.setenv("NATURELM_STUB_MODE", "1")
    else:
        monkeypatch.delenv("NATURELM_STUB_MODE", raising=False)

    if gcs_checkpoint_uri is None:
        monkeypatch.delenv("NATURELM_GCS_CHECKPOINT_URI", raising=False)
    else:
        monkeypatch.setenv("NATURELM_GCS_CHECKPOINT_URI", gcs_checkpoint_uri)

    if hf_repo_id is None:
        monkeypatch.delenv("NATURELM_HF_REPO_ID", raising=False)
    else:
        monkeypatch.setenv("NATURELM_HF_REPO_ID", hf_repo_id)

    if hf_revision is None:
        monkeypatch.delenv("NATURELM_HF_REVISION", raising=False)
    else:
        monkeypatch.setenv("NATURELM_HF_REVISION", hf_revision)

    repo_root = Path(__file__).resolve().parents[1]
    serve_path = repo_root / "examples" / "servers" / "naturelm-v1.1" / "serve.py"
    if not serve_path.exists():
        raise AssertionError(f"Expected NatureLM v1.1 serve.py at {serve_path}")

    module_name = (
        f"_naturelm_v11_serve_test_{id(monkeypatch)}_"
        f"{hash((stub_mode, gcs_checkpoint_uri, hf_repo_id, hf_revision))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, serve_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to create import spec for naturelm-v1.1 serve.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (
            "gs://bucket/path/merged_variations_f0_v5",
            "merged_variations_f0_v5",
        ),
        (
            "gs://bucket/path/merged_variations_f0_v5/",
            "merged_variations_f0_v5",
        ),
        (
            "gs://bucket/nested/checkpoint-1234",
            "checkpoint-1234",
        ),
    ],
)
def test_gcs_checkpoint_basename_derivation(uri: str, expected: str) -> None:
    # Reference implementation copied from the launcher:
    #   gcs_uri.rstrip("/").rsplit("/", 1)[-1]
    assert uri.rstrip("/").rsplit("/", 1)[-1] == expected


def test_info_identity_hf_vs_gcs_stub_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # HF mode: no GCS URI; /info must reflect repo id + configured revision.
    hf_repo_id = "EarthSpeciesProject/naturelm-audio-1.1.00-private"
    hf_revision = "main"
    serve_hf = _load_naturelm_v11_serve_module(
        monkeypatch=monkeypatch,
        stub_mode=True,
        gcs_checkpoint_uri=None,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
    )
    client_hf = TestClient(serve_hf.app)
    info_hf = client_hf.get("/info").json()
    assert info_hf["name"] == "beans-next-naturelm-v1.1"
    assert info_hf["model"] == hf_repo_id
    assert info_hf["model_revision"] == hf_revision

    # GCS mode: /info must use full GCS URI as model and basename as revision.
    gcs_uri = "gs://foundation-models/naturelm-audio-1.1/base_model/1290000/"
    serve_gcs = _load_naturelm_v11_serve_module(
        monkeypatch=monkeypatch,
        stub_mode=True,
        gcs_checkpoint_uri=gcs_uri,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
    )
    client_gcs = TestClient(serve_gcs.app)
    info_gcs = client_gcs.get("/info").json()
    assert info_gcs["name"] == "beans-next-naturelm-v1.1"
    assert info_gcs["model"] == gcs_uri
    assert info_gcs["model_revision"] == "1290000"
