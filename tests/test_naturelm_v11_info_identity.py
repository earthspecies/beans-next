"""Tests for NatureLM v1.1 `/info` identity behavior.

These tests validate that the launcher reports the correct model identity for both:

- HuggingFace-backed weights (default)
- Local checkpoint overrides via `NATURELM_LOCAL_CHECKPOINT_DIR`
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
    local_checkpoint_dir: str | None = None,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
) -> object:
    if stub_mode:
        monkeypatch.setenv("NATURELM_STUB_MODE", "1")
    else:
        monkeypatch.delenv("NATURELM_STUB_MODE", raising=False)

    if local_checkpoint_dir is None:
        monkeypatch.delenv("NATURELM_LOCAL_CHECKPOINT_DIR", raising=False)
    else:
        monkeypatch.setenv("NATURELM_LOCAL_CHECKPOINT_DIR", local_checkpoint_dir)

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
        f"{hash((stub_mode, local_checkpoint_dir, hf_repo_id, hf_revision))}"
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
            "/models/path/merged_variations_f0_v5",
            "merged_variations_f0_v5",
        ),
        (
            "/models/path/merged_variations_f0_v5/",
            "merged_variations_f0_v5",
        ),
        (
            "/models/nested/checkpoint-1234",
            "checkpoint-1234",
        ),
    ],
)
def test_local_checkpoint_basename_derivation(uri: str, expected: str) -> None:
    # Reference implementation copied from the launcher:
    #   local_uri.rstrip("/").rsplit("/", 1)[-1]
    assert uri.rstrip("/").rsplit("/", 1)[-1] == expected


def test_info_identity_hf_vs_local_stub_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # HF mode: no Local URI; /info must reflect repo id + configured revision.
    hf_repo_id = "test/model"
    hf_revision = "main"
    serve_hf = _load_naturelm_v11_serve_module(
        monkeypatch=monkeypatch,
        stub_mode=True,
        local_checkpoint_dir=None,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
    )
    client_hf = TestClient(serve_hf.app)
    info_hf = client_hf.get("/info").json()
    assert info_hf["name"] == "beans-next-naturelm-v1.1"
    assert info_hf["model"] == hf_repo_id
    assert info_hf["model_revision"] == hf_revision

    # Local mode: /info must use full Local URI as model and basename as revision.
    local_uri = "/models/checkpoint-1290000/"
    serve_local = _load_naturelm_v11_serve_module(
        monkeypatch=monkeypatch,
        stub_mode=True,
        local_checkpoint_dir=local_uri,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
    )
    client_local = TestClient(serve_local.app)
    info_local = client_local.get("/info").json()
    assert info_local["name"] == "beans-next-naturelm-v1.1"
    assert info_local["model"] == local_uri
    assert info_local["model_revision"] == "checkpoint-1290000"


def test_hf_checkpoint_download_uses_requested_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    serve = _load_naturelm_v11_serve_module(
        monkeypatch=monkeypatch,
        stub_mode=False,
        local_checkpoint_dir=None,
        hf_repo_id="test/model",
        hf_revision="pinned-revision",
    )
    snapshot = tmp_path / "resolved-commit"
    snapshot.mkdir()
    calls = []

    def download(**kwargs: object) -> str:
        """Capture the checkpoint request.

        Returns
        -------
        str
            Local snapshot directory.
        """
        calls.append(kwargs)
        return str(snapshot)

    model = object()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(serve, "_maybe_load_naturelm", lambda path: model)
    serve._ensure_ready_or_raise()
    assert calls == [
        {"repo_id": "test/model", "revision": "pinned-revision", "token": None}
    ]
    assert serve._state.model is model
    assert serve._state.snapshot_path == str(snapshot)
    assert serve.info().model_revision == "resolved-commit"
