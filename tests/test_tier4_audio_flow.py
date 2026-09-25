"""Exercise distinct audio signals from a Hub snapshot through prompt assembly."""

import ast
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from beans_next.datasets.beans_next_hub import _iter_multiaudio_examples
from beans_next.datasets.rows import (
    _build_multiaudio_dataset_example,
)
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER
from beans_next.prompts.renderer import PromptRenderer, load_builtin_prompt_yaml


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_snapshot_audio_order_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool, workers: int
) -> None:
    root = tmp_path / "test"
    (root / "audio").mkdir(parents=True)
    names = ["audio/a.wav", "audio/b.wav", "audio/q.wav"]
    for i, name in enumerate(names):
        sf.write(root / name, np.full(1600, (i + 1) / 10), 16000, subtype="FLOAT")
    prompt = (
        f"A: {AUDIO_PLACEHOLDER}\nB: {AUDIO_PLACEHOLDER}\nQuery: {AUDIO_PLACEHOLDER}"
    )
    row = {
        "sample_id": "order-check",
        "task": "crow-4way",
        "tier": 4,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "B"},
        ],
        "context_audio_paths": names if legacy else names[:-1],
        "query_audio_path": names[0] if legacy else names[-1],
    }
    pq.write_table(pa.Table.from_pylist([row]), root / "metadata.parquet")
    monkeypatch.setenv("BEANS_NEXT_HF_BEANS_NEXT_ROOT", str(tmp_path))
    (example,) = _iter_multiaudio_examples(
        "unused",
        subset="crow-4way",
        split="test",
        revision="unused",
        task_id=None,
        limit=None,
        workers=workers,
        load_audio=True,
    )
    ordered = example.metadata["audio_paths"]
    assert ordered == [str(root / n) for n in names]
    assert example.metadata["audio_path"] == ordered[-1]
    for i, path in enumerate(ordered):
        signal, rate = sf.read(path)
        assert rate == 16000
        assert np.allclose(signal, (i + 1) / 10)
    request = PromptRenderer(
        load_builtin_prompt_yaml("beans_next_multiaudio_passthrough_v1.yaml")
    ).render(example)
    assert [a.data for a in request.audio_inputs] == ordered
    query = PromptRenderer(
        load_builtin_prompt_yaml("beans_next_multiaudio_query_only_v1.yaml")
    ).render(example)
    assert [a.data for a in query.audio_inputs] == [ordered[-1]]


def test_query_alias_cannot_override_ordered_query() -> None:
    ex = _build_multiaudio_dataset_example(
        {},
        sample_id="x",
        audio_paths=["a", "q"],
        query_audio_path="a",
        split="test",
        task_id=None,
    )
    assert ex.metadata["audio_path"] == "q"


def test_missing_snapshot_clip_fails_without_dropping_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "test"
    (root / "audio").mkdir(parents=True)
    sf.write(root / "audio/a.wav", np.zeros(160), 16000)
    row = {
        "id": "missing-reference",
        "task": "crow-4way",
        "tier": 4,
        "audio_paths": ["audio/a.wav", "audio/missing.wav"],
        "messages": [{"role": "user", "content": AUDIO_PLACEHOLDER * 2}],
    }
    pq.write_table(pa.Table.from_pylist([row]), root / "metadata.parquet")
    monkeypatch.setenv("BEANS_NEXT_HF_BEANS_NEXT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        list(
            _iter_multiaudio_examples(
                "unused",
                subset="crow-4way",
                split="test",
                revision="unused",
                task_id=None,
                limit=None,
                workers=1,
                load_audio=True,
            )
        )


def test_naturelm_v10_rejects_multi_audio_before_inference() -> None:
    source = Path(__file__).parents[1] / "examples/servers/naturelm-v1.0/serve.py"
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_real_inference"
    )
    namespace: dict = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )  # noqa: S102
    with pytest.raises(ValueError, match="Refusing to drop"):
        namespace["_run_real_inference"](
            None, SimpleNamespace(audio_inputs=[object(), object()])
        )


@pytest.mark.parametrize("includes_query", [False, True])
def test_legacy_audio_table_retains_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, includes_query: bool
) -> None:
    from beans_next.datasets import beans_next_hub

    audio = []
    for i, key in enumerate(["a", "b", "q"]):
        buf = io.BytesIO()
        sf.write(buf, np.full(160, (i + 1) / 10), 16000, format="WAV", subtype="FLOAT")
        audio.append({"audio_id": key, "audio_bytes": buf.getvalue()})
    pq.write_table(pa.Table.from_pylist(audio), tmp_path / "beans_next_audio.parquet")
    row = {
        "sample_id": "legacy",
        "task": "crow-4way",
        "tier": 4,
        "messages": [{"role": "user", "content": AUDIO_PLACEHOLDER * 3}],
        "audio_ids": ["a", "b", "q"] if includes_query else ["a", "b"],
        "query_audio_id": "a" if includes_query else "q",
    }
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "metadata.parquet")
    monkeypatch.setenv("BEANS_NEXT_HF_BEANS_NEXT_ROOT", str(tmp_path))

    def materialize(data: bytes, *, stem: str) -> str:
        path = tmp_path / (stem + ".wav")
        path.write_bytes(data)
        return str(path)

    monkeypatch.setattr(beans_next_hub, "_materialize_wav_bytes", materialize)
    (ex,) = _iter_multiaudio_examples(
        "unused",
        subset="crow-4way",
        split="test",
        revision="unused",
        task_id=None,
        limit=None,
        workers=1,
        load_audio=True,
    )
    assert len(ex.metadata["audio_paths"]) == 3
    for i, path in enumerate(ex.metadata["audio_paths"]):
        assert np.allclose(sf.read(path)[0], (i + 1) / 10)
    assert ex.metadata["audio_path"] == ex.metadata["audio_paths"][-1]
