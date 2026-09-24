"""Check lossless migration and evaluation of compact Hub rows."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from beans_next.datasets.beans_next_hub import iter_hf_beans_next_examples
from beans_next.datasets.hub_schema import MAIN_COLUMNS, compact_hub_row
from beans_next.prompts.audio_tags import AUDIO_PLACEHOLDER
from beans_next.prompts.renderer import PromptRenderer, load_builtin_prompt_yaml


def legacy_row(tier: int) -> dict:
    row = dict.fromkeys(MAIN_COLUMNS)
    row.update(
        id="stable-id",
        sample_id="different-stable-sample-id",
        tier=tier,
        task="crow-4way" if tier == 4 else "crow-description",
        metadata=json.dumps({"source_dataset": "source", "sampling_seed": 42}),
        instruction=f"{AUDIO_PLACEHOLDER}\nChoose A or B.",
        output="B",
        instruction_text="Choose A or B.",
        passed={"valid_response": True},
        crop_start=1.0,
        crop_end=2.0,
        duration_sec=1.0,
    )
    if tier >= 3:
        row["messages"] = [
            {
                "role": "user",
                "content": f"{AUDIO_PLACEHOLDER}" * (3 if tier == 4 else 1),
            },
            {"role": "assistant", "content": "B"},
        ]
        row["instruction"] = row["output"] = None
    return row


@pytest.mark.parametrize("tier", [1, 2, 3, 4])
def test_compact_row_is_lossless_and_keeps_both_ids(tier: int) -> None:
    before = legacy_row(tier)
    after, source = compact_hub_row(before)
    assert tuple(after) == MAIN_COLUMNS
    assert after["id"] == before["id"]
    assert after["sample_id"] == before["sample_id"]
    assert {**after, **json.loads(source["original_fields"])} == before
    annotations = json.loads(after["metadata"])
    assert "sampling_seed" not in annotations
    assert annotations["crop_start"] == 1.0
    assert after["source_dataset"] == "source"
    if tier < 3:
        assert after["messages"][0]["content"] == before["instruction"]
        assert after["messages"][1]["content"] == before["output"]
    else:
        assert after["messages"] == before["messages"]


def test_migration_refuses_missing_target() -> None:
    row = legacy_row(1)
    row["output"] = None
    with pytest.raises(ValueError, match="instruction/output"):
        compact_hub_row(row)


@pytest.mark.parametrize("tier", [1, 4])
@pytest.mark.parametrize("workers", [1, 2])
def test_compact_snapshot_loads_without_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: int, workers: int
) -> None:
    root = tmp_path / "test"
    (root / "audio").mkdir(parents=True)
    row = legacy_row(tier)
    paths = []
    for index in range(3 if tier == 4 else 1):
        scratch = root / "audio/temp.wav"
        sf.write(scratch, np.full(160, (index + 1) / 10), 16000, subtype="FLOAT")
        digest = hashlib.sha256(scratch.read_bytes()).hexdigest()
        path = f"audio/{digest}.wav"
        scratch.rename(root / path)
        paths.append(path)
    if tier == 4:
        row["context_audio_paths"], row["query_audio_path"] = paths[:-1], paths[-1]
    else:
        row["file_name"] = paths[0]
    compact, _ = compact_hub_row(row)
    pq.write_table(pa.Table.from_pylist([compact]), root / "metadata.parquet")
    monkeypatch.setenv("BEANS_NEXT_HF_BEANS_NEXT_ROOT", str(tmp_path))
    (example,) = iter_hf_beans_next_examples(
        "unused", subset=row["task"], workers=workers
    )
    assert example.sample_id == row["id"]
    assert example.labels == "B"
    if tier == 4:
        request = PromptRenderer(
            load_builtin_prompt_yaml("beans_next_multiaudio_passthrough_v1.yaml")
        ).render(example)
        assert [audio.data for audio in request.audio_inputs] == [
            str(root / p) for p in paths
        ]
        assert example.metadata["audio_path"] == str(root / paths[-1])
    else:
        assert example.metadata["instruction"] == row["instruction"]
        assert example.metadata["audio_path"] == str(root / paths[0])
    verifier = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/verify_beans_next_hf_bundle.py")
    )["main"]
    assert verifier(["--bundle", str(tmp_path), "--sample", "0"]) == 0
    (root / paths[-1]).unlink()
    assert verifier(["--bundle", str(tmp_path), "--sample", "0"]) == 1
