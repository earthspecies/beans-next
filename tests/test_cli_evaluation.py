"""Evaluate an HF snapshot through the CLI and a live HTTP model server."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


def test_cli_evaluates_local_hf_snapshot(tmp_path: Path) -> None:
    """Exercise loading, audio transport, inference, scoring, and result files."""
    snapshot = tmp_path / "snapshot"
    split = snapshot / "test"
    split.mkdir(parents=True)
    sf.write(split / "sample.wav", np.zeros(1600), 16000)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "sample-1",
                    "task": "crow-description",
                    "audio_paths": ["sample.wav"],
                    "instruction": "<Audio><AudioHere></Audio> Choose A or B.",
                    "output": "A",
                }
            ]
        ),
        split / "metadata.parquet",
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = tmp_path / "results"
    env = {
        **os.environ,
        "BEANS_NEXT_HF_BEANS_NEXT_ROOT": str(snapshot),
        "HF_HUB_OFFLINE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "scripts/with_uvicorn.py",
            "--app",
            "examples.servers.dummy.serve:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--cwd",
            ".",
            "--",
            sys.executable,
            "-m",
            "beans_next.cli",
            "run",
            "--task-id",
            "beans_next_crow_description",
            "--predict-url",
            f"http://127.0.0.1:{port}/predict",
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_samples"] == 1
    assert summary["n_errors"] == 0
    for name in ("predictions", "processed_predictions", "scored_predictions"):
        rows = [
            json.loads(line)
            for line in (output / f"{name}.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["sample_id"] == "sample-1"
        assert rows[0].get("error") is None
    assert "top1_accuracy" in rows[0]["scores"]
