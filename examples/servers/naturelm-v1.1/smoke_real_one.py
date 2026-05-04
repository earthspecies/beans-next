"""Send one real BEANS-Zero sample to a running NatureLM v1.1 launcher.

Usage (from repo root, with launcher already running):

  uv run python examples/servers/naturelm-v1.1/smoke_real_one.py \
    --predict-url http://127.0.0.1:8001/predict \
    --split esc50
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from typing import Any

import requests


def _wav_base64_from_audio_array(audio: Any, sample_rate: int) -> str:  # noqa: ANN401
    import numpy as np
    import soundfile as sf

    arr = np.asarray(audio, dtype="float32")
    if arr.ndim == 2:
        axis = int(np.argmin(arr.shape))
        arr = arr.mean(axis=axis)

    buf = io.BytesIO()
    sf.write(buf, arr, samplerate=int(sample_rate), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-url", required=True)
    parser.add_argument("--split", default="esc50")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Missing dependency 'datasets'. Run from the repo root env "
            "(e.g. `uv sync`) or install `datasets` into the environment."
        ) from exc

    ds = load_dataset(
        "EarthSpeciesProject/BEANS-Zero",
        split="test",
        streaming=True,
    )

    sample: dict[str, Any] | None = None
    for row in ds:
        if row.get("dataset_name") == args.split:
            sample = row
            break

    if sample is None:
        raise SystemExit(f"Could not find any sample with dataset_name={args.split!r}")

    # HF datasets may represent audio either as an Audio feature dict or raw array.
    audio_obj = sample.get("audio")
    if (
        isinstance(audio_obj, dict)
        and "array" in audio_obj
        and "sampling_rate" in audio_obj
    ):
        audio_arr = audio_obj["array"]
        sr = int(audio_obj["sampling_rate"])
    else:
        audio_arr = audio_obj
        md = sample.get("metadata")
        sr = int(json.loads(md)["sample_rate"]) if isinstance(md, str) else 16000

    instruction = sample.get("instruction") or sample.get("instruction_text")
    if not isinstance(instruction, str) or not instruction:
        raise SystemExit("Sample is missing instruction text")

    wav_b64 = _wav_base64_from_audio_array(audio_arr, sample_rate=sr)

    payload = {
        "schema_version": "predictions_v1",
        "requests": [
            {
                "sample_id": "smoke_real_one",
                "messages": [{"role": "user", "content": instruction}],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": wav_b64,
                        "sample_rate": sr,
                    }
                ],
                "generation_config": {"max_tokens": 128, "temperature": 1.0},
            }
        ],
    }

    resp = requests.post(args.predict_url, json=payload, timeout=args.timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2))

    item = data["responses"][0]
    if item.get("error"):
        raise SystemExit(f"Launcher returned error: {item['error']}")

    preds = item.get("predictions") or []
    if not preds:
        raise SystemExit("No predictions returned")

    print("\n--- prediction ---")
    print(preds[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
