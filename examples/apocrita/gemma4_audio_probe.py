"""Decision-1 audio-sensitivity probe for the Gemma 4 launcher.

Per the plan (Decision 1: serving backend), the risk with the Gemma 4 audio
path is not a crash but silent failure: audio never reaches the model,
``processor_config.json``/``AutoProcessor`` quietly drops it, and the model
still emits fluent, plausible label text. A green ``/health`` and a
non-empty ``predictions.jsonl`` would not catch that.

This script builds a small (default 12-row) slice of a real BEANS-Zero eval
task (``beans_zero_esc50`` by default: short clips, clear single-label
ground truth), and for each row sends three variants of the *same* prompt to
the running Gemma 4 launcher's ``/predict`` endpoint:

1. ``real``    - the original clip.
2. ``silence`` - the same duration/sample-rate of digital silence.
3. ``swap``    - a different row's clip (with a different ground-truth
   label), substituted in place of the original.

If the three conditions return identical text for most rows (majority of
the sample, i.e. more than half), audio is not influencing the model's
output and this exits non-zero. All three conditions are sent as
``base64_wav`` payloads (regardless of the launcher's configured payload
type) so a synthetic silence/swap clip never needs a spot in
``--allowed-local-media-path``.

It also runs a cross-check against a Hugging Face Transformers reference
forward pass on a handful of rows (default 3), loading
``Gemma4UnifiedForConditionalGeneration`` + ``AutoProcessor`` directly from
``--model-path``. This is best-effort: if it cannot get a CUDA allocation
(for example because a vLLM server sharing the same GPU has already claimed
most of its memory, per ``--gpu-memory-utilization``), it is reported as a
skipped check with a clear warning rather than silently counted as a pass,
and does not change the exit code on its own -- the launcher-side check
above is the hard gate.

This script imports ``beans_next`` (dataset loading, prompt rendering) to
build realistic rows; unlike the launchers under ``examples/servers/``, it is
not part of the model-agnostic serving contract and is free to do so.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Explicit majority threshold for "the three conditions return the same text
# for most rows": more than half of a 12-row sample is 7.
DEFAULT_NUM_ROWS = 12
DEFAULT_MAJORITY_THRESHOLD = 7
DEFAULT_HF_CROSSCHECK_ROWS = 3
DEFAULT_EVAL_TASK = "beans_zero_esc50"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-url", required=True)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local snapshot dir for the HF Transformers reference cross-check.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-task", default=DEFAULT_EVAL_TASK)
    parser.add_argument(
        "--data-source",
        default="esp_data",
        help="beans_next data backend to load probe rows from (esp_data or huggingface).",
    )
    parser.add_argument("--num-rows", type=int, default=DEFAULT_NUM_ROWS)
    parser.add_argument(
        "--majority-threshold",
        type=int,
        default=DEFAULT_MAJORITY_THRESHOLD,
        help="Fail if at least this many rows have identical text across all 3 conditions.",
    )
    parser.add_argument(
        "--hf-crosscheck-rows", type=int, default=DEFAULT_HF_CROSSCHECK_ROWS
    )
    parser.add_argument(
        "--hf-crosscheck",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto: best-effort, warn and continue on CUDA OOM/import failure.",
    )
    parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    return parser


def _load_eval_task_mapping(task_id: str) -> dict[str, Any]:
    path = REPO_ROOT / "beans_next" / "registry" / "eval_task" / f"{task_id}.yaml"
    if not path.is_file():
        raise SystemExit(f"Unknown eval task (no registry file): {path}")
    data = yaml.safe_load(path.read_text())
    if task_id not in data:
        raise SystemExit(f"Eval task file {path} does not define key {task_id!r}")
    mapping = dict(data[task_id])
    mapping.setdefault("eval_task_id", task_id)
    return mapping


def _build_namespace(args: argparse.Namespace) -> Namespace:
    return Namespace(
        data_source=args.data_source,
        modality_mode="audio",
        hf_path=None,
        split="test",
        hf_config=None,
        dataset_name=None,
        sample_fraction=None,
        limit=max(args.num_rows * 3, args.num_rows + 8),
        hf_revision=None,
        prompt_yaml=None,
    )


def _read_audio(item: Any) -> tuple[np.ndarray, int] | None:
    """Read the first audio slot of a rendered wire item to (samples, sr)."""

    if not item.audio_inputs:
        return None
    slot = item.audio_inputs[0]
    if slot.payload_type == "file_path":
        data, sr = sf.read(slot.data, dtype="float32", always_2d=False)
        return data, sr
    if slot.payload_type == "base64_wav":
        import base64
        import io

        raw = base64.b64decode(slot.data)
        with io.BytesIO(raw) as buf:
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
        return data, sr
    raise ValueError(f"Unsupported payload_type for probe: {slot.payload_type!r}")


def _wav_base64(data: np.ndarray, sample_rate: int) -> str:
    import base64
    import io

    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _wire_item_with_audio_b64(item: Any, wav_b64: str, sample_rate: int) -> dict[str, Any]:
    payload = item.model_dump(mode="json")
    payload["audio_inputs"] = [
        {"payload_type": "base64_wav", "data": wav_b64, "sample_rate": sample_rate}
    ]
    return payload


def _post_batch(
    client: httpx.Client, predict_url: str, requests: list[dict[str, Any]]
) -> dict[str, str]:
    body = {"schema_version": "predictions_v1", "requests": requests}
    resp = client.post(predict_url, json=body)
    resp.raise_for_status()
    payload = resp.json()
    out: dict[str, str] = {}
    for row in payload.get("responses", []):
        sid = row.get("sample_id")
        preds = row.get("predictions") or [""]
        out[sid] = (preds[0] if preds else "") or ""
        err = row.get("error")
        if err:
            print(f"WARNING: sample_id={sid} returned error: {err}", file=sys.stderr)
    return out


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _run_hf_crosscheck(
    model_path: str,
    rows: list[tuple[Any, np.ndarray, int, np.ndarray, int]],
    output_dir: Path,
) -> str:
    """Best-effort HF Transformers reference cross-check.

    Returns
    -------
    str
        One of "passed", "flagged", "skipped".
    """

    try:
        import torch
        from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: HF Transformers cross-check unavailable: {exc}", file=sys.stderr)
        return "skipped"

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: HF Transformers cross-check could not load the model "
            f"(likely a CUDA allocation conflict with a co-resident vLLM server): {exc}",
            file=sys.stderr,
        )
        return "skipped"

    flagged = 0
    log: list[dict[str, Any]] = []
    for item, real_audio, real_sr, silence_audio, silence_sr in rows:
        text_prompt = next(
            (m.content for m in item.messages if m.role == "user"), ""
        )
        outputs = {}
        for label, audio, sr in (
            ("real", real_audio, real_sr),
            ("silence", silence_audio, silence_sr),
        ):
            inputs = processor(
                text=text_prompt, audio=audio, sampling_rate=sr, return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=32)
            outputs[label] = processor.batch_decode(gen, skip_special_tokens=True)[0]
        same = _normalize(outputs["real"]) == _normalize(outputs["silence"])
        flagged += int(same)
        log.append(
            {
                "sample_id": item.sample_id,
                "real_text": outputs["real"],
                "silence_text": outputs["silence"],
                "identical": same,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hf_crosscheck.json").write_text(json.dumps(log, indent=2))
    majority = (len(rows) // 2) + 1
    return "flagged" if flagged >= majority else "passed"


def main() -> int:
    """Run the Decision-1 audio-sensitivity probe.

    Returns
    -------
    int
        Zero if the probe passes (audio appears to reach the model), non-zero
        if the launcher-side check is flagged (most rows insensitive to
        audio content) or on a hard error.
    """

    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_task = _load_eval_task_mapping(args.eval_task)
    ns = _build_namespace(args)

    from beans_next.prompts.renderer import PromptRenderer
    from beans_next.runner.runner import (
        _load_examples_for_eval_task,
        _prompt_spec_from_eval_task,
        model_request_to_wire_item,
    )

    examples = _load_examples_for_eval_task(eval_task, args=ns)
    if len(examples) < args.num_rows + 1:
        raise SystemExit(
            f"Loaded only {len(examples)} examples for {args.eval_task!r}, "
            f"need at least {args.num_rows + 1} (num_rows + 1 spare for swaps)."
        )

    spec = _prompt_spec_from_eval_task(eval_task, args=ns)
    renderer = PromptRenderer(spec, modality_mode="audio")

    items = [renderer.render(ex) for ex in examples]
    wire_items = [model_request_to_wire_item(it, preserve_file_paths=True) for it in items]

    probe_rows = wire_items[: args.num_rows]
    probe_examples = examples[: args.num_rows]
    spare_items = wire_items[args.num_rows :]
    spare_examples = examples[args.num_rows :]

    def _swap_candidate(idx: int) -> tuple[Any, Any]:
        own_label = probe_examples[idx].labels
        for spare_item, spare_ex in zip(spare_items, spare_examples, strict=False):
            if spare_ex.labels != own_label:
                return spare_item, spare_ex
        # Fall back to the next probe row if no spare has a different label.
        for j, ex in enumerate(probe_examples):
            if j != idx and ex.labels != own_label:
                return probe_rows[j], ex
        raise SystemExit(
            "Could not find any row with a different ground-truth label to build "
            "the 'swap' condition; increase --num-rows or pick a task with more "
            "label diversity."
        )

    real_reqs: list[dict[str, Any]] = []
    silence_reqs: list[dict[str, Any]] = []
    swap_reqs: list[dict[str, Any]] = []
    hf_rows: list[tuple[Any, np.ndarray, int, np.ndarray, int]] = []

    for idx, item in enumerate(probe_rows):
        audio = _read_audio(item)
        if audio is None:
            raise SystemExit(
                f"sample_id={item.sample_id} has no audio_inputs; "
                f"{args.eval_task!r} must be an audio task for this probe."
            )
        data, sr = audio
        real_reqs.append(_wire_item_with_audio_b64(item, _wav_base64(data, sr), sr))

        silence = np.zeros_like(data)
        silence_reqs.append(
            _wire_item_with_audio_b64(item, _wav_base64(silence, sr), sr)
        )

        swap_item, _swap_ex = _swap_candidate(idx)
        swap_audio = _read_audio(swap_item)
        assert swap_audio is not None
        swap_data, swap_sr = swap_audio
        swap_reqs.append(
            _wire_item_with_audio_b64(item, _wav_base64(swap_data, swap_sr), swap_sr)
        )

        if len(hf_rows) < args.hf_crosscheck_rows:
            hf_rows.append((item, data, sr, silence, sr))

    with httpx.Client(timeout=args.request_timeout_sec) as client:
        real_out = _post_batch(client, args.predict_url, real_reqs)
        silence_out = _post_batch(client, args.predict_url, silence_reqs)
        swap_out = _post_batch(client, args.predict_url, swap_reqs)

    per_row: list[dict[str, Any]] = []
    flagged = 0
    for item in probe_rows:
        sid = item.sample_id
        real_text = real_out.get(sid, "")
        silence_text = silence_out.get(sid, "")
        swap_text = swap_out.get(sid, "")
        identical = (
            _normalize(real_text) == _normalize(silence_text)
            and _normalize(real_text) == _normalize(swap_text)
        )
        flagged += int(identical)
        per_row.append(
            {
                "sample_id": sid,
                "real_text": real_text,
                "silence_text": silence_text,
                "swap_text": swap_text,
                "identical_across_conditions": identical,
            }
        )

    report = {
        "eval_task": args.eval_task,
        "num_rows": len(probe_rows),
        "majority_threshold": args.majority_threshold,
        "flagged_rows": flagged,
        "rows": per_row,
    }
    (args.output_dir / "probe_report.json").write_text(json.dumps(report, indent=2))

    launcher_failed = flagged >= args.majority_threshold
    print(
        f"launcher-side probe: {flagged}/{len(probe_rows)} rows identical across "
        f"real/silence/swap (fail threshold={args.majority_threshold})"
    )

    hf_status = "skipped"
    if args.hf_crosscheck != "off":
        hf_status = _run_hf_crosscheck(args.model_path, hf_rows, args.output_dir)
        print(f"HF Transformers cross-check: {hf_status}")
        if hf_status == "flagged" and args.hf_crosscheck == "on":
            launcher_failed = True

    if launcher_failed:
        print(
            "FATAL: audio does not appear to reach the model for most probed rows. "
            "Per Decision 1, do not proceed with GEMMA4_BACKEND=vllm for the full "
            "run; fall back to GEMMA4_BACKEND=transformers and re-probe, or "
            "investigate AutoProcessor/feature_extractor per "
            "setup_gemma4_runtime.sbatch.",
            file=sys.stderr,
        )
        return 1

    print("Audio-sensitivity probe passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
