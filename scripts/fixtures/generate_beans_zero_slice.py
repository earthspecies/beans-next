"""Generate a tiny deterministic BEANS-Zero-like fixture slice (CPU-only).

This script writes a small, self-contained fixture bundle intended for later
golden-run regression tests. It does not download HuggingFace datasets and does
not run any model inference.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import textwrap
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CREATED_AT_UTC = "1970-01-01T00:00:00Z"
FIXTURE_FORMAT_VERSION = "1"
BUNDLE_ID = "beans_zero_slice_v1"


@dataclass(frozen=True)
class FixtureExample:
    """Synthetic fixture row used to generate the on-disk bundle.

    Notes
    -----
    This is intentionally minimal and does not depend on HuggingFace datasets. It exists
    only to generate stable `inputs/slice.json` and `inputs/requests.jsonl` for CPU-only
    fixture validation.
    """

    sample_id: str
    task_id: str
    split: str
    labels: str | list[str] | dict[str, Any] | None
    metadata: dict[str, Any]

    def to_dataset_example_v1(self) -> dict[str, Any]:
        return {
            "schema_version": "beans_next.dataset_example.v1",
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "split": self.split,
            "labels": self.labels,
            "metadata": dict(self.metadata),
        }


def _utc_now_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def _silence_wav_mono_16bit_bytes(*, sample_rate: int, duration_sec: float) -> bytes:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    n_frames = int(round(sample_rate * duration_sec))
    if n_frames <= 0:
        raise ValueError("computed n_frames must be positive")

    # 16-bit PCM little-endian: silence = 0.
    raw_pcm = (b"\x00\x00") * n_frames

    # Write a minimal RIFF/WAV container to bytes using stdlib `wave`.
    # Vendor only base64 in the fixture bundle (CI-friendly; no binary).
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def _regen_command(argv: list[str]) -> str:
    # Keep it copy/pasteable and stable-ish across machines:
    # - If invoked via `uv run python ...`, argv[0] is the script path.
    # - The recommended regeneration command is explicit about `uv run python`.
    script = Path(argv[0]).as_posix()
    return f"uv run python {script} --out tests/fixtures/{BUNDLE_ID} --force"


def _build_examples(out_dir: Path) -> list[FixtureExample]:
    meta_base = {
        "fixture_note": (
            "Synthetic tiny fixture (no HF download). Audio is short silence WAV; "
            "labels are minimal to drive prompt rendering + scoring paths."
        ),
    }

    return [
        FixtureExample(
            sample_id="fixture:beans_zero_slice_v1:esc50:0000",
            task_id="beans_zero_esc50",
            split="test",
            labels="chainsaw",
            metadata={
                **meta_base,
                "source_dataset": "synthetic",
                "dataset_name": "esc50",
                "instruction_text": "Classify the audio into one label.",
            },
        ),
        FixtureExample(
            sample_id="fixture:beans_zero_slice_v1:dcase:0000",
            task_id="beans_zero_dcase",
            split="test",
            labels=["sparrow", "wind"],
            metadata={
                **meta_base,
                "source_dataset": "synthetic",
                "dataset_name": "dcase",
                "instruction_text": "Detect all labels that apply.",
            },
        ),
        FixtureExample(
            sample_id="fixture:beans_zero_slice_v1:captioning:0000",
            task_id="beans_zero_captioning",
            split="test",
            labels="A short description of the recording.",
            metadata={
                **meta_base,
                "source_dataset": "synthetic",
                "dataset_name": "captioning",
                "instruction_text": "Describe the audio recording.",
            },
        ),
    ]


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _write_manifest_yaml(
    out_dir: Path,
    *,
    created_at_utc: str,
    regenerate_command: str,
    audio_payload_type: str,
) -> None:
    manifest = textwrap.dedent(
        f"""\
        fixture_format_version: "{FIXTURE_FORMAT_VERSION}"
        bundle_id: "{BUNDLE_ID}"
        created_at_utc: "{created_at_utc}"
        description: "Tiny synthetic BEANS-Zero-like slice (CPU-only inputs)."
        phase: "phase_a_inputs_only"

        model_identity:
          source: "info_endpoint"
          info:
            name: "UNKNOWN_YET"
            model: "UNKNOWN_YET"
            model_revision: "UNKNOWN_YET"
          info_captured_at_utc: "{created_at_utc}"

        inputs:
          slice_path: "inputs/slice.json"
          requests_path: "inputs/requests.jsonl"
          audio:
            payload_type: "{audio_payload_type}"
            storage: "none"

        expected:
          variant_id: "golden_pending"
          predictions_path: "expected/predictions.jsonl"
          processed_predictions_path: "expected/processed_predictions.jsonl"
          scored_predictions_path: "expected/scored_predictions.jsonl"
          summary_path: "expected/summary.json"
          model_identity_path: "expected/model_identity.json"

        regenerate:
          command: |
            {regenerate_command}
          notes: >
            Phase B (GPU) will populate expected/* after capturing goldens
            from a real launcher.
        """
    )
    _write_text(out_dir / "manifest.yaml", manifest)


def _verify_bundle(out_dir: Path) -> None:
    manifest = out_dir / "manifest.yaml"
    slice_path = out_dir / "inputs" / "slice.json"
    req_path = out_dir / "inputs" / "requests.jsonl"
    expected_readme = out_dir / "expected" / "README.md"
    versions_path = out_dir / "metadata" / "versions.json"
    regen_path = out_dir / "metadata" / "regeneration.md"
    prov_path = out_dir / "metadata" / "provenance.md"

    required_files = (
        manifest,
        slice_path,
        req_path,
        expected_readme,
        versions_path,
        regen_path,
        prov_path,
    )
    for p in required_files:
        if not p.exists():
            raise RuntimeError(f"missing required file: {p}")
        if p.stat().st_size <= 0:
            raise RuntimeError(f"empty required file: {p}")

    # Parse YAML manifest (minimal checks; pytest does deeper validation).
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        msg = f"PyYAML is required to verify manifest.yaml: {exc}"
        raise RuntimeError(msg) from exc

    m = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(m, dict):
        raise RuntimeError("manifest.yaml must be a YAML mapping")
    if m.get("fixture_format_version") != FIXTURE_FORMAT_VERSION:
        raise RuntimeError("manifest.yaml has unexpected fixture_format_version")
    if m.get("bundle_id") != BUNDLE_ID:
        raise RuntimeError("manifest.yaml bundle_id mismatch")

    # Parse requests JSONL and basic shape checks.
    n = 0
    with req_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                raise RuntimeError("inputs/requests.jsonl contains blank lines")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError("request item row must be a JSON object")
            if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
                raise RuntimeError("request item missing sample_id")
            n += 1
    if n <= 0:
        raise RuntimeError("inputs/requests.jsonl had zero rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output fixture directory (e.g. tests/fixtures/beans_zero_slice_v1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--created-at",
        default=None,
        help=(
            "ISO-8601 created_at timestamp for the manifest. "
            f"Default is deterministic ({DEFAULT_CREATED_AT_UTC})."
        ),
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    if out_dir.exists():
        if not args.force:
            raise SystemExit(
                f"--out already exists: {out_dir} (use --force to overwrite)"
            )
        if not out_dir.is_dir():
            raise SystemExit(f"--out exists but is not a directory: {out_dir}")
        for child in list(out_dir.iterdir()):
            _rmtree(child)
    out_dir.mkdir(parents=True, exist_ok=True)

    created_at = args.created_at or os.environ.get("SOURCE_DATE_EPOCH")
    if created_at is not None:
        # If SOURCE_DATE_EPOCH is set and is numeric, normalize to ISO.
        try:
            epoch = int(created_at)
        except ValueError:
            # Treat as an ISO string provided by --created-at.
            created_at_iso = str(created_at)
        else:
            created_at_iso = datetime.fromtimestamp(epoch, tz=timezone.utc).replace(
                microsecond=0
            )
            created_at_iso = created_at_iso.isoformat().replace("+00:00", "Z")
    else:
        created_at_iso = DEFAULT_CREATED_AT_UTC

    regen_cmd = _regen_command([sys.argv[0]])

    # Build a tiny canonical audio payload (base64_wav).
    wav_bytes = _silence_wav_mono_16bit_bytes(sample_rate=16000, duration_sec=0.10)
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # Write slice.json (minimal; synthetic provenance).
    examples = _build_examples(out_dir)
    slice_obj = {
        "dataset_id": "beans_zero",
        "split": "test",
        "selection_method": "explicit_sample_ids",
        "audio_strategy": {"type": "vendored_base64_wav"},
        "samples": [
            {
                "sample_id": ex.sample_id,
                "source": {"synthetic": True, "task_id": ex.task_id},
                "task": {"eval_task_id": ex.task_id},
                "labels": ex.labels,
            }
            for ex in examples
        ],
    }
    _write_json(out_dir / "inputs" / "slice.json", slice_obj)

    # Write requests.jsonl (request items only; outer wrapper is constructed by client).
    req_rows: list[dict[str, Any]] = []
    for ex in examples:
        req_rows.append(
            {
                "sample_id": ex.sample_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "<Audio><AudioHere></Audio>\n"
                        + ex.metadata.get("instruction_text", ""),
                    },
                ],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": wav_b64,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 16, "temperature": 0.0},
            }
        )
    n_rows = _write_jsonl(out_dir / "inputs" / "requests.jsonl", req_rows)

    # Write manifest.yaml + required placeholder dirs/files.
    _write_manifest_yaml(
        out_dir,
        created_at_utc=created_at_iso,
        regenerate_command=regen_cmd,
        audio_payload_type="base64_wav",
    )

    _write_text(
        out_dir / "inputs" / "audio" / "README.md",
        "Audio is vendored inline in `inputs/requests.jsonl` as `base64_wav`.\n",
    )

    _write_text(
        out_dir / "expected" / "README.md",
        "\n".join(
            [
                "# Golden outputs (Phase B) — pending",
                "",
                "This bundle is currently Phase A (inputs only).",
                "A GPU machine should populate `expected/*` by running BEANS-Next",
                "against a real launcher",
                "with the exact inputs in `inputs/requests.jsonl`.",
                "",
            ]
        ),
    )
    _write_text(out_dir / "expected" / "predictions.jsonl", "")
    _write_text(out_dir / "expected" / "processed_predictions.jsonl", "")
    _write_text(out_dir / "expected" / "scored_predictions.jsonl", "")
    _write_text(out_dir / "expected" / "summary.json", "{}\n")
    _write_text(out_dir / "expected" / "model_identity.json", "{}\n")

    # Metadata (best-effort, stable).
    _write_json(
        out_dir / "metadata" / "versions.json",
        {
            "beans_next": {
                "git_sha": os.environ.get("GIT_SHA"),
                "package_version": os.environ.get("BEANS_PRO_VERSION"),
            },
            "schemas": {"predictions_wire_schema": "predictions_v1"},
        },
    )
    _write_text(
        out_dir / "metadata" / "provenance.md",
        "\n".join(
            [
                "# Provenance",
                "",
                "This is a synthetic, tiny fixture slice",
                "for CI validation of BEANS-Next.",
                "It does not include HuggingFace downloads.",
                "It does not include model outputs.",
                "",
            ]
        )
        + "\n",
    )
    _write_text(
        out_dir / "metadata" / "regeneration.md",
        "\n".join(
            [
                "# Regeneration",
                "",
                "Phase A (CPU-only):",
                "",
                "```bash",
                regen_cmd,
                "```",
                "",
                "Phase B (GPU): populate `expected/*` by running a real launcher and",
                "capturing outputs.",
                "",
            ]
        )
        + "\n",
    )

    _write_text(
        out_dir / "README.md",
        "\n".join(
            [
                "# beans_zero_slice_v1 (synthetic)",
                "",
                "This directory is a tiny, deterministic fixture bundle intended",
                "for later",
                "golden-run regression tests. It is CPU-only and contains no",
                "HuggingFace",
                "downloads and no model outputs.",
                "",
                "Regenerate with:",
                "",
                f"```bash\n{regen_cmd}\n```",
                "",
            ]
        )
        + "\n",
    )

    # Verify after writing.
    _verify_bundle(out_dir)

    # Optional small status line for humans.
    print(f"Wrote fixture bundle to {out_dir} with {n_rows} request item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
