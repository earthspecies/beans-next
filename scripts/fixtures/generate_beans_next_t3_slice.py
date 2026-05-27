"""Generate a tiny deterministic BEANS-Next T3 fixture slice (CPU-only).

This script writes a self-contained fixture bundle covering all 21 Tier-3 task
types. It does not download any datasets and does not run model inference.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
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
BUNDLE_ID = "beans_next_t3_slice_v1"

# One representative instruction per T3 subset.
_T3_EXAMPLES: list[tuple[str, str, str, str]] = [
    # (subset, eval_task_id, label, instruction)
    (
        "t3-vocalization-presence-binary",
        "beans_next_t3_vocalization_presence_binary",
        "Yes",
        "<Audio><AudioHere></Audio> Is there a bird vocalizing in this recording? "
        "Answer Yes or No.",
    ),
    (
        "t3-vocalization-cooccurrence-binary",
        "beans_next_t3_vocalization_cooccurrence_binary",
        "No",
        "<Audio><AudioHere></Audio> Are Thrush nightingale and Common chaffinch "
        "vocalizing at the same time? Answer Yes or No.",
    ),
    (
        "t3-species-count-oe",
        "beans_next_t3_species_count_oe",
        "2",
        "<Audio><AudioHere></Audio> How many different bird species can you hear "
        "in this recording? Answer with just a number.",
    ),
    (
        "t3-vocalization-count-total-oe",
        "beans_next_t3_vocalization_count_total_oe",
        "5",
        "<Audio><AudioHere></Audio> How many bird vocalizations are there in "
        "total in this recording? Answer with just a number.",
    ),
    (
        "t3-vocalization-count-total-mcq",
        "beans_next_t3_vocalization_count_total_mcq",
        "(A) 3",
        "<Audio><AudioHere></Audio> How many bird vocalizations are there in "
        "total? (A) 3 (B) 5 (C) 7 (D) 10. Answer with just the letter.",
    ),
    (
        "t3-vocalization-count-per-species-oe",
        "beans_next_t3_vocalization_count_per_species_oe",
        "Thrush nightingale: 3, Common chaffinch: 2",
        "<Audio><AudioHere></Audio> How many times does each species vocalize? "
        "List each species and its count.",
    ),
    (
        "t3-vocalization-referring-mcq",
        "beans_next_t3_vocalization_referring_mcq",
        "(B) Common chaffinch",
        "<Audio><AudioHere></Audio> Which bird is vocalizing in the second "
        "vocalization? (A) Thrush nightingale (B) Common chaffinch "
        "(C) Eurasian blackbird. Answer with just the letter.",
    ),
    (
        "t3-species-by-vocalization-order-oe",
        "beans_next_t3_species_by_vocalization_order_oe",
        "Thrush nightingale, Common chaffinch",
        "<Audio><AudioHere></Audio> List the bird species in the order they "
        "first vocalize, separated by commas.",
    ),
    (
        "t3-species-by-vocalization-order-mcq",
        "beans_next_t3_species_by_vocalization_order_mcq",
        "(A) Thrush nightingale, Common chaffinch",
        "<Audio><AudioHere></Audio> In what order do the species first vocalize? "
        "(A) Thrush nightingale then Common chaffinch "
        "(B) Common chaffinch then Thrush nightingale. Answer with just the letter.",
    ),
    (
        "t3-species-by-highest-pitch-oe",
        "beans_next_t3_species_by_highest_pitch_oe",
        "Thrush nightingale",
        "<Audio><AudioHere></Audio> Which species has the highest pitched "
        "vocalization in this recording?",
    ),
    (
        "t3-species-by-highest-pitch-mcq",
        "beans_next_t3_species_by_highest_pitch_mcq",
        "(A) Thrush nightingale",
        "<Audio><AudioHere></Audio> Which species has the highest pitched "
        "vocalization? (A) Thrush nightingale (B) Common chaffinch. "
        "Answer with just the letter.",
    ),
    (
        "t3-species-by-lowest-pitch-oe",
        "beans_next_t3_species_by_lowest_pitch_oe",
        "Common chaffinch",
        "<Audio><AudioHere></Audio> Which species has the lowest pitched "
        "vocalization in this recording?",
    ),
    (
        "t3-species-by-lowest-pitch-mcq",
        "beans_next_t3_species_by_lowest_pitch_mcq",
        "(B) Common chaffinch",
        "<Audio><AudioHere></Audio> Which species has the lowest pitched "
        "vocalization? (A) Thrush nightingale (B) Common chaffinch. "
        "Answer with just the letter.",
    ),
    (
        "t3-species-by-longest-vocalization-oe",
        "beans_next_t3_species_by_longest_vocalization_oe",
        "Thrush nightingale",
        "<Audio><AudioHere></Audio> Which species has the longest individual "
        "vocalization in this recording?",
    ),
    (
        "t3-species-by-longest-vocalization-mcq",
        "beans_next_t3_species_by_longest_vocalization_mcq",
        "(A) Thrush nightingale",
        "<Audio><AudioHere></Audio> Which species has the longest individual "
        "vocalization? (A) Thrush nightingale (B) Common chaffinch. "
        "Answer with just the letter.",
    ),
    (
        "t3-species-by-vocalization-frequency-oe",
        "beans_next_t3_species_by_vocalization_frequency_oe",
        "Common chaffinch",
        "<Audio><AudioHere></Audio> Which species vocalizes most frequently "
        "in this recording?",
    ),
    (
        "t3-species-by-vocalization-frequency-mcq",
        "beans_next_t3_species_by_vocalization_frequency_mcq",
        "(B) Common chaffinch",
        "<Audio><AudioHere></Audio> Which species vocalizes most frequently? "
        "(A) Thrush nightingale (B) Common chaffinch. "
        "Answer with just the letter.",
    ),
    (
        "t3-structural-captioning",
        "beans_next_t3_structural_captioning",
        "Two birds vocalize. Thrush nightingale vocalizes first at high pitch.",
        "<Audio><AudioHere></Audio> Describe the structure of the bird "
        "vocalizations in this recording.",
    ),
    (
        "t3-species-listing-open-list",
        "beans_next_t3_species_listing_open_list",
        "Thrush nightingale, Common chaffinch",
        "<Audio><AudioHere></Audio> List all the bird species you can hear "
        "in this recording, separated by commas.",
    ),
    (
        "t3-frequency-range-description",
        "beans_next_t3_frequency_range_description",
        "3140 Hz",
        "<Audio><AudioHere></Audio> What is the dominant frequency of the "
        "bird vocalizations in this recording? Give a value in Hz.",
    ),
    (
        "t3-ordered-species-summary",
        "beans_next_t3_ordered_species_summary",
        "Thrush nightingale, Common chaffinch",
        "<Audio><AudioHere></Audio> Summarize the bird vocalizations in this "
        "recording in order of first occurrence.",
    ),
]


@dataclass(frozen=True)
class FixtureExample:
    """Synthetic fixture row for the T3 slice bundle.

    Notes
    -----
    Intentionally minimal — no HuggingFace downloads, no real audio.
    Exists only to generate stable ``inputs/slice.json`` and
    ``inputs/requests.jsonl`` for CPU-only fixture validation.
    """

    sample_id: str
    task_id: str
    subset: str
    split: str
    labels: str | list[str] | None
    instruction: str

    def to_slice_sample(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": {"synthetic": True, "task_id": self.task_id},
            "task": {"eval_task_id": self.task_id},
            "labels": self.labels,
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

    raw_pcm = (b"\x00\x00") * n_frames

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def _regen_command(argv: list[str]) -> str:
    script = Path(argv[0]).as_posix()
    return f"uv run python {script} --out tests/fixtures/{BUNDLE_ID} --force"


def _build_examples() -> list[FixtureExample]:
    return [
        FixtureExample(
            sample_id=f"fixture:{BUNDLE_ID}:{subset}:0000",
            task_id=task_id,
            subset=subset,
            split="test",
            labels=label,
            instruction=instruction,
        )
        for subset, task_id, label, instruction in _T3_EXAMPLES
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
        description: "Synthetic BEANS-Next Tier-3 slice — CPU-only, all 21 T3 tasks."
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
        help="Output fixture directory (e.g. tests/fixtures/beans_next_t3_slice_v1)",
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
        try:
            epoch = int(created_at)
        except ValueError:
            created_at_iso = str(created_at)
        else:
            created_at_iso = datetime.fromtimestamp(epoch, tz=timezone.utc).replace(
                microsecond=0
            )
            created_at_iso = created_at_iso.isoformat().replace("+00:00", "Z")
    else:
        created_at_iso = DEFAULT_CREATED_AT_UTC

    regen_cmd = _regen_command([sys.argv[0]])

    wav_bytes = _silence_wav_mono_16bit_bytes(sample_rate=16000, duration_sec=0.10)
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

    examples = _build_examples()

    slice_obj = {
        "dataset_id": "beans_next_t3",
        "split": "test",
        "selection_method": "explicit_sample_ids",
        "audio_strategy": {"type": "vendored_base64_wav"},
        "samples": [ex.to_slice_sample() for ex in examples],
    }
    _write_json(out_dir / "inputs" / "slice.json", slice_obj)

    req_rows: list[dict[str, Any]] = []
    for ex in examples:
        req_rows.append(
            {
                "sample_id": ex.sample_id,
                "messages": [
                    {
                        "role": "user",
                        "content": ex.instruction,
                    },
                ],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": wav_b64,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 64, "temperature": 0.0},
            }
        )
    n_rows = _write_jsonl(out_dir / "inputs" / "requests.jsonl", req_rows)

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
                "Synthetic, tiny Tier-3 fixture slice for BEANS-Next CI validation.",
                "Covers all 21 T3 task types (one example per task).",
                "No HuggingFace downloads. No model outputs.",
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
                "# beans_next_t3_slice_v1 (synthetic)",
                "",
                "Deterministic fixture bundle for BEANS-Next Tier-3 CI validation.",
                "Covers all 21 T3 task types with one synthetic example each.",
                "CPU-only — no HuggingFace downloads, no model outputs.",
                "",
                "Regenerate with:",
                "",
                f"```bash\n{regen_cmd}\n```",
                "",
            ]
        )
        + "\n",
    )

    _verify_bundle(out_dir)

    print(f"Wrote fixture bundle to {out_dir} with {n_rows} request item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
