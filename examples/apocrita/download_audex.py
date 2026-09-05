"""Download the NVIDIA Audex 30B-A3B snapshot in a CPU job.

Skips the audio-generation and text-only checkpoint variants shipped in the
same repo (the model card documents ``checkpoint_folder_full`` as the
understanding checkpoint this launcher serves), which shrinks the download
from 71.7 GB to roughly 67 GB. After downloading, verifies every safetensors
shard listed in each checkpoint's ``*.safetensors.index.json`` actually
landed on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL_ID = "nvidia/Nemotron-Labs-Audex-30B-A3B"

# Directories not needed to serve audio understanding via vLLM.
_SKIP_PATTERNS = [
    "enhancement_VAE/*",
    "audex_causal_speech_decoder/*",
    "checkpoint_folder_audiogen/*",
    "checkpoint_folder_textonly/*",
    "assets/*",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def _verify_safetensors_shards(snapshot_path: Path) -> list[str]:
    """Return a list of missing safetensors shard paths (empty if complete).

    Parameters
    ----------
    snapshot_path
        Local snapshot directory to check.

    Returns
    -------
    list[str]
        Relative paths (to ``snapshot_path``) of shards referenced by an
        index file but missing on disk.
    """

    missing: list[str] = []
    for index_path in snapshot_path.rglob("*.safetensors.index.json"):
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        shard_dir = index_path.parent
        shard_names = sorted(set(weight_map.values()))
        for shard_name in shard_names:
            shard_path = shard_dir / shard_name
            if not shard_path.is_file():
                missing.append(str(shard_path.relative_to(snapshot_path)))
    return missing


def main() -> int:
    """Download the Audex snapshot and verify shard completeness.

    Returns
    -------
    int
        Zero on success, non-zero if any safetensors shard is missing.

    Raises
    ------
    SystemExit
        If any referenced safetensors shard is missing after download.
    """
    args = _parser().parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(MODEL_ID)
    revision = str(info.sha)

    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            cache_dir=args.cache_dir,
            max_workers=max(1, args.workers),
            ignore_patterns=_SKIP_PATTERNS,
        )
    ).resolve()

    missing = _verify_safetensors_shards(snapshot_path)
    if missing:
        for path in missing:
            print(f"MISSING SHARD: {path}")
        raise SystemExit(f"{len(missing)} safetensors shard(s) missing after download")

    payload = {
        "schema_version": "beans_next.audex_snapshot.v1",
        "model_id": MODEL_ID,
        "resolved_revision": revision,
        "snapshot_path": str(snapshot_path),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.manifest)
    print(f"snapshot_path={snapshot_path}")
    print(f"resolved_revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
