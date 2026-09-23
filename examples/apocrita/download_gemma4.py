"""Download the ``google/gemma-4-12B-it`` snapshot in a CPU job.

Reuses ``download_text_models.py``'s generic resolve-and-manifest structure
(``HfApi.model_info`` to pin ``info.sha``, ``snapshot_download``, then a JSON
manifest) but adds Audex's ``*.safetensors.index.json`` shard-completeness
check from ``download_audex.py``: Gemma 4 12B is a ~24 GB multimodal
snapshot, and a truncated download should fail here, in a CPU job, rather
than surface as a confusing load failure in a GPU job later.

Unlike Audex, gemma-4-12B-it is not gated and has no checkpoint variants to
skip (no separate audio-generation/text-only checkpoint folders), so there is
no ``--cache-dir``-relative ``ignore_patterns`` list here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL_ID = "google/gemma-4-12B-it"


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
    """Download the Gemma 4 12B snapshot and verify shard completeness.

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
        )
    ).resolve()

    missing = _verify_safetensors_shards(snapshot_path)
    if missing:
        for path in missing:
            print(f"MISSING SHARD: {path}")
        raise SystemExit(f"{len(missing)} safetensors shard(s) missing after download")

    payload = {
        "schema_version": "beans_next.gemma4_snapshot.v1",
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
