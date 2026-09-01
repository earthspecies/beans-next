"""Download exact text-model snapshots from Hugging Face in a CPU job."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    """Resolve model revisions, download them, and write a manifest.

    Returns
    -------
    int
        Zero after all requested snapshots are complete.
    """
    args = _parser().parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    models: list[dict[str, str]] = []

    for model_id in args.model:
        info = api.model_info(model_id)
        revision = str(info.sha)
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=args.cache_dir,
            max_workers=max(1, args.workers),
        )
        models.append(
            {
                "model_id": model_id,
                "resolved_revision": revision,
                "snapshot_path": str(Path(snapshot_path).resolve()),
            }
        )

    payload = {
        "schema_version": "beans_next.text_model_snapshots.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "models": models,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
