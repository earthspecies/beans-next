#!/usr/bin/env bash
# Start the BEANS-Next NatureLM-audio 1.1 launcher (FastAPI + uvicorn).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8001}"
export PORT

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage:"
  echo "  PORT=8001 ./serve.sh"
  echo ""
  echo "Options:"
  echo "  --check-access   verify checkpoint availability, then exit"
  echo ""
  echo "                   Behavior:"
  echo "                   - If NATURELM_GCS_CHECKPOINT_URI is set: verify GCS objects exist (HF-free)."
  echo "                   - Else: verify HF_TOKEN can access the gated HuggingFace repo."
  echo ""
  echo "Environment:"
  echo "  HF_TOKEN                 required (unless NATURELM_STUB_MODE=1)"
  echo "  NATURELM_GCS_CHECKPOINT_URI  optional GCS checkpoint prefix for HF-free mode"
  echo "  NATURELM_STUB_MODE=1     conformance-only mode (no HF, no weights)"
  echo "  NATURELM_HF_REPO_ID      override model repo (default: EarthSpeciesProject/naturelm-audio-1.1.00-private)"
  echo "  NATURELM_HF_REVISION     override revision (default: main)"
  echo "  NATURELM_BIND_HOST       bind address (default: 127.0.0.1)"
  exit 0
fi

if [[ "${1:-}" == "--check-access" ]]; then
  python3 - <<'PY'
import os
import sys

gcs_uri = os.environ.get("NATURELM_GCS_CHECKPOINT_URI", "").strip()
if gcs_uri:
    if not gcs_uri.startswith("gs://"):
        print(f"NATURELM_GCS_CHECKPOINT_URI must start with gs://, got: {gcs_uri!r}", file=sys.stderr)
        sys.exit(2)
    try:
        import gcsfs  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        print(
            "HF-free --check-access requires gcsfs to be installed in the launcher venv. "
            f"Import failed: {e!r}",
            file=sys.stderr,
        )
        sys.exit(3)
    fs = gcsfs.GCSFileSystem()
    prefix = gcs_uri.removeprefix("gs://")
    objects = fs.find(prefix)
    if not objects:
        print(f"no objects found under GCS checkpoint uri: {gcs_uri!r}", file=sys.stderr)
        sys.exit(4)
    print(f"ok: GCS checkpoint available under {gcs_uri!r} (n_objects={len(objects)})")
    sys.exit(0)

# Fallback: HuggingFace access verification (requires huggingface_hub).
try:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
except Exception as e:  # noqa: BLE001
    print(
        "Neither NATURELM_GCS_CHECKPOINT_URI is set nor huggingface_hub can be imported. "
        f"Cannot run --check-access real-mode check. Import error: {e!r}",
        file=sys.stderr,
    )
    sys.exit(5)

repo_id = os.environ.get("NATURELM_HF_REPO_ID", "EarthSpeciesProject/naturelm-audio-1.1.00-private")
rev = os.environ.get("NATURELM_HF_REVISION", "main")
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    print(
        "HF_TOKEN is not set. Export HF_TOKEN with access to "
        f"{repo_id!r}, or set NATURELM_GCS_CHECKPOINT_URI for HF-free checking.",
        file=sys.stderr,
    )
    sys.exit(2)

api = HfApi()
try:
    info = api.model_info(repo_id=repo_id, revision=rev, token=token)
except GatedRepoError:
    print(
        f"HF_TOKEN lacks access to gated repo {repo_id!r}. "
        "Request access on the model page (HuggingFace) and wait for approval.",
        file=sys.stderr,
    )
    sys.exit(3)
except HfHubHTTPError as e:
    status = getattr(getattr(e, "response", None), "status_code", None)
    print(
        f"HF_TOKEN is invalid or unauthorized for {repo_id!r} (HTTP {status}).",
        file=sys.stderr,
    )
    sys.exit(4)
except Exception as e:
    print(f"failed to reach HuggingFace for {repo_id!r}: {e}", file=sys.stderr)
    sys.exit(5)

sha = getattr(info, "sha", None)
print(f"ok: access verified for {repo_id!r} @ {sha or rev}")
PY
  exit 0
fi

PY="${PYTHON:-python3}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$PY" -m uvicorn serve:app --host "${NATURELM_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  exec "$PY" -m uvicorn serve:app --host "${NATURELM_BIND_HOST:-127.0.0.1}" --port "$PORT"
fi

echo "No active venv and no $ROOT/.venv found." >&2
echo "Create one and install deps, e.g.:" >&2
echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
echo "Or from repo root with uv:" >&2
echo "  cd $ROOT && uv venv && uv pip install -r requirements.txt && . .venv/bin/activate && ./serve.sh" >&2
exit 1

