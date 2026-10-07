#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec uv run uvicorn serve:app --host "${NATURELM_BIND_HOST:-127.0.0.1}" --port "${PORT:-8001}"
