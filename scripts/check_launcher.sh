#!/bin/sh
# BEANS-Next launcher conformance: predictions_v1 + /info + /health (DESIGN §4).
# Usage: scripts/check_launcher.sh <base_url>
# Example: scripts/check_launcher.sh http://127.0.0.1:8000
set -eu

usage() {
    printf '%s\n' "usage: $0 <base_url>" "  Example: $0 http://127.0.0.1:8000" >&2
    exit 2
}

[ "${1-}" ] || usage
BASE_URL_RAW=$1

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if command -v uv >/dev/null 2>&1 && [ -f "$REPO_ROOT/pyproject.toml" ]; then
    (cd "$REPO_ROOT" && uv run python "$SCRIPT_DIR/check_launcher.py" "$BASE_URL_RAW")
else
    python3 "$SCRIPT_DIR/check_launcher.py" "$BASE_URL_RAW"
fi
