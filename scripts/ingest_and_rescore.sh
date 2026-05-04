#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/ingest_and_rescore.sh <run_dir>

CPU-only helper for offline ingestion + rescoring after copying a cluster run dir back.

What it does (stop-on-error):
  1) Validates the run dir fast-fails:
     scripts/validate_run_dir.sh <run_dir>
  2) Offline-rescores predictions into a sibling output directory:
     uv run beans-next score-from-file <run_dir>/predictions.jsonl -o <run_dir>__rescored

Notes / prerequisites:
  - Requires `uv` on PATH and the repo venv synced (run `uv sync` once).
  - Offline rescoring requires a sibling `processed_predictions.jsonl` next to `predictions.jsonl`
    in <run_dir>. This sidecar must contain `sample_id` and non-null `targets`.

EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ $# -ne 1 ]; then
  usage >&2
  exit 2
fi

run_dir="$1"

die() {
  echo "FAIL: $*" >&2
  exit 1
}

info() {
  echo "INFO: $*" >&2
}

if [ ! -e "$run_dir" ]; then
  die "run dir does not exist: $run_dir"
fi
if [ ! -d "$run_dir" ]; then
  die "path is not a directory: $run_dir"
fi

run_dir="$(cd -- "$run_dir" && pwd -P)"

predictions_jsonl="$run_dir/predictions.jsonl"
sidecar_processed="$run_dir/processed_predictions.jsonl"
out_dir="${run_dir%/}__rescored"

if [ ! -e "$predictions_jsonl" ]; then
  die "missing required file: $predictions_jsonl"
fi
if [ ! -s "$predictions_jsonl" ]; then
  die "required file is empty: $predictions_jsonl"
fi
if [ ! -e "$sidecar_processed" ]; then
  die "missing required file: $sidecar_processed (required targets sidecar for offline rescoring; copy it alongside predictions.jsonl)"
fi
if [ ! -s "$sidecar_processed" ]; then
  die "required file is empty: $sidecar_processed"
fi

if ! command -v uv >/dev/null 2>&1; then
  die "uv is not installed or not on PATH (required). Install uv and run `uv sync` in this repo."
fi

info "Validating run directory: $run_dir"
bash "${script_dir}/validate_run_dir.sh" "$run_dir"

if [ "${BEANS_PRO_SKIP_RESCORE:-0}" = "1" ]; then
  info "BEANS_PRO_SKIP_RESCORE=1 set; skipping offline rescoring step"
  echo "PASS: input run directory validated (rescore skipped): $run_dir"
  exit 0
fi

info "Offline rescoring: $predictions_jsonl -> $out_dir"
(cd -- "$repo_root" && uv run beans-next score-from-file "$predictions_jsonl" -o "$out_dir")

info "Validating rescored output directory: $out_dir"
bash "${script_dir}/validate_run_dir.sh" "$out_dir"

echo "PASS: rescored outputs written to: $out_dir"

