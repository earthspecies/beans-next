#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_run_dir.sh results/ingested/<run_id>

Validates that a BEANS-Next artifacts directory is usable for:

- Offline rescoring input (copied cluster run dir):
  - must contain `predictions.jsonl` and a sibling `processed_predictions.jsonl` with targets
- Offline rescoring output (a `score-from-file -o <output_dir>` directory):
  - typically does NOT contain `predictions.jsonl`, but should contain rescored artifacts like
    `processed_predictions.jsonl`, `scored_predictions.jsonl`, and `summary.json`

This validator uses cheap string checks (not full JSON parsing) to fast-fail on the first
missing required file/content.

This script also supports validating a "run root" directory that contains one or more
per-model/per-run subdirectories (common when copying `results/<model>/<run_id>/` trees
from a cluster). In that case it will validate each discovered leaf run dir and fail on
the first invalid one.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

run_dir="$1"

warnings=0

info() {
  echo "INFO: $*" >&2
}

warn() {
  echo "WARN: $*" >&2
  warnings=$((warnings + 1))
}

die() {
  echo "FAIL: $*" >&2
  exit 1
}

require_nonempty_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    die "missing required file: $path"
  fi
  if [[ ! -f "$path" ]]; then
    die "required path is not a file: $path"
  fi
  if [[ ! -s "$path" ]]; then
    die "required file is empty: $path"
  fi

  # `wc -c` reads the whole file; `stat` is constant-time and keeps this validator fast.
  local bytes
  bytes="$(stat -c '%s' "$path" 2>/dev/null || true)"
  if [[ -n "$bytes" ]]; then
    echo "OK: $path (${bytes} bytes)"
  else
    echo "OK: $path"
  fi
}

optional_file() {
  local path="$1"
  if [[ -e "$path" && -s "$path" && -f "$path" ]]; then
    echo "OK: $path (present)"
  else
    echo "INFO: optional file missing or empty: $path"
  fi
}

is_offline_rescore_input_leaf_dir() {
  local dir="$1"
  [[ -f "$dir/predictions.jsonl" ]]
}

is_offline_rescore_output_leaf_dir() {
  local dir="$1"
  [[ -f "$dir/processed_predictions.jsonl" && -f "$dir/scored_predictions.jsonl" && -f "$dir/summary.json" ]]
}

discover_leaf_run_dirs() {
  local root="$1"
  local -a dirs=()

  if is_offline_rescore_input_leaf_dir "$root" || is_offline_rescore_output_leaf_dir "$root"; then
    dirs+=("$root")
    printf '%s\n' "${dirs[@]}"
    return 0
  fi

  # Common layouts:
  # - results/<run_id>/... (user passes leaf dir) -> handled above
  # - results/<model>/<run_id>/... (user passes results/ or ingested root)
  # - results/<cluster_bundle>/<model>/<run_id>/... (nested bundle)
  shopt -s nullglob
  local d
  for d in "$root"/* "$root"/*/*; do
    if [[ -d "$d" ]]; then
      if is_offline_rescore_input_leaf_dir "$d" || is_offline_rescore_output_leaf_dir "$d"; then
        dirs+=("$d")
      fi
    fi
  done
  shopt -u nullglob

  printf '%s\n' "${dirs[@]}"
}

require_contains() {
  local path="$1"
  local needle="$2"
  local hint="$3"

  if ! grep -a -m 1 -q -- "$needle" "$path"; then
    die "required content not found in $path: $hint. Fix: recopy the run directory, or copy the correct artifact file(s) from the original run output."
  else
    echo "OK: $path contains ${hint}"
  fi
}

require_line_matching() {
  local path="$1"
  local pattern="$2"
  local hint="$3"
  local fix="$4"

  if ! grep -a -m 1 -q -E -- "$pattern" "$path"; then
    die "required content not found in $path: $hint. Fix: ${fix}"
  else
    echo "OK: $path contains ${hint}"
  fi
}

warn_if_missing_content() {
  local path="$1"
  local needle="$2"
  local hint="$3"

  if ! grep -a -m 1 -q -- "$needle" "$path"; then
    warn "missing expected content in $path: $hint. This may be fine depending on your run, but can reduce the usefulness of offline inspection."
  else
    echo "OK: $path contains ${hint}"
  fi
}

validate_leaf_dir() {
  local leaf_dir="$1"

  if is_offline_rescore_input_leaf_dir "$leaf_dir"; then
    echo "Validating offline-rescore INPUT directory: $leaf_dir"

    require_nonempty_file "$leaf_dir/predictions.jsonl"
    if [[ ! -e "$leaf_dir/processed_predictions.jsonl" ]]; then
      die "missing required file: $leaf_dir/processed_predictions.jsonl (required for offline rescoring; copy it from the original 'beans-next run' output alongside predictions.jsonl)"
    fi
    require_nonempty_file "$leaf_dir/processed_predictions.jsonl"

    # Basic shape checks. These are cheap string checks (not full JSON parsing).
    require_line_matching \
      "$leaf_dir/predictions.jsonl" \
      '^[[:space:]]*{' \
      "at least one JSON object line (JSONL rows should be JSON objects)" \
      "recopy the run directory; if your harness wrote logs/traces into the file, regenerate artifacts using BEANS-Next's standard output writer"

    require_line_matching \
      "$leaf_dir/processed_predictions.jsonl" \
      '^[[:space:]]*{' \
      "at least one JSON object line (JSONL rows should be JSON objects)" \
      "recopy the run directory; if the file is truncated/corrupted, copy it again from the original run output"

    require_line_matching \
      "$leaf_dir/predictions.jsonl" \
      '"predictions"[[:space:]]*:[[:space:]]*\[' \
      'at least one row with `"predictions": [...]` (expected list shape)' \
      "ensure your launcher responses are being captured correctly; if this was a partial/failed run, re-run inference and recopy artifacts"

    require_contains \
      "$leaf_dir/predictions.jsonl" \
      '"sample_id"' \
      'at least one `"sample_id"` field'

    # `score-from-file` does not reload datasets; it needs targets from the sidecar.
    require_contains \
      "$leaf_dir/processed_predictions.jsonl" \
      '"sample_id"' \
      'at least one `"sample_id"` field (targets sidecar is keyed by sample_id)'

    require_contains \
      "$leaf_dir/processed_predictions.jsonl" \
      '"targets"' \
      'at least one `"targets"` field (required for offline scoring)'

    require_line_matching \
      "$leaf_dir/processed_predictions.jsonl" \
      '"targets"[[:space:]]*:[[:space:]]*(\[|\{|")' \
      'at least one row with non-null `"targets"` (expected list/dict/string shape)' \
      "recopy the run directory; if targets are missing in the original run, the cluster run did not complete scoring successfully"

    warn_if_missing_content \
      "$leaf_dir/processed_predictions.jsonl" \
      '"task_id"' \
      'at least one `"task_id"` field (helps with per-task summaries; not always required)'

    optional_file "$leaf_dir/run_config.yaml"
    optional_file "$leaf_dir/summary.json"
    optional_file "$leaf_dir/model_identity.json"
    optional_file "$leaf_dir/checkpoint.json"
    return 0
  fi

  if is_offline_rescore_output_leaf_dir "$leaf_dir"; then
    echo "Validating offline-rescore OUTPUT directory (no predictions.jsonl expected): $leaf_dir"

    require_nonempty_file "$leaf_dir/processed_predictions.jsonl"
    require_nonempty_file "$leaf_dir/scored_predictions.jsonl"
    require_nonempty_file "$leaf_dir/summary.json"

    require_line_matching \
      "$leaf_dir/processed_predictions.jsonl" \
      '^[[:space:]]*{' \
      "at least one JSON object line in processed_predictions.jsonl" \
      "re-run 'beans-next score-from-file' or recopy the output directory"

    require_line_matching \
      "$leaf_dir/scored_predictions.jsonl" \
      '^[[:space:]]*{' \
      "at least one JSON object line in scored_predictions.jsonl" \
      "re-run 'beans-next score-from-file' or recopy the output directory"

    require_line_matching \
      "$leaf_dir/summary.json" \
      '^[[:space:]]*{' \
      "summary.json looks like a JSON object" \
      "re-run 'beans-next score-from-file' or recopy the output directory"

    require_contains \
      "$leaf_dir/processed_predictions.jsonl" \
      '"sample_id"' \
      'at least one `"sample_id"` field'

    require_contains \
      "$leaf_dir/processed_predictions.jsonl" \
      '"targets"' \
      'at least one `"targets"` field'

    warn_if_missing_content \
      "$leaf_dir/processed_predictions.jsonl" \
      '"task_id"' \
      'at least one `"task_id"` field (helps with per-task summaries; not always required)'

    optional_file "$leaf_dir/model_identity.json"
    optional_file "$leaf_dir/checkpoint.json"
    optional_file "$leaf_dir/run_config.yaml"
    return 0
  fi

  die "directory does not look like a leaf run dir (expected input leaf with predictions.jsonl, or output leaf with processed+scored+summary): $leaf_dir"
}

if [[ ! -e "$run_dir" ]]; then
  die "run directory does not exist: $run_dir"
elif [[ ! -d "$run_dir" ]]; then
  die "path is not a directory: $run_dir"
fi

mapfile -t leaf_dirs < <(discover_leaf_run_dirs "$run_dir")
if [[ ${#leaf_dirs[@]} -eq 0 ]]; then
  die "no leaf run directories found under: $run_dir (expected either predictions.jsonl in this directory, or subdirs like <model>/<run_id>/predictions.jsonl or <run_id>/predictions.jsonl)"
fi

if [[ ${#leaf_dirs[@]} -gt 1 ]]; then
  info "discovered ${#leaf_dirs[@]} leaf run dirs under: $run_dir"
fi

for leaf in "${leaf_dirs[@]}"; do
  validate_leaf_dir "$leaf"
done

echo "PASS: artifacts directory looks usable (warnings=${warnings})."
exit 0

