#!/usr/bin/env bash
# Sync a *full* beans-next run directory to GCS (recursive), matching
# examples/slurm/run_inference.sh.
#
# Use this to backfill GCS when artifacts were only partially copied (e.g. a lone
# summary.json) or when upload was skipped (BEANS_PRO_UPLOAD_GCS=0, missing gsutil).
#
# Do *not* use `beans_next.results.upload_run_artifacts` for suite runs: it only
# uploads a flat set of files in one directory, not suite/<suite>/<task>/ trees.
#
# Usage:
#   # Full tree under NFS ingested (set path prefix manually — must match prior sweep layout):
#   LOCAL_SRC=$HOME/beans-next-results/ingested/adhoc/my_run \
#     BEANS_PRO_GCS_REL_PATH=adhoc/my_run \
#     bash scripts/sync_beans_next_results_to_gcs.sh
#
#   # Tree still under scratch cache (same layout as run_inference.sh — rel path auto):
#   LOCAL_SRC=/scratch/$USER/.cache/beans-next-results/i29/naturelm_v1_0/beans_zero_core/run_id \
#     bash scripts/sync_beans_next_results_to_gcs.sh
#
# Requires: gsutil (Google Cloud SDK), authenticated for the bucket.

set -euo pipefail

LOCAL_SRC="${LOCAL_SRC:-}"
if [[ -z "$LOCAL_SRC" ]]; then
  echo "ERROR: Set LOCAL_SRC to the root directory of one run (contains suite/ or JSONL trees)." >&2
  exit 1
fi

if [[ ! -d "$LOCAL_SRC" ]]; then
  echo "ERROR: LOCAL_SRC is not a directory: $LOCAL_SRC" >&2
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "ERROR: gsutil not found; install Google Cloud SDK." >&2
  exit 1
fi

USER="${USER:-$(id -un)}"
SCRATCH_ROOT="/scratch/${USER}/.cache/beans-next-results/"
GCS_PREFIX_BASE="${BEANS_PRO_GCS_PREFIX_BASE:-gs://foundation-model-data/synthetic/predictions/beans-next-results}"

rel_path="${BEANS_PRO_GCS_REL_PATH:-}"
if [[ -z "$rel_path" ]]; then
  # Mirror examples/slurm/run_inference.sh
  if [[ "${LOCAL_SRC%/}/" == "${SCRATCH_ROOT}"* ]]; then
    rel_path="${LOCAL_SRC#${SCRATCH_ROOT}}"
    rel_path="${rel_path%/}"
  else
    echo "ERROR: LOCAL_SRC is not under ${SCRATCH_ROOT}" >&2
    echo "Set BEANS_PRO_GCS_REL_PATH to the bucket suffix (under beans-next-results/), e.g. i29/model/suite/run_id" >&2
    exit 1
  fi
fi

GCS_DEST="${GCS_PREFIX_BASE%/}/${rel_path}/"
echo "Local:  $LOCAL_SRC"
echo "GCS:    $GCS_DEST"
echo "Running: gsutil -m rsync -r ${LOCAL_SRC%/}/ ${GCS_DEST}"
gsutil -m rsync -r "${LOCAL_SRC%/}/" "${GCS_DEST}"
echo "Done."
