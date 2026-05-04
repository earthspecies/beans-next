#!/usr/bin/env bash
# SLURM job: merge node-local scratch beans-next results into NFS ($HOME) and mirror to GCS.
#
# Scratch under /scratch/$USER/.cache/beans-next-results/ is **node-local**. Results from
# runs on one CPU box only exist there until copied. Submit once per node that may hold data.
# Replace NODE with your site's CPU hostname(s), e.g.:
#
#   sbatch --nodelist=slurm-cpu-48vcpu-384gb-1 examples/slurm/rsync_scratch_results_to_nfs_and_gcs.sh
#   sbatch --nodelist=slurm-cpu-48vcpu-384gb-2 examples/slurm/rsync_scratch_results_to_nfs_and_gcs.sh
#
# Discover names: sinfo -N -p cpu -o '%N %t'
#
# Environment (optional):
#   BEANS_PRO_SCRATCH_RESULTS   Root on scratch (default: /scratch/$USER/.cache/beans-next-results)
#   BEANS_PRO_RESULTS_HOME_DIR  NFS merge target (default: $HOME/beans-next-results/ingested)
#   BEANS_PRO_UPLOAD_GCS        1=gsutil rsync to GCS after NFS merge (default: 1)
#   BEANS_PRO_GCS_PREFIX_BASE   Default gs://foundation-model-data/synthetic/predictions/beans-next-results
#   BEANS_PRO_RSYNC_DELETE      If 1, rsync to NFS uses --delete (dangerous; default 0)
#
# Does **not** delete scratch unless you add a separate cleanup step.

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --output="/home/%u/logs/%x-%j-%N.log"
#SBATCH --job-name="beans-next rsync scratch→nfs+gcs"

set -euo pipefail

USER="${USER:-$(id -un)}"
HOST="${SLURMD_NODENAME:-$(hostname -s)}"

SCRATCH_ROOT="${BEANS_PRO_SCRATCH_RESULTS:-/scratch/${USER}/.cache/beans-next-results}"
DEST_ROOT="${BEANS_PRO_RESULTS_HOME_DIR:-${HOME}/beans-next-results/ingested}"
GCS_PREFIX_BASE="${BEANS_PRO_GCS_PREFIX_BASE:-gs://foundation-model-data/synthetic/predictions/beans-next-results}"
UPLOAD_GCS="${BEANS_PRO_UPLOAD_GCS:-1}"
RSYNC_DELETE="${BEANS_PRO_RSYNC_DELETE:-0}"

_step() {
  echo "$(date -Is 2>/dev/null || date) [beans-next][rsync] $*"
}

_step "host=${HOST} SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

if [[ ! -d "$SCRATCH_ROOT" ]]; then
  _step "WARNING: scratch root missing or empty: $SCRATCH_ROOT"
  exit 0
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync not found" >&2
  exit 1
fi

mkdir -p "$DEST_ROOT"

RSYNC_FLAGS=(-av)
if [[ "$RSYNC_DELETE" == "1" ]]; then
  RSYNC_FLAGS+=(--delete)
  _step "rsync --delete enabled (DEST matches SRC exactly per subtree)"
fi

_step "NFS merge: ${SCRATCH_ROOT}/ → ${DEST_ROOT}/"
rsync "${RSYNC_FLAGS[@]}" "${SCRATCH_ROOT}/" "${DEST_ROOT}/"
_step "NFS merge done → $DEST_ROOT"

if [[ "$UPLOAD_GCS" != "1" ]]; then
  _step "BEANS_PRO_UPLOAD_GCS!=1 — skipping GCS"
  exit 0
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "WARNING: gsutil not found; skipping GCS upload" >&2
  exit 0
fi

GCS_DEST="${GCS_PREFIX_BASE%/}/"
_step "GCS mirror: ${SCRATCH_ROOT}/ → ${GCS_DEST}"
gsutil -m rsync -r "${SCRATCH_ROOT}/" "${GCS_DEST}"
_step "GCS mirror done → ${GCS_DEST}"
