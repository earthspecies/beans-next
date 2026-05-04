#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output="/home/%u/logs/%A-cleanup-literal-scratch-user.log"
#SBATCH --job-name="cleanup literal /scratch/$USER"
#
# Cluster-wide cleanup helper.
#
# This runs `cleanup_literal_scratch_user.sh` once on every allocated node.
# To cover a whole partition, submit with a larger node count, e.g.:
#   sbatch --partition=a100-40 --nodes=8 examples/slurm/cleanup_literal_scratch_user_job.sh
#
# Note: Slurm can only run on nodes you allocate. If you truly need "all nodes",
# run this once per partition with enough nodes to cover that partition, or have an
# admin run it with an appropriate node list / reservation.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

script="$REPO/examples/slurm/cleanup_literal_scratch_user.sh"
if [[ ! -f "$script" ]]; then
  echo "ERROR: cleanup script not found: $script" >&2
  exit 2
fi

echo "Running cleanup on ${SLURM_JOB_NUM_NODES:-1} node(s)."
srun --ntasks-per-node=1 --kill-on-bad-exit=0 bash "$script"

