#!/usr/bin/env bash
# Remove per-task JSONL + checkpoint + summary under suite/<suite_id>/<task_id>/ so a
# subsequent `beans-next run --resume --suite ...` will re-run inference for those tasks.
#
# Usage (from repo root on a node that sees RUN_ROOT, e.g. Slurm login):
#   bash examples/slurm/wipe_suite_task_artifacts.sh \
#     "$HOME/beans-next-results/ingested/full_af3_a100_beans_zero_core_r1" \
#     beans_zero_core \
#     beans_zero_unseen_species_cmn beans_zero_captioning
#
# WARNING: Destructive. Only use on tasks you intend to re-predict.

set -euo pipefail

RUN_ROOT="${1:?RUN_ROOT (path to run dir containing suite/) required}"
SUITE_ID="${2:?SUITE_ID (e.g. beans_zero_core) required}"
shift 2
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "ERROR: pass at least one eval_task id after SUITE_ID" >&2
  exit 1
fi

for t in "${TASKS[@]}"; do
  d="${RUN_ROOT%/}/suite/${SUITE_ID}/${t}"
  if [[ ! -d "$d" ]]; then
    echo "SKIP (missing dir): $d" >&2
    continue
  fi
  rm -f \
    "${d}/checkpoint.json" \
    "${d}/predictions.jsonl" \
    "${d}/processed_predictions.jsonl" \
    "${d}/scored_predictions.jsonl" \
    "${d}/summary.json"
  echo "Wiped: $d"
done
