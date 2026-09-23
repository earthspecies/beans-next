#!/usr/bin/env bash
# Chain gpushort run_audex_eval.sbatch (RUN_MODE=full) submissions with
# --resume-from until the suite summary appears (all 51,653 rows scored).
# Runs detached on the Apocrita login node; each iteration submits one
# <=55min job, waits for its terminal state, and resubmits if incomplete.
set -uo pipefail

SNAP=/gpfs/scratch/acw777/beans-next-models/models--nvidia--Nemotron-Labs-Audex-30B-A3B/snapshots/4e7e342045736382ddf3e2952c313847a08642b8
OUTDIR=/gpfs/scratch/acw777/beans-next-runs/results/audex/audex_30b_a3b/full_greedy
SUMMARY="$OUTDIR/suite/beans_next_tiers_1_3_hf/suite_summary.json"
REPO=/gpfs/scratch/acw777/beans-next-audex-code
LOG=/gpfs/scratch/acw777/beans-next-runs/logs/audex_full_chain.log
CODE_COMMIT="87ff061+uncommitted(christos-audex-eval)"

cd "$REPO"
echo "$(date -Is) chain start" >> "$LOG"

iter=0
# If a job from this chain is already running (e.g. the first iteration was
# submitted interactively before this script started), pass its id as $1 so
# the loop waits on it instead of submitting a duplicate first.
JOBID="${1:-}"
while true; do
  iter=$((iter + 1))
  if [[ -f "$SUMMARY" ]]; then
    echo "$(date -Is) summary found, chain complete after $iter iterations" >> "$LOG"
    break
  fi

  if [[ -n "$JOBID" ]]; then
    echo "$(date -Is) iter=$iter reusing pre-submitted job=$JOBID" >> "$LOG"
  else
    JOBID=$(sbatch --parsable --partition gpushort --constraint "ampere&80G" --gres=gpu:2 --time=00:55:00 \
      --export=ALL,BEANS_NEXT_AUDEX_MODEL_PATH="$SNAP/checkpoint_folder_full",BEANS_NEXT_AUDEX_MODEL_REVISION=4e7e342045736382ddf3e2952c313847a08642b8,BEANS_NEXT_RUN_MODE=full,BEANS_NEXT_AUDEX_DECODING_ARM=greedy,BEANS_NEXT_CODE_COMMIT_OVERRIDE="$CODE_COMMIT",BEANS_NEXT_AUDEX_FULL_OUTPUT_DIR="$OUTDIR",BEANS_NEXT_RESUME_FROM="$OUTDIR" \
      examples/apocrita/run_audex_eval.sbatch)
    echo "$(date -Is) iter=$iter submitted job=$JOBID" >> "$LOG"
  fi

  # Wait for terminal state.
  while true; do
    sleep 60
    state=$(sacct -j "$JOBID" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$state" == COMPLETED* || "$state" == FAILED* || "$state" == CANCELLED* || "$state" == TIMEOUT* || "$state" == NODE_FAIL* || "$state" == OUT_OF_ME* ]]; then
      echo "$(date -Is) iter=$iter job=$JOBID terminal state=$state" >> "$LOG"
      break
    fi
  done
  JOBID=""
done

echo "$(date -Is) chain script exiting" >> "$LOG"
