#!/usr/bin/env bash
# Chain gpushort run_gemma4_eval.sbatch (RUN_MODE=full) submissions with
# --resume-from until the beans_next_tiers_1_3_hf suite summary appears (all
# 51,653 rows scored). Runs detached on the Apocrita login node; each
# iteration submits one <=55min job, waits for its terminal state, and
# resubmits if incomplete. Clone of chain_audex_full.sh (and, for the
# env-var-driven structure, chain_audex_beans_zero.sh), retargeted at
# google/gemma-4-12B-it (single GPU, no tensor parallelism).
#
# Writes into results/gemma4/gemma4_12b/tiers_1_3_greedy -- one writer per
# output dir. Do not run this concurrently with chain_gemma4_beans_zero.sh
# against the same gpushort allocation: run the two chains sequentially,
# since they would otherwise compete for the same GPU pool and double the
# failure surface for no wall-clock gain worth having.
set -uo pipefail

# GEMMA4_TIERS_CHAIN_MODEL_REVISION must be set to the revision resolved by
# download_gemma4.sbatch's manifest (examples/apocrita/download_gemma4.py
# pins google/gemma-4-12B-it's `info.sha` at download time, mirroring how the
# Audex chain pins 4e7e342045736382ddf3e2952c313847a08642b8).
SNAP="${GEMMA4_TIERS_CHAIN_SNAPSHOT:?Set GEMMA4_TIERS_CHAIN_SNAPSHOT to the resolved gemma-4-12B-it snapshot dir from the download_gemma4.py manifest}"
MODEL_REVISION="${GEMMA4_TIERS_CHAIN_MODEL_REVISION:?Set GEMMA4_TIERS_CHAIN_MODEL_REVISION to the resolved_revision from the same manifest}"
BACKEND="${GEMMA4_TIERS_CHAIN_BACKEND:-vllm}"
OUTDIR="${GEMMA4_TIERS_CHAIN_OUTDIR:-/gpfs/scratch/acw777/beans-next-runs/results/gemma4/gemma4_12b/tiers_1_3_greedy}"
SUMMARY="$OUTDIR/suite/beans_next_tiers_1_3_hf/suite_summary.json"
REPO="${GEMMA4_TIERS_CHAIN_REPO:-/gpfs/scratch/acw777/beans-next-audex-code}"
LOG="${GEMMA4_TIERS_CHAIN_LOG:-/gpfs/scratch/acw777/beans-next-runs/logs/gemma4_tiers_chain.log}"
CODE_COMMIT="${GEMMA4_TIERS_CHAIN_CODE_COMMIT:?Set GEMMA4_TIERS_CHAIN_CODE_COMMIT (git rev-parse HEAD on $REPO)}"

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
    JOBID=$(sbatch --parsable --partition gpushort --constraint "ampere&80G" --gres=gpu:1 --time=00:55:00 \
      --export=ALL,BEANS_NEXT_GEMMA4_MODEL_PATH="$SNAP",BEANS_NEXT_GEMMA4_MODEL_REVISION="$MODEL_REVISION",BEANS_NEXT_GEMMA4_BACKEND="$BACKEND",BEANS_NEXT_RUN_MODE=full,BEANS_NEXT_SUITE=beans_next_tiers_1_3_hf,BEANS_NEXT_GEMMA4_DECODING_ARM=greedy,BEANS_NEXT_CODE_COMMIT_OVERRIDE="$CODE_COMMIT",BEANS_NEXT_GEMMA4_FULL_OUTPUT_DIR="$OUTDIR",BEANS_NEXT_RESUME_FROM="$OUTDIR" \
      examples/apocrita/run_gemma4_eval.sbatch)
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
