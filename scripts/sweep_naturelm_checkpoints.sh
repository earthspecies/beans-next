#!/usr/bin/env bash
set -euo pipefail

#
# I17-D — NatureLM v1.1 checkpoint sweep script
#

: "${SWEEP_CHECKPOINT_LIST:?SWEEP_CHECKPOINT_LIST is required (newline-delimited GCS URIs)}"

: "${SWEEP_BEANS_ZERO_TASK:=beans_zero_esc50_official}"
: "${SWEEP_BEANS_PRO_TASK:=beans_next_crow_description}"
: "${SWEEP_LIMIT:=10}"
: "${SWEEP_OUTPUT_DIR:=results/sweep/naturelm_v1.1}"
: "${SWEEP_SLURM_PARTITION:=a100-40}"
: "${SWEEP_PORT:=19080}"

: "${SWEEP_LOCAL_REPO_ROOT:=${HOME}/code/beans-next}"
: "${SWEEP_NFS_REPO_ROOT:=/mnt/home/${USER}/code/beans-next}"
: "${SWEEP_SLURM_REPO_ROOT:=/home/${USER}/code/beans-next}"
: "${SWEEP_URL_DIR:=/mnt/home/${USER}/beans-next-launchers}"

LOCAL_REPO_ROOT="${SWEEP_LOCAL_REPO_ROOT}"
NFS_REPO_ROOT="${SWEEP_NFS_REPO_ROOT}"
SLURM_REPO_ROOT="${SWEEP_SLURM_REPO_ROOT}"
URL_DIR="${SWEEP_URL_DIR}"

SUMMARY_TSV="${SWEEP_OUTPUT_DIR%/}/summary_table.tsv"

mkdir -p "${SWEEP_OUTPUT_DIR%/}"

if [[ ! -f "$SUMMARY_TSV" ]]; then
  printf '%s\n' $'checkpoint_basename\ttask\tn_samples\tn_errors\ttop1_accuracy\tgcs_model\tmodel_revision' >"$SUMMARY_TSV"
fi

current_job_id=""
cleanup_current_job() {
  if [[ -n "${current_job_id:-}" ]]; then
    ssh slurm "scancel ${current_job_id}" >/dev/null 2>&1 || true
    current_job_id=""
  fi
}
trap cleanup_current_job EXIT

sync_to_nfs() {
  rsync -av --delete --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude '.ruff_cache/' --exclude '.pytest_cache/' \
    "${LOCAL_REPO_ROOT%/}/" "${NFS_REPO_ROOT%/}/"
}

poll_squeue_running() {
  local job_id="$1"
  local max_iters="$2"
  local state=""
  for _ in $(seq 1 "$max_iters"); do
    state="$(ssh slurm "squeue --me --jobs ${job_id} -h -o %T" 2>/dev/null || true)"
    if [[ "$state" == "RUNNING" ]]; then
      return 0
    fi
    if [[ "$state" == "FAILED" || "$state" == "CANCELLED" || "$state" == "TIMEOUT" || "$state" == "OUT_OF_MEMORY" ]]; then
      return 2
    fi
    sleep 30
  done
  return 1
}

wait_for_url_file() {
  local job_id="$1"
  local max_iters="$2"
  local url_file="${URL_DIR%/}/${job_id}.url"
  for _ in $(seq 1 "$max_iters"); do
    if [[ -f "$url_file" ]]; then
      printf '%s\n' "$url_file"
      return 0
    fi
    sleep 15
  done
  return 1
}

poll_health_200() {
  local predict_url="$1"
  local max_iters="$2"
  local base_url="${predict_url%/predict}"
  local http_code=""
  for _ in $(seq 1 "$max_iters"); do
    http_code="$(curl -s -o /dev/null -w "%{http_code}" "${base_url%/}/health" || true)"
    if [[ "$http_code" == "200" ]]; then
      return 0
    fi
    sleep 5
  done
  return 1
}

json_get_key() {
  local key="$1"
  uv run python -c "import json,sys; d=json.load(sys.stdin); print(d.get('${key}','?'))"
}

run_one_task() {
  local checkpoint_basename="$1"
  local out_dir="$2"
  local task_id="$3"
  local predict_url="$4"
  local limit="$5"
  local run_dir="$6"

  mkdir -p "$run_dir"
  uv run beans-next run \
    --task-id "$task_id" \
    --predict-url "$predict_url" \
    --limit "$limit" \
    --output-dir "$run_dir"

  if [[ ! -f "$run_dir/summary.json" ]]; then
    echo "BLOCKED: missing summary.json at $run_dir/summary.json" >&2
    return 3
  fi

  local n_samples n_errors top1_accuracy
  n_samples="$(uv run python -c "import json; print(json.load(open('${run_dir}/summary.json'))['n_samples'])" 2>/dev/null || echo "?")"
  n_errors="$(uv run python -c "import json; print(json.load(open('${run_dir}/summary.json'))['n_errors'])" 2>/dev/null || echo "?")"
  top1_accuracy="$(uv run python -c "import json; d=json.load(open('${run_dir}/summary.json')); print(d.get('top1_accuracy','?'))" 2>/dev/null || echo "?")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$checkpoint_basename" "$task_id" "$n_samples" "$n_errors" "$top1_accuracy" "$MODEL" "$REVISION" >>"$SUMMARY_TSV"
}

while IFS= read -r uri; do
  [[ -z "${uri// }" ]] && continue
  [[ "${uri:0:1}" == "#" ]] && continue

  checkpoint_basename="$(basename "${uri%/}")"
  out_dir="${SWEEP_OUTPUT_DIR%/}/${checkpoint_basename}"
  mkdir -p "$out_dir"

  echo "=== Checkpoint: ${checkpoint_basename} ==="

  sync_to_nfs

  current_job_id="$(
    ssh slurm "cd \"${SLURM_REPO_ROOT}\" && \
      BEANS_PRO_PORT=\"${SWEEP_PORT}\" BEANS_PRO_DEBUG=1 NATURELM_GCS_CHECKPOINT_URI=\"${uri}\" \
      sbatch --partition=\"${SWEEP_SLURM_PARTITION}\" --parsable examples/slurm/serve_naturelm_v1_1.sh"
  )"
  echo "  Serve job: ${current_job_id}"

  if ! poll_squeue_running "$current_job_id" 40; then
    echo "BLOCKED: serve job ${current_job_id} did not reach RUNNING" >&2
    cleanup_current_job
    continue
  fi

  url_file="$(wait_for_url_file "$current_job_id" 40 || true)"
  if [[ -z "${url_file:-}" || ! -f "$url_file" ]]; then
    echo "BLOCKED: URL file never appeared for ${current_job_id}" >&2
    cleanup_current_job
    continue
  fi

  predict_url="$(cat "$url_file")"

  if ! poll_health_200 "$predict_url" 120; then
    echo "BLOCKED: /health never returned 200 for ${current_job_id}" >&2
    cleanup_current_job
    continue
  fi

  info_json="$(curl -sf "${predict_url%/predict}/info" || echo '{}')"
  MODEL="$(printf '%s' "$info_json" | json_get_key model || echo '?')"
  REVISION="$(printf '%s' "$info_json" | json_get_key model_revision || echo '?')"
  echo "  /info model=${MODEL} revision=${REVISION}"
  if [[ "$REVISION" != "$checkpoint_basename" ]]; then
    echo "  WARNING: model_revision mismatch (expected ${checkpoint_basename}, got ${REVISION})"
  fi

  for task_id in "$SWEEP_BEANS_ZERO_TASK" "$SWEEP_BEANS_PRO_TASK"; do
    smoke_dir="${out_dir%/}/smoke_${task_id}"
    echo "  Smoke: task=${task_id} limit=1"
    run_one_task "$checkpoint_basename" "$out_dir" "$task_id" "$predict_url" 1 "$smoke_dir"
  done

  for task_id in "$SWEEP_BEANS_ZERO_TASK" "$SWEEP_BEANS_PRO_TASK"; do
    run_dir="${out_dir%/}/${task_id}_limit${SWEEP_LIMIT}"
    echo "  Run: task=${task_id} limit=${SWEEP_LIMIT}"
    run_one_task "$checkpoint_basename" "$out_dir" "$task_id" "$predict_url" "${SWEEP_LIMIT}" "$run_dir"
  done

  cleanup_current_job
  echo "  Serve job cancelled."
done <"$SWEEP_CHECKPOINT_LIST"

