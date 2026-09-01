#!/usr/bin/env bash
# Run a vLLM server, concurrent BEANS proxy, and text-only evaluation together.
# Submit with the partition, account, and constraint required by the model.

#SBATCH --job-name=beans-text-eval
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=12:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly DATA_ROOT="${BEANS_NEXT_HF_BEANS_NEXT_ROOT:-$SCRATCH_ROOT/esp/beans-next}"
readonly RUNNER_RUNTIME="${BEANS_NEXT_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-runtime}"
readonly SERVER_RUNTIME="${BEANS_NEXT_VLLM_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-vllm-runtime}"
readonly RESULTS_ROOT="${BEANS_NEXT_RESULTS_ROOT:-$SCRATCH_ROOT/beans-next-runs/results}"
readonly MODEL_ID="${BEANS_NEXT_MODEL_ID:?Set BEANS_NEXT_MODEL_ID}"
readonly MODEL_PATH="${BEANS_NEXT_MODEL_PATH:?Set BEANS_NEXT_MODEL_PATH to an exact snapshot}"
readonly MODEL_REVISION="${BEANS_NEXT_MODEL_REVISION:?Set BEANS_NEXT_MODEL_REVISION}"
readonly MODEL_TAG="${BEANS_NEXT_MODEL_TAG:?Set BEANS_NEXT_MODEL_TAG}"
readonly CODE_COMMIT="${BEANS_NEXT_CODE_COMMIT_OVERRIDE:-$(git -C "$REPO" rev-parse HEAD)}"
readonly RUN_MODE="${BEANS_NEXT_RUN_MODE:-pilot}"
readonly PILOT_LIMIT="${BEANS_NEXT_PILOT_LIMIT:-16}"
readonly MAX_MODEL_LEN="${BEANS_NEXT_MAX_MODEL_LEN:-8192}"
readonly MAX_NUM_SEQS="${BEANS_NEXT_MAX_NUM_SEQS:-64}"
readonly GPU_MEMORY_UTILIZATION="${BEANS_NEXT_GPU_MEMORY_UTILIZATION:-0.90}"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
readonly SERVER_PYTHON="$SERVER_RUNTIME/.venv/bin/python"
readonly VLLM_BIN="$SERVER_RUNTIME/.venv/bin/vllm"
readonly JOB_ROOT="$RESULTS_ROOT/text_only/$MODEL_TAG/job_${SLURM_JOB_ID}"

if [[ ! -f "$DATA_ROOT/test/metadata.parquet" ]]; then
  echo "Missing dataset metadata: $DATA_ROOT/test/metadata.parquet" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model snapshot: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$RUNNER_PYTHON" || ! -x "$SERVER_PYTHON" || ! -x "$VLLM_BIN" ]]; then
  echo "Missing runner or vLLM runtime" >&2
  exit 1
fi

find_free_port() {
  "$RUNNER_PYTHON" -c \
    'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

VLLM_PORT="${BEANS_NEXT_VLLM_PORT:-$(find_free_port)}"
PROXY_PORT="${BEANS_NEXT_PROXY_PORT:-$(find_free_port)}"
while [[ "$PROXY_PORT" == "$VLLM_PORT" ]]; do
  PROXY_PORT="$(find_free_port)"
done
readonly VLLM_PORT PROXY_PORT

mkdir -p "$JOB_ROOT" "$SERVER_RUNTIME/tmp" "$SERVER_RUNTIME/cache/huggingface"
cd "$REPO"

module use /data/home/acw777/privatemodules
module load uv/latest

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SERVER_RUNTIME/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${BEANS_NEXT_MODEL_CACHE:-$SCRATCH_ROOT/beans-next-models}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$SERVER_RUNTIME/cache"
export VLLM_CACHE_ROOT="$SERVER_RUNTIME/cache/vllm"
export TMPDIR="$SERVER_RUNTIME/tmp"
export BEANS_NEXT_HF_BEANS_NEXT_ROOT="$DATA_ROOT"
export BEANS_NEXT_HF_WORKERS="${SLURM_CPUS_PER_GPU:-12}"
export TOKENIZERS_PARALLELISM=false

vllm_pid=""
proxy_pid=""
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$proxy_pid" ]]; then kill "$proxy_pid" 2>/dev/null || true; fi
  if [[ -n "$vllm_pid" ]]; then kill "$vllm_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local url="$1"
  local label="$2"
  local pid="$3"
  local attempts="${4:-180}"
  for ((i = 1; i <= attempts; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label process exited before $url became healthy" >&2
      return 1
    fi
    if curl --fail --silent "$url" >/dev/null; then return 0; fi
    sleep 5
  done
  echo "Timed out while waiting for $label at $url" >&2
  return 1
}

"$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  >"$JOB_ROOT/vllm.log" 2>&1 &
vllm_pid="$!"
wait_for_url "http://127.0.0.1:$VLLM_PORT/health" "vLLM" "$vllm_pid"

start_proxy() {
  local concurrency="$1"
  local proxy_log="$2"
  export OPENAI_PROXY_STUB=0
  export OPENAI_BASE_URL="http://127.0.0.1:$VLLM_PORT"
  export OPENAI_MODEL="$MODEL_ID"
  export OPENAI_MODEL_REVISION="$MODEL_REVISION"
  export OPENAI_PROXY_SYSTEM_PROMPT=""
  export OPENAI_PROXY_MAX_BATCH_SIZE="$concurrency"
  export OPENAI_PROXY_MAX_CONCURRENCY="$concurrency"
  export OPENAI_PROXY_TIMEOUT_SEC=300
  export OPENAI_PROXY_RETRIES=2
  export PORT="$PROXY_PORT"
  "$SERVER_PYTHON" examples/servers/openai_compatible_proxy/serve.py \
    >"$proxy_log" 2>&1 &
  proxy_pid="$!"
  wait_for_url \
    "http://127.0.0.1:$PROXY_PORT/health" "BEANS proxy" "$proxy_pid" 60
}

stop_proxy() {
  if [[ -n "$proxy_pid" ]]; then
    kill "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
    proxy_pid=""
  fi
}

run_benchmark() {
  local out_dir="$1"
  local run_id="$2"
  local modality="$3"
  shift 3
  mkdir -p "$out_dir"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv -l 1 >"$out_dir/gpu.csv" 2>&1 &
  monitor_pid="$!"
  curl --silent "http://127.0.0.1:$VLLM_PORT/metrics" \
    >"$out_dir/vllm_metrics_before.prom" || true
  /usr/bin/time -v -o "$out_dir/time.txt" \
    "$RUNNER_PYTHON" -m beans_next.cli run \
      --predict-url "http://127.0.0.1:$PROXY_PORT/predict" \
      --backend huggingface \
      --suite beans_next_all_tiers_hf \
      --run-id "$run_id" \
      --output-dir "$out_dir" \
      --workers "${SLURM_CPUS_PER_GPU:-12}" \
      --modality-mode "$modality" \
      --seed 20260901 \
      "$@"
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=""
  curl --silent "http://127.0.0.1:$VLLM_PORT/metrics" \
    >"$out_dir/vllm_metrics_after.prom" || true
}

cat >"$JOB_ROOT/run.env" <<EOF
code_commit=$CODE_COMMIT
dataset_root=$DATA_ROOT
model_id=$MODEL_ID
model_revision=$MODEL_REVISION
model_path=$MODEL_PATH
run_mode=$RUN_MODE
slurm_job_id=$SLURM_JOB_ID
slurm_partition=${SLURM_JOB_PARTITION:-unknown}
slurm_node=${SLURMD_NODENAME:-unknown}
max_model_len=$MAX_MODEL_LEN
max_num_seqs=$MAX_NUM_SEQS
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
vllm_port=$VLLM_PORT
proxy_port=$PROXY_PORT
EOF

if [[ "$RUN_MODE" == "pilot" ]]; then
  IFS=',' read -r -a grid <<<"${BEANS_NEXT_CONCURRENCY_GRID:-16,32,64}"
  for concurrency in "${grid[@]}"; do
    out_dir="$JOB_ROOT/pilot_c${concurrency}"
    start_proxy "$concurrency" "$out_dir.proxy.log"
    run_benchmark \
      "$out_dir" \
      "${MODEL_TAG}_pilot_c${concurrency}" \
      text-only-informed \
      --limit "$PILOT_LIMIT"
    stop_proxy
  done
elif [[ "$RUN_MODE" == "full" ]]; then
  concurrency="${BEANS_NEXT_CONCURRENCY:-32}"
  start_proxy "$concurrency" "$JOB_ROOT/proxy.log"
  run_benchmark \
    "$JOB_ROOT/text_only_informed" \
    "${MODEL_TAG}_text_only_informed" \
    text-only-informed
  run_benchmark \
    "$JOB_ROOT/text_only_sample" \
    "${MODEL_TAG}_text_only_sample" \
    text-only \
    --sample-fraction 0.1
  stop_proxy
else
  echo "BEANS_NEXT_RUN_MODE must be pilot or full, got: $RUN_MODE" >&2
  exit 1
fi

echo "results=$JOB_ROOT"
