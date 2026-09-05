#!/usr/bin/env bash
# Run a vLLM server (Audex plugin), the Audex adapter sidecar, and the
# BEANS-Next tiers 1-3 evaluation together.
#
# Submit with the partition/account/constraint for a 2x80GB-class GPU pool,
# e.g.:
#   sbatch --partition sae --account pilot_sae_gpu --constraint "ampere&80G" \
#     --gres=gpu:2 --ntasks-per-node=24 --cpus-per-gpu=12 --mem-per-cpu=7500M \
#     examples/apocrita/run_audex_eval.sbatch
# or for the gpushort smoke test (<=1h, single GPU, no account line needed):
#   sbatch --partition gpushort --constraint "ampere&80G" --gres=gpu:2 \
#     --time=00:55:00 examples/apocrita/run_audex_eval.sbatch
#
# BEANS_NEXT_RUN_MODE=beans_zero with BEANS_NEXT_DATA_BACKEND=esp_data and
# BEANS_NEXT_SUITE=beans_zero_core evaluates the full BEANS-Zero suite
# instead of BEANS-Next tiers 1-3 (see stage_beans_zero_esp_data.sbatch,
# which must run first, and chain_audex_beans_zero.sh for chaining).

#SBATCH --job-name=beans-audex-eval
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=48:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next-audex repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly DATA_ROOT="${BEANS_NEXT_HF_BEANS_NEXT_ROOT:-$SCRATCH_ROOT/esp/beans-next}"
readonly RUNNER_RUNTIME="${BEANS_NEXT_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-runtime}"
readonly SERVER_RUNTIME="${BEANS_NEXT_AUDEX_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-audex-runtime}"
readonly RESULTS_ROOT="${BEANS_NEXT_RESULTS_ROOT:-$SCRATCH_ROOT/beans-next-runs/results}"
readonly MODEL_PATH="${BEANS_NEXT_AUDEX_MODEL_PATH:?Set BEANS_NEXT_AUDEX_MODEL_PATH to the Audex snapshot dir}"
readonly MODEL_REVISION="${BEANS_NEXT_AUDEX_MODEL_REVISION:?Set BEANS_NEXT_AUDEX_MODEL_REVISION}"
readonly MODEL_TAG="${BEANS_NEXT_AUDEX_MODEL_TAG:-audex_30b_a3b}"
readonly CODE_COMMIT="${BEANS_NEXT_CODE_COMMIT_OVERRIDE:-$(git -C "$REPO" rev-parse HEAD)}"
readonly RUN_MODE="${BEANS_NEXT_RUN_MODE:-pilot}"
readonly DATA_BACKEND="${BEANS_NEXT_DATA_BACKEND:-huggingface}"
readonly SUITE="${BEANS_NEXT_SUITE:-beans_next_tiers_1_3_hf}"
readonly TASK_LIMIT="${BEANS_NEXT_LIMIT:-}"
readonly RESUME_FROM="${BEANS_NEXT_RESUME_FROM:-}"
readonly BEANS_ZERO_ESP_DATA_ROOT="${BEANS_ZERO_ESP_DATA_ROOT:-$SCRATCH_ROOT/beans-zero-esp-data}"
readonly BEANS_ZERO_RUN_ROOT="${BEANS_ZERO_RUN_ROOT:-$SCRATCH_ROOT/beans-zero-run}"
readonly DATA_REVISION="${BEANS_NEXT_DATA_REVISION:-}"
readonly TENSOR_PARALLEL_SIZE="${BEANS_NEXT_TP_SIZE:-2}"
readonly MAX_MODEL_LEN="${BEANS_NEXT_MAX_MODEL_LEN:-32768}"
readonly GPU_MEMORY_UTILIZATION="${BEANS_NEXT_GPU_MEMORY_UTILIZATION:-0.85}"
readonly CONCURRENCY="${BEANS_NEXT_CONCURRENCY:-8}"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
readonly SERVER_PYTHON="$SERVER_RUNTIME/.venv/bin/python"
readonly VLLM_BIN="$SERVER_RUNTIME/.venv/bin/vllm"
readonly JOB_ROOT="$RESULTS_ROOT/audex/$MODEL_TAG/job_${SLURM_JOB_ID}"

# Decoding arm ("greedy" or "nvidia_sampling"); drives AUDEX_EXTRA_BODY_JSON
# with no launcher code change, so the pilot A/B is just this switch.
readonly DECODING_ARM="${BEANS_NEXT_AUDEX_DECODING_ARM:-greedy}"
case "$DECODING_ARM" in
  greedy)
    AUDEX_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false},"temperature":0.0,"logit_bias":{"29":-100,"30":-100,"31":-100}}'
    ;;
  nvidia_sampling)
    AUDEX_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false},"temperature":0.7,"top_p":0.9,"logit_bias":{"29":-100,"30":-100,"31":-100}}'
    ;;
  *)
    echo "BEANS_NEXT_AUDEX_DECODING_ARM must be greedy or nvidia_sampling, got: $DECODING_ARM" >&2
    exit 1
    ;;
esac
readonly AUDEX_EXTRA_BODY_JSON

if [[ "$DATA_BACKEND" == "huggingface" || "$DATA_BACKEND" == "hf" ]]; then
  if [[ ! -f "$DATA_ROOT/test/metadata.parquet" ]]; then
    echo "Missing dataset metadata: $DATA_ROOT/test/metadata.parquet" >&2
    exit 1
  fi
else
  if [[ ! -d "$BEANS_ZERO_ESP_DATA_ROOT/beans-zero/v0.1.0/raw" ]]; then
    echo "Missing staged BEANS-Zero esp_data root:" \
      "$BEANS_ZERO_ESP_DATA_ROOT/beans-zero/v0.1.0/raw" \
      "(run stage_beans_zero_esp_data.sbatch first)" >&2
    exit 1
  fi
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model snapshot: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]] || ! grep -q '"architectures"' "$MODEL_PATH/config.json"; then
  echo "MODEL_PATH must point at the checkpoint dir (e.g. .../checkpoint_folder_full)," \
    "not the release root: $MODEL_PATH has no model config.json with an" \
    "'architectures' field (the release root's config.json is just a repo manifest)." >&2
  exit 1
fi
if [[ ! -x "$RUNNER_PYTHON" || ! -x "$SERVER_PYTHON" || ! -x "$VLLM_BIN" ]]; then
  echo "Missing runner or Audex vLLM runtime" >&2
  exit 1
fi
if ! "$SERVER_PYTHON" -c 'import audex_30b_a3b_vllm' 2>/dev/null; then
  echo "audex_30b_a3b_vllm plugin not importable in $SERVER_PYTHON" >&2
  echo "Re-run setup_audex_runtime.sbatch with BEANS_NEXT_AUDEX_SNAPSHOT_PATH=$MODEL_PATH" >&2
  exit 1
fi

find_free_port() {
  "$RUNNER_PYTHON" -c \
    'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

VLLM_PORT="${BEANS_NEXT_VLLM_PORT:-$(find_free_port)}"
ADAPTER_PORT="${BEANS_NEXT_ADAPTER_PORT:-$(find_free_port)}"
while [[ "$ADAPTER_PORT" == "$VLLM_PORT" ]]; do
  ADAPTER_PORT="$(find_free_port)"
done
readonly VLLM_PORT ADAPTER_PORT

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

# vLLM refuses to serve a file:// URI outside --allowed-local-media-path. The
# tiers 1-3 (huggingface backend) path serves audio straight out of $DATA_ROOT;
# BEANS-Zero (esp_data backend) materializes audio into a cache dir (and, when
# a clip is over its per-subset cap, a further trimmed-clip cache) that lives
# outside $DATA_ROOT, so it needs its own allowed root.
if [[ "$DATA_BACKEND" == "huggingface" || "$DATA_BACKEND" == "hf" ]]; then
  readonly ALLOWED_MEDIA_PATH="$DATA_ROOT"
else
  readonly ALLOWED_MEDIA_PATH="$BEANS_ZERO_RUN_ROOT"
  mkdir -p "$BEANS_ZERO_RUN_ROOT/audio-cache" "$BEANS_ZERO_RUN_ROOT/trimmed" \
    "$SERVER_RUNTIME/cache/empty-gcloud"
  export CLOUDSDK_CONFIG="$SERVER_RUNTIME/cache/empty-gcloud"
  export ALP_DATA_HOME="${BEANS_NEXT_ALP_DATA_HOME:-$BEANS_ZERO_ESP_DATA_ROOT}"
  export BEANS_NEXT_ESP_AUDIO_CACHE_DIR="$BEANS_ZERO_RUN_ROOT/audio-cache"
  export BEANS_NEXT_ESP_DATA_WORKERS="${BEANS_NEXT_ESP_DATA_WORKERS:-12}"
  export AUDEX_TRIMMED_CACHE_DIR="$BEANS_ZERO_RUN_ROOT/trimmed"
fi

vllm_pid=""
adapter_pid=""
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$adapter_pid" ]]; then kill "$adapter_pid" 2>/dev/null || true; fi
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
  --served-model-name audex_audio \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"audio": 1}' \
  --allowed-local-media-path "$ALLOWED_MEDIA_PATH" \
  >"$JOB_ROOT/vllm.log" 2>&1 &
vllm_pid="$!"
wait_for_url "http://127.0.0.1:$VLLM_PORT/health" "vLLM" "$vllm_pid" 360

export AUDEX_ADAPTER_STUB=0
export AUDEX_UPSTREAM_BASE_URL="http://127.0.0.1:$VLLM_PORT"
export AUDEX_MODEL_ID=audex_audio
export AUDEX_MODEL_REVISION="$MODEL_REVISION"
export AUDEX_MAX_BATCH_SIZE="$CONCURRENCY"
export AUDEX_MAX_CONCURRENCY="$CONCURRENCY"
export AUDEX_UPSTREAM_TIMEOUT_SEC=300
export AUDEX_UPSTREAM_RETRIES=2
export AUDEX_EXTRA_BODY_JSON
export PORT="$ADAPTER_PORT"
"$SERVER_PYTHON" examples/servers/audex/serve.py \
  >"$JOB_ROOT/adapter.log" 2>&1 &
adapter_pid="$!"
wait_for_url "http://127.0.0.1:$ADAPTER_PORT/health" "Audex adapter" "$adapter_pid" 60

run_benchmark() {
  local out_dir="$1"
  local run_id="$2"
  local -a extra_args=()
  shift 2
  extra_args=("$@")
  if [[ -n "$RESUME_FROM" ]]; then
    extra_args+=(--resume-from "$RESUME_FROM")
  fi
  mkdir -p "$out_dir"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv -l 1 >"$out_dir/gpu.csv" 2>&1 &
  monitor_pid="$!"
  curl --silent "http://127.0.0.1:$VLLM_PORT/metrics" \
    >"$out_dir/vllm_metrics_before.prom" || true
  /usr/bin/time -v -o "$out_dir/time.txt" \
    "$RUNNER_PYTHON" -m beans_next.cli run \
      --predict-url "http://127.0.0.1:$ADAPTER_PORT/predict" \
      --backend "$DATA_BACKEND" \
      --suite "$SUITE" \
      --run-id "$run_id" \
      --output-dir "$out_dir" \
      --workers "${SLURM_CPUS_PER_GPU:-12}" \
      --preserve-file-paths \
      --seed 20260901 \
      "${extra_args[@]}"
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=""
  curl --silent "http://127.0.0.1:$VLLM_PORT/metrics" \
    >"$out_dir/vllm_metrics_after.prom" || true
}

cat >"$JOB_ROOT/run.env" <<EOF
code_commit=$CODE_COMMIT
dataset_root=$DATA_ROOT
data_backend=$DATA_BACKEND
data_revision=${DATA_REVISION:-none}
model_id=audex_audio
model_path=$MODEL_PATH
model_revision=$MODEL_REVISION
run_mode=$RUN_MODE
suite=$SUITE
decoding_arm=$DECODING_ARM
resume_from=${RESUME_FROM:-none}
slurm_job_id=$SLURM_JOB_ID
slurm_partition=${SLURM_JOB_PARTITION:-unknown}
slurm_node=${SLURMD_NODENAME:-unknown}
tensor_parallel_size=$TENSOR_PARALLEL_SIZE
max_model_len=$MAX_MODEL_LEN
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
concurrency=$CONCURRENCY
vllm_port=$VLLM_PORT
adapter_port=$ADAPTER_PORT
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv \
    > "$JOB_ROOT/apocrita_context.json.gpu.csv" 2>&1 || true
fi

if [[ "$RUN_MODE" == "smoke" ]]; then
  if [[ "$DATA_BACKEND" == "huggingface" || "$DATA_BACKEND" == "hf" ]]; then
    run_benchmark \
      "$JOB_ROOT/smoke" \
      "${MODEL_TAG}_smoke_${DECODING_ARM}" \
      --suite beans_next_tier_1_hf \
      --limit "${TASK_LIMIT:-2}"
  else
    # esp_data smoke (e.g. BEANS-Zero): honour $SUITE/$BEANS_NEXT_SUITE
    # (typically beans_zero_smoke) rather than the HF tier-1 default above.
    run_benchmark \
      "$JOB_ROOT/smoke" \
      "${MODEL_TAG}_smoke_${DECODING_ARM}" \
      --limit "${TASK_LIMIT:-2}"
  fi
elif [[ "$RUN_MODE" == "pilot" ]]; then
  run_benchmark \
    "$JOB_ROOT/pilot" \
    "${MODEL_TAG}_pilot_${DECODING_ARM}" \
    --limit "${TASK_LIMIT:-32}"
elif [[ "$RUN_MODE" == "full" ]]; then
  # Full runs may be chained across several wall-clock-limited jobs (e.g. a
  # 1h queue with --resume-from); BEANS_NEXT_AUDEX_FULL_OUTPUT_DIR lets each
  # submission in the chain write into the same persistent output dir instead
  # of a fresh one under this job's own JOB_ROOT.
  run_benchmark \
    "${BEANS_NEXT_AUDEX_FULL_OUTPUT_DIR:-$JOB_ROOT/full}" \
    "${MODEL_TAG}_full_${DECODING_ARM}"
elif [[ "$RUN_MODE" == "beans_zero" ]]; then
  # Same chaining pattern as `full`, but for beans_zero_core (esp_data
  # backend) instead of beans_next_tiers_1_3_hf.
  run_benchmark \
    "${BEANS_NEXT_AUDEX_BEANS_ZERO_OUTPUT_DIR:-$JOB_ROOT/beans_zero_greedy}" \
    "${MODEL_TAG}_beans_zero_${DECODING_ARM}"
else
  echo "BEANS_NEXT_RUN_MODE must be smoke, pilot, full, or beans_zero, got: $RUN_MODE" >&2
  exit 1
fi

echo "results=$JOB_ROOT"
