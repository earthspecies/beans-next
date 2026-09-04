#!/usr/bin/env bash
# Run a vLLM server (stock vllm[audio] >= 0.23.0, no vendor plugin), the
# Gemma 4 adapter sidecar, and the BEANS-Next tiers 1-3 / BEANS-Zero
# evaluation together. Clone of run_audex_eval.sbatch retargeted at
# google/gemma-4-12B-it: single GPU (no tensor parallelism), a smaller
# --max-model-len, and Gemma-specific serving flags (--limit-mm-per-prompt,
# --reasoning-parser).
#
# Submit for a single-GPU 80GB-class node, e.g.:
#   sbatch --partition sae --account pilot_sae_gpu --constraint "ampere&80G" \
#     --gres=gpu:1 --cpus-per-gpu=12 --mem-per-cpu=7500M \
#     examples/apocrita/run_gemma4_eval.sbatch
# or for the gpushort smoke test (<=1h, single GPU, no account line needed):
#   sbatch --partition gpushort --constraint "ampere&80G" --gres=gpu:1 \
#     --time=00:55:00 examples/apocrita/run_gemma4_eval.sbatch
#
# BEANS_NEXT_RUN_MODE=beans_zero with BEANS_NEXT_DATA_BACKEND=esp_data and
# BEANS_NEXT_SUITE=beans_zero_core evaluates the full BEANS-Zero suite
# instead of BEANS-Next tiers 1-3. Both datasets are already staged on
# scratch (no staging job needed here, unlike Audex): see
# BEANS_ZERO_ESP_DATA_ROOT / BEANS_NEXT_HF_BEANS_NEXT_ROOT below.
#
# BEANS_NEXT_RUN_MODE=probe runs the Decision-1 audio-sensitivity probe
# (examples/apocrita/gemma4_audio_probe.py) against the running launcher and
# exits non-zero if audio does not appear to reach the model. Run this before
# committing any full run to GEMMA4_BACKEND=vllm.
#
# GEMMA4_BACKEND selects the serving backend:
#   vllm         (default) - starts a local vLLM OpenAI-compatible server and
#                 proxies through it, exactly like the Audex launcher.
#   transformers - skips the vLLM server entirely; the launcher loads
#                 Gemma4UnifiedForConditionalGeneration + AutoProcessor
#                 in-process and serves directly. This is the Decision-1
#                 costed fallback (no continuous batching, ~3-5x slower).

#SBATCH --job-name=beans-gemma4-eval
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=48:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next-audex repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly DATA_ROOT="${BEANS_NEXT_HF_BEANS_NEXT_ROOT:-$SCRATCH_ROOT/esp/beans-next}"
readonly RUNNER_RUNTIME="${BEANS_NEXT_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-runtime}"
readonly SERVER_RUNTIME="${BEANS_NEXT_GEMMA4_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-gemma4-runtime}"
readonly RESULTS_ROOT="${BEANS_NEXT_RESULTS_ROOT:-$SCRATCH_ROOT/beans-next-runs/results}"
readonly MODEL_PATH="${BEANS_NEXT_GEMMA4_MODEL_PATH:?Set BEANS_NEXT_GEMMA4_MODEL_PATH to the gemma-4-12B-it snapshot dir}"
readonly MODEL_REVISION="${BEANS_NEXT_GEMMA4_MODEL_REVISION:?Set BEANS_NEXT_GEMMA4_MODEL_REVISION}"
readonly MODEL_TAG="${BEANS_NEXT_GEMMA4_MODEL_TAG:-gemma4_12b}"
readonly CODE_COMMIT="${BEANS_NEXT_CODE_COMMIT_OVERRIDE:-$(git -C "$REPO" rev-parse HEAD)}"
readonly RUN_MODE="${BEANS_NEXT_RUN_MODE:-pilot}"
readonly DATA_BACKEND="${BEANS_NEXT_DATA_BACKEND:-huggingface}"
readonly SUITE="${BEANS_NEXT_SUITE:-beans_next_tiers_1_3_hf}"
readonly TASK_LIMIT="${BEANS_NEXT_LIMIT:-}"
readonly RESUME_FROM="${BEANS_NEXT_RESUME_FROM:-}"
readonly BEANS_ZERO_ESP_DATA_ROOT="${BEANS_ZERO_ESP_DATA_ROOT:-/gpfs/scratch/acw777/beans-zero-esp-data}"
readonly GEMMA4_RUN_ROOT="${BEANS_NEXT_GEMMA4_RUN_ROOT:-$SCRATCH_ROOT/gemma4-run}"
readonly DATA_REVISION="${BEANS_NEXT_DATA_REVISION:-}"
readonly MAX_MODEL_LEN="${BEANS_NEXT_MAX_MODEL_LEN:-16384}"
readonly GPU_MEMORY_UTILIZATION="${BEANS_NEXT_GPU_MEMORY_UTILIZATION:-0.90}"
readonly CONCURRENCY="${BEANS_NEXT_CONCURRENCY:-8}"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
readonly SERVER_PYTHON="$SERVER_RUNTIME/.venv/bin/python"
readonly VLLM_BIN="$SERVER_RUNTIME/.venv/bin/vllm"
readonly JOB_ROOT="$RESULTS_ROOT/gemma4/$MODEL_TAG/job_${SLURM_JOB_ID}"

# GEMMA4_BACKEND selects vllm (default, continuous batching via a local vLLM
# server) or transformers (Decision-1 fallback: in-process HF Transformers,
# no vLLM server started at all).
readonly GEMMA4_BACKEND="${BEANS_NEXT_GEMMA4_BACKEND:-vllm}"
if [[ "$GEMMA4_BACKEND" != "vllm" && "$GEMMA4_BACKEND" != "transformers" ]]; then
  echo "BEANS_NEXT_GEMMA4_BACKEND must be vllm or transformers, got: $GEMMA4_BACKEND" >&2
  exit 1
fi

# Decoding arm ("greedy" or "vendor_sampling"); drives GEMMA4_EXTRA_BODY_JSON
# with no launcher code change, so the pilot A/B is just this switch. Greedy
# is the reported arm (Decision 4: thinking disabled either way).
readonly DECODING_ARM="${BEANS_NEXT_GEMMA4_DECODING_ARM:-greedy}"
case "$DECODING_ARM" in
  greedy)
    GEMMA4_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false},"temperature":0.0}'
    ;;
  vendor_sampling)
    GEMMA4_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false},"temperature":1.0,"top_p":0.95,"top_k":64}'
    ;;
  *)
    echo "BEANS_NEXT_GEMMA4_DECODING_ARM must be greedy or vendor_sampling, got: $DECODING_ARM" >&2
    exit 1
    ;;
esac
readonly GEMMA4_EXTRA_BODY_JSON

if [[ "$DATA_BACKEND" == "huggingface" || "$DATA_BACKEND" == "hf" ]]; then
  if [[ ! -f "$DATA_ROOT/test/metadata.parquet" ]]; then
    echo "Missing dataset metadata: $DATA_ROOT/test/metadata.parquet" >&2
    exit 1
  fi
else
  if [[ ! -d "$BEANS_ZERO_ESP_DATA_ROOT/beans-zero/v0.1.0/raw" ]]; then
    echo "Missing staged BEANS-Zero esp_data root:" \
      "$BEANS_ZERO_ESP_DATA_ROOT/beans-zero/v0.1.0/raw" \
      "(this should already be staged from the Audex run; see the plan)" >&2
    exit 1
  fi
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model snapshot: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing model config.json under $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$RUNNER_PYTHON" || ! -x "$SERVER_PYTHON" ]]; then
  echo "Missing runner or gemma4 runtime" >&2
  exit 1
fi
if [[ "$GEMMA4_BACKEND" == "vllm" && ! -x "$VLLM_BIN" ]]; then
  echo "Missing vLLM binary in gemma4 runtime: $VLLM_BIN" >&2
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

# vLLM (and the transformers-backend launcher's own file reads) refuses a
# file:// URI outside --allowed-local-media-path / its own root. The tiers
# 1-3 (huggingface backend) path serves audio straight out of $DATA_ROOT;
# BEANS-Zero (esp_data backend) materializes audio into a cache dir (and, when
# a clip is over its per-subset cap or Gemma's 30s hard limit, a further
# trimmed-clip cache) that lives outside $DATA_ROOT, so it needs its own
# allowed root.
if [[ "$DATA_BACKEND" == "huggingface" || "$DATA_BACKEND" == "hf" ]]; then
  # Any clip over Gemma's 30s cap (tiers 1-3 sends no per-subset protocol
  # cap, so this is the only limit in play here) still needs a trimmed-clip
  # cache dir, and it must fall under the same allowed root as $DATA_ROOT --
  # otherwise vLLM 400s every trimmed request with "must be a subpath of
  # --allowed-local-media-path", which silently zeroes out predictions for
  # every over-cap row (data lost this way is not visible until someone
  # greps predictions.jsonl for a non-null "error"). Use the common scratch
  # root as the allowed path so both $DATA_ROOT and the trimmed-cache dir
  # (which lives outside $DATA_ROOT, a shared read-only-in-spirit HF
  # dataset snapshot) are covered.
  readonly ALLOWED_MEDIA_PATH="$SCRATCH_ROOT"
  mkdir -p "$GEMMA4_RUN_ROOT/trimmed"
  export GEMMA4_TRIMMED_CACHE_DIR="$GEMMA4_RUN_ROOT/trimmed"
else
  readonly ALLOWED_MEDIA_PATH="$GEMMA4_RUN_ROOT"
  mkdir -p "$GEMMA4_RUN_ROOT/audio-cache" "$GEMMA4_RUN_ROOT/trimmed" \
    "$SERVER_RUNTIME/cache/empty-gcloud"
  export CLOUDSDK_CONFIG="$SERVER_RUNTIME/cache/empty-gcloud"
  export ALP_DATA_HOME="${BEANS_NEXT_ALP_DATA_HOME:-$BEANS_ZERO_ESP_DATA_ROOT}"
  export BEANS_NEXT_ESP_AUDIO_CACHE_DIR="$GEMMA4_RUN_ROOT/audio-cache"
  export BEANS_NEXT_ESP_DATA_WORKERS="${BEANS_NEXT_ESP_DATA_WORKERS:-12}"
  export GEMMA4_TRIMMED_CACHE_DIR="$GEMMA4_RUN_ROOT/trimmed"
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
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "$label process exited before $url became healthy" >&2
      return 1
    fi
    if curl --fail --silent "$url" >/dev/null; then return 0; fi
    sleep 5
  done
  echo "Timed out while waiting for $label at $url" >&2
  return 1
}

export GEMMA4_ADAPTER_STUB=0
export GEMMA4_BACKEND
export GEMMA4_MODEL_ID="$MODEL_PATH"
export GEMMA4_MODEL_REVISION="$MODEL_REVISION"
export GEMMA4_MAX_BATCH_SIZE="$CONCURRENCY"
export GEMMA4_MAX_CONCURRENCY="$CONCURRENCY"
export GEMMA4_UPSTREAM_TIMEOUT_SEC=300
export GEMMA4_UPSTREAM_RETRIES=2
export GEMMA4_EXTRA_BODY_JSON
export GEMMA4_MAX_AUDIO_SECONDS="${BEANS_NEXT_GEMMA4_MAX_AUDIO_SECONDS:-30}"
export GEMMA4_AUDIO_POSITION="${BEANS_NEXT_GEMMA4_AUDIO_POSITION:-after_text}"
export PORT="$ADAPTER_PORT"

# vllm==0.28.0's bundled native extension (_C_stable_libtorch) links against
# libcudart.so.13 even though the resolved torch wheel is cu129 (so.12); the
# cu13 runtime ships as its own pip package (nvidia-cu13) but isn't on the
# loader path by default. Without this, `vllm serve` dies at import time
# with "ImportError: libcudart.so.13: cannot open shared object file".
readonly SERVER_SITE_PACKAGES="$SERVER_RUNTIME/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$SERVER_SITE_PACKAGES/nvidia/cu13/lib:$SERVER_SITE_PACKAGES/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

# flashinfer's sampler JIT-compiles a CUDA kernel on first use, which needs
# nvcc/a full CUDA toolkit under $CUDA_HOME (not just the runtime libs pip
# installs, and not present on this cluster's GPU compute nodes). Fall back
# to vLLM's native-torch sampler, which needs no compilation.
export VLLM_USE_FLASHINFER_SAMPLER=0

if [[ "$GEMMA4_BACKEND" == "vllm" ]]; then
  "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name gemma4_audio \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
    --reasoning-parser gemma4 \
    --allowed-local-media-path "$ALLOWED_MEDIA_PATH" \
    >"$JOB_ROOT/vllm.log" 2>&1 &
  vllm_pid="$!"
  wait_for_url "http://127.0.0.1:$VLLM_PORT/health" "vLLM" "$vllm_pid" 360

  export GEMMA4_UPSTREAM_BASE_URL="http://127.0.0.1:$VLLM_PORT"
  export GEMMA4_MODEL_ID=gemma4_audio
else
  # transformers backend: no vLLM server at all. The launcher loads
  # Gemma4UnifiedForConditionalGeneration + AutoProcessor in-process on the
  # single allocated GPU and serves /predict directly.
  export GEMMA4_ALLOWED_MEDIA_PATH="$ALLOWED_MEDIA_PATH"
fi

"$SERVER_PYTHON" examples/servers/gemma4/serve.py \
  >"$JOB_ROOT/adapter.log" 2>&1 &
adapter_pid="$!"
wait_for_url "http://127.0.0.1:$ADAPTER_PORT/health" "Gemma4 adapter" "$adapter_pid" 180

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
  if [[ "$GEMMA4_BACKEND" == "vllm" ]]; then
    curl --silent "http://127.0.0.1:$VLLM_PORT/metrics" \
      >"$out_dir/vllm_metrics_before.prom" || true
  fi
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
  if [[ "$GEMMA4_BACKEND" == "vllm" ]]; then
    curl --silent "http://127.0.0.1:$VLLM_PORT/metrics" \
      >"$out_dir/vllm_metrics_after.prom" || true
  fi
}

cat >"$JOB_ROOT/run.env" <<EOF
code_commit=$CODE_COMMIT
dataset_root=$DATA_ROOT
data_backend=$DATA_BACKEND
data_revision=${DATA_REVISION:-none}
model_id=google/gemma-4-12B-it
model_path=$MODEL_PATH
model_revision=$MODEL_REVISION
gemma4_backend=$GEMMA4_BACKEND
run_mode=$RUN_MODE
suite=$SUITE
decoding_arm=$DECODING_ARM
audio_position=$GEMMA4_AUDIO_POSITION
max_audio_seconds=$GEMMA4_MAX_AUDIO_SECONDS
thinking=disabled
max_length_seconds=honoured
resume_from=${RESUME_FROM:-none}
slurm_job_id=$SLURM_JOB_ID
slurm_partition=${SLURM_JOB_PARTITION:-unknown}
slurm_node=${SLURMD_NODENAME:-unknown}
max_model_len=$MAX_MODEL_LEN
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
concurrency=$CONCURRENCY
vllm_port=${VLLM_PORT:-none}
adapter_port=$ADAPTER_PORT
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv \
    > "$JOB_ROOT/apocrita_context.json.gpu.csv" 2>&1 || true
fi

if [[ "$RUN_MODE" == "probe" ]]; then
  # Decision-1 audio-sensitivity probe: 12 rows, 3 conditions each (real
  # clip / digital silence / a different clip with a different ground-truth
  # label). Exits non-zero if audio does not appear to reach the model, which
  # decides GEMMA4_BACKEND for every run after it.
  set +e
  "$RUNNER_PYTHON" "$REPO/examples/apocrita/gemma4_audio_probe.py" \
    --predict-url "http://127.0.0.1:$ADAPTER_PORT/predict" \
    --model-path "$MODEL_PATH" \
    --output-dir "$JOB_ROOT/probe"
  probe_status=$?
  set -e
  echo "results=$JOB_ROOT"
  exit "$probe_status"
elif [[ "$RUN_MODE" == "smoke" ]]; then
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
  # 1h queue with --resume-from); BEANS_NEXT_GEMMA4_FULL_OUTPUT_DIR lets each
  # submission in the chain write into the same persistent output dir instead
  # of a fresh one under this job's own JOB_ROOT.
  run_benchmark \
    "${BEANS_NEXT_GEMMA4_FULL_OUTPUT_DIR:-$JOB_ROOT/full}" \
    "${MODEL_TAG}_full_${DECODING_ARM}"
elif [[ "$RUN_MODE" == "beans_zero" ]]; then
  # Same chaining pattern as `full`, but for beans_zero_core (esp_data
  # backend) instead of beans_next_tiers_1_3_hf.
  run_benchmark \
    "${BEANS_NEXT_GEMMA4_BEANS_ZERO_OUTPUT_DIR:-$JOB_ROOT/beans_zero_greedy}" \
    "${MODEL_TAG}_beans_zero_${DECODING_ARM}"
else
  echo "BEANS_NEXT_RUN_MODE must be probe, smoke, pilot, full, or beans_zero, got: $RUN_MODE" >&2
  exit 1
fi

echo "results=$JOB_ROOT"
