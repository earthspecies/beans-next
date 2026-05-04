#!/usr/bin/env bash
# Run a BEANS-Next NatureLM-audio benchmark on a SLURM GPU node.
#
# Usage
# -----
# Full BEANS-Zero suite (both models, side-by-side):
#   sbatch scripts/slurm_naturelm_run.sh
#
# Single task, one model:
#   LAUNCHER=naturelm-v1.0 TASK=beans_zero_esc50 sbatch scripts/slurm_naturelm_run.sh
#
# Override limit (e.g. 20 samples per task for a quick validation):
#   LIMIT=20 sbatch scripts/slurm_naturelm_run.sh
#
# Environment variables (all optional):
#   LAUNCHER      naturelm-v1.0 | naturelm-v1.1 | both  (default: naturelm-v1.0)
#   TASK          eval-task id or shorthand (default: full suite beans_zero_core)
#   SUITE         suite id (default: beans_zero_core; ignored when TASK is set)
#   LIMIT         max samples per task (default: no limit = full dataset)
#   HF_TOKEN      HuggingFace token (required for naturelm-v1.1 gated weights)
#   HF_HOME       HuggingFace cache dir (default: ~/.cache/huggingface)
#   NATURELM_V1_0_MODEL   HF repo id for v1.0 (default: EarthSpeciesProject/NatureLM-audio)
#   NATURELM_V1_1_MODEL   HF repo id for v1.1 (default: EarthSpeciesProject/naturelm-audio-1.1.00-private)
#   PORT_V1_0     port for naturelm-v1.0 launcher (default: 8000)
#   PORT_V1_1     port for naturelm-v1.1 launcher (default: 8001)
#   OUT_DIR       output directory for artifacts (default: results/slurm-<SLURM_JOB_ID>)
#   NATURELM_MAX_BATCH_SIZE  batch size sent to launcher (default: 4)
#
# SLURM directives
# ----------------
# Adjust the lines below for your cluster. Common things to change:
#   --partition / -p      your GPU partition name
#   --gres                GPU type: --gres=gpu:a100:1, --gres=gpu:v100:1, etc.
#   --time                wall time: full BEANS-Zero suite is ~4-8h on A100
#   --mem                 host RAM: NatureLM-audio 8B needs ~30GB GPU VRAM + ~32GB host
#   --account / --qos     billing account if required by your cluster

#SBATCH --job-name=beans-next-naturelm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/naturelm_%j.out
#SBATCH --error=logs/slurm/naturelm_%j.err
# Uncomment and set these if your cluster requires them:
##SBATCH --partition=gpu
##SBATCH --account=myaccount
##SBATCH --qos=normal

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[beans-next] $*"; }
die() { echo "[beans-next] ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

wait_for_health() {
  local url="$1" timeout_s="${2:-300}"
  log "Waiting for $url (timeout ${timeout_s}s) …"
  python3 - <<PY
import sys, time
from urllib.error import URLError
from urllib.request import urlopen

url = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
last = "not started"
while time.time() < deadline:
    try:
        with urlopen(url, timeout=2.0) as r:
            if r.status == 200:
                print(f"  health OK after {time.time() - (deadline - float(sys.argv[2])):.0f}s")
                raise SystemExit(0)
            last = f"HTTP {r.status}"
    except SystemExit:
        raise
    except Exception as e:
        last = str(e)
    time.sleep(2.0)
print(f"  TIMEOUT — last error: {last}", file=sys.stderr)
raise SystemExit(1)
PY "$url" "$timeout_s"
}

ensure_launcher_env() {
  local dir="$1"
  if [[ -f "$dir/pyproject.toml" ]]; then
    (cd "$dir" && uv sync)
  elif [[ -f "$dir/requirements.txt" ]]; then
    (cd "$dir" && uv venv && uv pip install -r requirements.txt)
  else
    die "launcher at $dir has no pyproject.toml or requirements.txt"
  fi
}

install_gpu_deps() {
  local dir="$1"
  local gpu_req="$dir/requirements-gpu.txt"
  if [[ ! -f "$gpu_req" ]]; then
    log "No requirements-gpu.txt in $dir — skipping GPU dep install"
    return
  fi

  log "Installing GPU deps from $gpu_req …"

  # Install CUDA-enabled PyTorch first if not already present.
  # Detect CUDA version from nvidia-smi; fall back to cu121.
  local cuda_tag="cu121"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cuda_ver
    cuda_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    log "nvidia-smi driver version: ${cuda_ver:-unknown}"
    # Map common CUDA versions (adjust for your cluster)
    local nvcc_ver
    nvcc_ver="$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || true)"
    case "${nvcc_ver:-}" in
      12.4*|12.5*|12.6*) cuda_tag="cu124" ;;
      12.1*|12.2*|12.3*) cuda_tag="cu121" ;;
      11.8*)              cuda_tag="cu118" ;;
    esac
    log "Detected CUDA → using torch index tag: $cuda_tag"
  fi

  (
    cd "$dir"
    # Install torch with CUDA support first (pip won't downgrade if already correct)
    uv pip install torch>=2.1.0 torchaudio>=2.1.0 \
      --index-url "https://download.pytorch.org/whl/${cuda_tag}" || true
    # Install remaining GPU deps
    uv pip install -r requirements-gpu.txt
    # Install NatureLM-audio from GitHub (pinned to a specific SHA is recommended)
    local naturelm_sha="${NATURELM_GIT_SHA:-}"
    if [[ -n "$naturelm_sha" ]]; then
      uv pip install "git+https://github.com/earthspeciesproject/NatureLM-audio.git@${naturelm_sha}"
    else
      uv pip install "git+https://github.com/earthspeciesproject/NatureLM-audio.git"
    fi
  )
}

start_launcher() {
  local launcher_id="$1"   # naturelm-v1.0 or naturelm-v1.1
  local port="$2"
  local bind_env_key="$3"  # env var name that sets bind host inside serve.sh
  local stub_env_key="$4"  # env var name that controls stub mode
  local dir="$REPO_ROOT/examples/servers/$launcher_id"

  log "--- Starting launcher: $launcher_id on port $port ---"

  [[ -d "$dir" ]] || die "launcher directory not found: $dir"
  [[ -x "$dir/serve.sh" ]] || die "serve.sh not found or not executable: $dir/serve.sh"

  # Install base deps + GPU deps
  ensure_launcher_env "$dir"
  install_gpu_deps "$dir"

  # Start the server process (real inference mode)
  (
    cd "$dir"
    PORT="$port" \
    "${bind_env_key}=0.0.0.0" \
    "${stub_env_key}=0" \
    NATURELM_V1_0_MAX_BATCH_SIZE="${NATURELM_MAX_BATCH_SIZE:-4}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    HF_HOME="${HF_HOME:-}" \
    bash ./serve.sh
  ) &
  echo "$!"
}

stop_pid() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "Stopping pid=$pid"
    kill "$pid" 2>/dev/null || true
    local i=0
    while kill -0 "$pid" 2>/dev/null && [[ $i -lt 30 ]]; do
      sleep 0.5; ((i++))
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAUNCHER="${LAUNCHER:-naturelm-v1.0}"  # naturelm-v1.0 | naturelm-v1.1 | both
TASK="${TASK:-}"                        # empty = run full SUITE
SUITE="${SUITE:-beans_zero_core}"
LIMIT="${LIMIT:-}"                      # empty = no limit (full dataset)
PORT_V1_0="${PORT_V1_0:-8000}"
PORT_V1_1="${PORT_V1_1:-8001}"

JOB_ID="${SLURM_JOB_ID:-local-$(date +%s)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/slurm-${JOB_ID}}"

mkdir -p "$OUT_DIR" logs/slurm

# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

log "======================================================="
log "BEANS-Next NatureLM-audio SLURM run"
log "======================================================="
log "Date        : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
log "SLURM job   : ${SLURM_JOB_ID:-<local>}"
log "Node        : $(hostname)"
log "Launcher    : $LAUNCHER"
log "Suite/task  : ${TASK:-$SUITE}"
log "Limit       : ${LIMIT:-<none — full dataset>}"
log "Output      : $OUT_DIR"
log "-------------------------------------------------------"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r name mem drv; do
      log "GPU: $name | VRAM: $mem | driver: $drv"
    done
else
  log "WARNING: nvidia-smi not found — is a GPU allocated?"
fi

# ---------------------------------------------------------------------------
# Cluster setup (uncomment / adapt as needed for your cluster)
# ---------------------------------------------------------------------------

# module purge
# module load cuda/12.1 python/3.12 gcc/12

# If you use conda instead of uv on your cluster:
# conda activate beans-next-env

require_cmd uv
require_cmd python3

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if [[ "$LAUNCHER" == "naturelm-v1.1" ]] || [[ "$LAUNCHER" == "both" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    die "HF_TOKEN is required for naturelm-v1.1 (gated weights). Export it before sbatch."
  fi
fi

# ---------------------------------------------------------------------------
# Start launchers
# ---------------------------------------------------------------------------

pids=()

if [[ "$LAUNCHER" == "naturelm-v1.0" ]] || [[ "$LAUNCHER" == "both" ]]; then
  pid_v10="$(start_launcher naturelm-v1.0 "$PORT_V1_0" NATURELM_V1_0_BIND_HOST NATURELM_V1_0_STUB)"
  pids+=("$pid_v10")
  log "naturelm-v1.0 pid=$pid_v10  url=http://localhost:$PORT_V1_0"
fi

if [[ "$LAUNCHER" == "naturelm-v1.1" ]] || [[ "$LAUNCHER" == "both" ]]; then
  pid_v11="$(start_launcher naturelm-v1.1 "$PORT_V1_1" NATURELM_BIND_HOST NATURELM_STUB)"
  pids+=("$pid_v11")
  log "naturelm-v1.1 pid=$pid_v11  url=http://localhost:$PORT_V1_1"
fi

cleanup() {
  log "=== Cleanup ==="
  for pid in "${pids[@]:-}"; do stop_pid "$pid"; done
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Wait for health + conformance
# ---------------------------------------------------------------------------

HEALTH_TIMEOUT=600  # 10 min — model loading can be slow on first run

if [[ "${pid_v10:-}" != "" ]]; then
  wait_for_health "http://localhost:$PORT_V1_0/health" "$HEALTH_TIMEOUT"
  log "naturelm-v1.0 healthy"
  log "Running conformance check …"
  uv run bash "$REPO_ROOT/scripts/check_launcher.sh" "http://localhost:$PORT_V1_0" \
    || log "WARNING: conformance check failed — results may be unreliable"
fi

if [[ "${pid_v11:-}" != "" ]]; then
  wait_for_health "http://localhost:$PORT_V1_1/health" "$HEALTH_TIMEOUT"
  log "naturelm-v1.1 healthy"
  log "Running conformance check …"
  uv run bash "$REPO_ROOT/scripts/check_launcher.sh" "http://localhost:$PORT_V1_1" \
    || log "WARNING: conformance check failed — results may be unreliable"
fi

# ---------------------------------------------------------------------------
# Build beans-next run command(s)
# ---------------------------------------------------------------------------

run_one_model() {
  local model_label="$1"
  local predict_url="$2"
  local model_out="$OUT_DIR/$model_label"
  mkdir -p "$model_out"

  log "=== Benchmarking $model_label ==="
  log "predict_url : $predict_url"
  log "output      : $model_out"

  local cmd=(uv run beans-next run
    --predict-url "$predict_url"
    --run-id "slurm-${JOB_ID}-${model_label}"
    -o "$model_out"
  )

  if [[ -n "$TASK" ]]; then
    # Single eval task — resolve HF params from registry
    task_kv="$(uv run python3 "$REPO_ROOT/scripts/_resolve_task.py" "$TASK" 2>/dev/null || true)"
    if [[ -z "$task_kv" ]]; then
      # Fall back: pass suite with just this task
      cmd+=(--suite "$TASK")
    else
      eval "$task_kv"
      cmd+=(
        --hf-path "$HF_PATH"
        --hf-config "$HF_CONFIG"
        --split "$SPLIT"
        --dataset-name "$DATASET_NAME"
        --task-id "$EVAL_TASK_ID"
      )
      if [[ -n "${PROMPT_YAML:-}" ]]; then
        cmd+=(--prompt-yaml "$REPO_ROOT/beans_next/registry/prompt/${PROMPT_YAML}")
      fi
    fi
  else
    cmd+=(--suite "$SUITE")
  fi

  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  log "Command: ${cmd[*]}"
  log "---"
  time "${cmd[@]}"
  log "--- done: $model_label ---"
}

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

if [[ "$LAUNCHER" == "naturelm-v1.0" ]] || [[ "$LAUNCHER" == "both" ]]; then
  run_one_model "naturelm-v1.0" "http://localhost:$PORT_V1_0/predict"
fi

if [[ "$LAUNCHER" == "naturelm-v1.1" ]] || [[ "$LAUNCHER" == "both" ]]; then
  run_one_model "naturelm-v1.1" "http://localhost:$PORT_V1_1/predict"
fi

# ---------------------------------------------------------------------------
# Print result summary
# ---------------------------------------------------------------------------

log "=== Results summary ==="
python3 - <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summaries = sorted(out_root.rglob("summary.json"))
if not summaries:
    print("  No summary.json files found.")
    sys.exit(0)

for path in summaries:
    try:
        doc = json.loads(path.read_text())
    except Exception as e:
        print(f"  {path}: parse error: {e}")
        continue
    rel = path.relative_to(out_root)
    n = doc.get("n_samples", "?")
    n_err = doc.get("n_errors", 0)
    metrics = doc.get("metrics", {}).get("mean", {})
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
    print(f"  {rel}  n={n}  errors={n_err}  {metric_str or '(no metrics)'}")
PY "$OUT_DIR"

log "=== Artifacts written to: $OUT_DIR ==="
log "=== SLURM job $JOB_ID complete ==="
