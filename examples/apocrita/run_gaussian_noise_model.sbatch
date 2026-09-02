#!/usr/bin/env bash
# Serve one exact audio-language model and run Gaussian ablation phases.
# Qwen3-Omni requires submission to SAE A100 80 GB (override partition/account).
#SBATCH --job-name=beans-gaussian
#SBATCH --output=/gpfs/scratch/acw777/beans-next-gaussian/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-gaussian/logs/%x.%j.err
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=10-00:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next repository root}"
readonly ROOT="/gpfs/scratch/acw777/beans-next-gaussian"
readonly MODEL_KIND="${BEANS_NEXT_MODEL_KIND:?Set naturelm-v1.0, qwen3-omni, or audio-flamingo-next}"
readonly RUN_PHASE="${BEANS_NEXT_RUN_PHASE:-pilots}"
readonly DATASET_REVISION="eee57626950ffa95a76e39c87a1de46b113dce4c"
readonly GLOBAL_SEED="${BEANS_NEXT_GAUSSIAN_NOISE_SEED:-0}"
readonly PROTOCOL_VERSION="beans-next.gaussian-noise.v1"
readonly NOISE_CACHE="$ROOT/noise-cache"
readonly URL_DIR="$ROOT/urls"
readonly JOB_ROOT="${BEANS_NEXT_JOB_ROOT:-$ROOT/results/$MODEL_KIND/$RUN_PHASE/job_${SLURM_JOB_ID}}"
readonly CORE_PYTHON="$ROOT/runtime/core/bin/python"

mkdir -p "$ROOT/logs" "$URL_DIR" "$NOISE_CACHE" "$JOB_ROOT"
test -x "$CORE_PYTHON"

module use /data/home/acw777/privatemodules
module load uv/latest

if [[ -z "${BEANS_NEXT_CODE_COMMIT:-}" ]]; then
  export BEANS_NEXT_CODE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
fi
if [[ ! "$BEANS_NEXT_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "BEANS_NEXT_CODE_COMMIT must be a full Git commit SHA" >&2
  exit 2
fi
export BEANS_NEXT_URL_DIR="$URL_DIR"
export BEANS_NEXT_HF_BEANS_NEXT_ROOT="/gpfs/scratch/acw777/esp/beans-next"
export BEANS_NEXT_HF_AUDIO_CACHE_DIR="$ROOT/cache/dataset-audio"
export BEANS_NEXT_HF_WORKERS="${SLURM_CPUS_PER_GPU:-12}"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export TMPDIR="$ROOT/cache/tmp/$SLURM_JOB_ID"
mkdir -p "$BEANS_NEXT_HF_AUDIO_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

case "$MODEL_KIND" in
  naturelm-v1.0)
    export UV_PROJECT_ENVIRONMENT="$ROOT/runtime/naturelm"
    export BEANS_NEXT_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
    export NATURELM_V1_0_LLAMA_PATH="/gpfs/scratch/acw777/beans-next-models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    export NATURELM_CODE_DIR="$ROOT/models/NatureLM-audio-c708df7a4cc294ca8d4aaf0498794b5674ce20b1"
    export NATURELM_V1_0_MODEL="EarthSpeciesProject/NatureLM-audio"
    export NATURELM_V1_0_MODEL_REVISION="77855d9b0acd039f90798dcf0861f626bfefbd6f"
    export NATURELM_V1_0_STUB=0
    export BEANS_NEXT_SKIP_REAL_DEPS_INSTALL=1
    export BEANS_NEXT_SKIP_NATURELM_RUNTIME_INSTALL=1
    export NATURELM_V1_0_MAX_BATCH_SIZE=1
    bash "$REPO/examples/slurm/serve_naturelm_v1_0.sh" &
    ;;
  audio-flamingo-next)
    export UV_PROJECT_ENVIRONMENT="$ROOT/runtime/afnext"
    export HF_HOME="$ROOT/cache/afnext-huggingface"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export XDG_CACHE_HOME="$HF_HOME/xdg"
    mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"
    export AF3_MODEL="nvidia/audio-flamingo-next-hf"
    export AF3_MODEL_REVISION="5634886e2615c2f587dcf8b93c6edfe9907930ca"
    export BEANS_NEXT_SKIP_UV_SYNC=1
    bash "$REPO/examples/slurm/serve_af3.sh" &
    ;;
  qwen3-omni)
    export UV_PROJECT_ENVIRONMENT="$ROOT/runtime/qwen"
    export HF_HOME="$ROOT/cache/qwen-huggingface"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export XDG_CACHE_HOME="$HF_HOME/xdg"
    mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"
    export BEANS_NEXT_HF_HOME="$HF_HOME"
    export BEANS_NEXT_HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
    export BEANS_NEXT_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
    export BEANS_NEXT_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
    export BEANS_NEXT_XDG_CACHE_HOME="$XDG_CACHE_HOME"
    export VLLM_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    export VLLM_MODEL_REVISION="26291f793822fb6be9555850f06dfe95f2d7e695"
    export VLLM_OMNI=1
    export VLLM_OMNI_INSTALL=0
    export VLLM_MAX_BATCH_SIZE=1
    export VLLM_MAX_MODEL_LEN=16384
    export VLLM_AUDIO_CONTENT_FORMAT=audio_url_data
    export BEANS_NEXT_SKIP_UV_SYNC=1
    export VLLM_EXTRA_ARGS="--revision $VLLM_MODEL_REVISION --stage-configs-path $REPO/examples/servers/vllm/qwen3_omni_moe_instruct_text_single_h100.yaml --download-dir $HUGGINGFACE_HUB_CACHE"
    bash "$REPO/examples/slurm/serve_vllm.sh" &
    ;;
  *)
    echo "Unsupported BEANS_NEXT_MODEL_KIND=$MODEL_KIND" >&2
    exit 2
    ;;
esac
readonly SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

readonly URL_FILE="$URL_DIR/${SLURM_JOB_ID}.url"
deadline=$((SECONDS + 3600))
while [[ ! -s "$URL_FILE" && $SECONDS -lt $deadline ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Model server exited before publishing $URL_FILE" >&2
    wait "$SERVER_PID"
  fi
  sleep 5
done
test -s "$URL_FILE"
readonly PREDICT_URL="$(head -n 1 "$URL_FILE")"

run_eval() {
  local condition="$1"
  local suite="$2"
  local limit="$3"
  local rms_dbfs="$4"
  local out_dir="$JOB_ROOT/$condition"
  local args=(
    run --predict-url "$PREDICT_URL" --backend huggingface --suite "$suite"
    --run-id "${MODEL_KIND}_${condition}_${SLURM_JOB_ID}" --output-dir "$out_dir"
    --workers "${SLURM_CPUS_PER_GPU:-12}" --hf-revision "$DATASET_REVISION"
    --preserve-file-paths --seed 0
  )
  if [[ "$condition" == noise_* || "$condition" == beans_zero_noise_smoke ]]; then
    args+=(
      --modality-mode gaussian-noise
      --gaussian-noise-cache-dir "$NOISE_CACHE"
      --gaussian-noise-seed "$GLOBAL_SEED"
      --gaussian-noise-rms-dbfs "$rms_dbfs"
      --gaussian-noise-protocol-version "$PROTOCOL_VERSION"
    )
  else
    args+=(--modality-mode audio)
  fi
  if [[ -n "$limit" ]]; then
    args+=(--limit "$limit")
  fi
  if [[ "${BEANS_NEXT_RESUME:-1}" == "1" ]]; then
    args+=(--resume-from "$out_dir")
  fi
  mkdir -p "$out_dir"
  printf '%q ' "$CORE_PYTHON" -m beans_next.cli "${args[@]}"
  printf '\n'
  time "$CORE_PYTHON" -m beans_next.cli "${args[@]}"
  if [[ "$condition" == noise_* || "$condition" == beans_zero_noise_smoke ]]; then
    local -a manifests=()
    while IFS= read -r manifest; do
      manifests+=("$manifest")
    done < <(find "$out_dir" -name gaussian_noise_manifest.json -type f | sort)
    if (( ${#manifests[@]} == 0 )); then
      echo "No Gaussian-noise manifest found under $out_dir" >&2
      return 1
    fi
    for manifest in "${manifests[@]}"; do
      "$CORE_PYTHON" "$REPO/scripts/validate_gaussian_noise_manifest.py" "$manifest"
    done
  fi
}

"$CORE_PYTHON" - <<PY
import json
from pathlib import Path
Path("$JOB_ROOT/apocrita_context.json").write_text(json.dumps({
  "schema_version": "beans_next.gaussian_noise_apocrita.v1",
  "slurm_job_id": "$SLURM_JOB_ID",
  "partition": "${SLURM_JOB_PARTITION:-unknown}",
  "node": "${SLURMD_NODENAME:-unknown}",
  "model_kind": "$MODEL_KIND",
  "run_phase": "$RUN_PHASE",
  "dataset_revision": "$DATASET_REVISION",
  "code_commit": "$BEANS_NEXT_CODE_COMMIT",
  "global_seed": "$GLOBAL_SEED",
  "protocol_version": "$PROTOCOL_VERSION",
  "predict_url": "$PREDICT_URL",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

case "$RUN_PHASE" in
  pilots)
    run_eval noise_smoke beans_next_all_tiers_hf 2 -20
    run_eval beans_zero_noise_smoke beans_zero_smoke 2 -20
    run_eval noise_sensitivity_m30 beans_next_all_tiers_hf 5 -30
    run_eval noise_sensitivity_m20 beans_next_all_tiers_hf 5 -20
    run_eval noise_sensitivity_m10 beans_next_all_tiers_hf 5 -10
    run_eval noise_throughput beans_next_all_tiers_hf 20 -20
    ;;
  full-noise)
    run_eval noise_full beans_next_all_tiers_hf "" -20
    ;;
  full-real)
    run_eval real_full beans_next_all_tiers_hf "" -20
    ;;
  *)
    echo "Unsupported BEANS_NEXT_RUN_PHASE=$RUN_PHASE" >&2
    exit 2
    ;;
esac
