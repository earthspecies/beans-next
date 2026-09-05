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
readonly BEANS_ZERO_DATASET_REVISION="e9e45ffff7500867c885ef74c245c07e66677084"
readonly BEANS_ZERO_HF_HOME="/gpfs/scratch/acw777/beans-next-zero-cache/huggingface"
readonly GLOBAL_SEED="${BEANS_NEXT_GAUSSIAN_NOISE_SEED:-0}"
readonly PROTOCOL_VERSION="beans-next.gaussian-noise.v1"
readonly NOISE_CACHE="$ROOT/noise-cache"
readonly URL_DIR="$ROOT/urls"
readonly JOB_ROOT="${BEANS_NEXT_JOB_ROOT:-$ROOT/results/$MODEL_KIND/$RUN_PHASE/job_${SLURM_JOB_ID}}"
readonly CORE_PYTHON="$ROOT/runtime/core/bin/python"

# The reusable core environment may have been staged from an earlier source
# archive.  Import the immutable source tree that this job was submitted from.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

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
    export SPEECH_VOICE_SAMPLES="$TMPDIR/voice_samples"
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
  local workers="${SLURM_CPUS_PER_GPU:-12}"
  if [[ "$MODEL_KIND" == "audio-flamingo-next" ]]; then
    # Keep generation serial: one CUDA assertion poisons the process, and AF-NeXT
    # does not gain useful throughput from concurrent client requests here.
    workers=1
  fi
  local args=(
    run --predict-url "$PREDICT_URL" --backend huggingface --suite "$suite"
    --run-id "${MODEL_KIND}_${condition}_${SLURM_JOB_ID}" --output-dir "$out_dir"
    --workers "$workers" --seed 0
  )
  if [[ "$MODEL_KIND" != "qwen3-omni" ]]; then
    args+=(--preserve-file-paths)
  fi
  local -a dataset_env=()
  if [[ "$suite" == "beans_zero_smoke" ]]; then
    args+=(--hf-revision "$BEANS_ZERO_DATASET_REVISION")
    dataset_env+=(
      "HF_HOME=$BEANS_ZERO_HF_HOME"
      "HUGGINGFACE_HUB_CACHE=$BEANS_ZERO_HF_HOME/hub"
      "HF_DATASETS_CACHE=$BEANS_ZERO_HF_HOME/datasets"
    )
  else
    args+=(--hf-revision "$DATASET_REVISION")
  fi
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
  time env "${dataset_env[@]}" "$CORE_PYTHON" -m beans_next.cli "${args[@]}"
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
      local -a validator_args=(--expected-rms-dbfs "$rms_dbfs")
      if [[ "$rms_dbfs" == "-10" || "$rms_dbfs" == "-10.0" ]]; then
        validator_args+=(--allow-over-full-scale)
      fi
      "$CORE_PYTHON" "$REPO/scripts/validate_gaussian_noise_manifest.py" \
        "$manifest" "${validator_args[@]}"
    done
  fi
}

run_af_boundary_pilot() {
  if [[ "$MODEL_KIND" != "audio-flamingo-next" ]]; then
    echo "af-boundary-pilot requires audio-flamingo-next" >&2
    return 2
  fi
  "$CORE_PYTHON" - <<PY
import json
import urllib.request

cases = [
    ("real-30s", "/gpfs/scratch/acw777/esp/beans-next/test/audio/c5/c54f4dd4003f46c4c54c9e35aa12251dc121bbb23e34fd365ccc80fe83f566b6.wav", 96000),
    ("noise-30s", "/gpfs/scratch/acw777/beans-next-gaussian/noise-cache/6b9ee4c9bd7723df2a17aad5c0b665356dcfba9f72596980d5a1cd35aa7450ce.wav", 96000),
    ("real-60s", "/gpfs/scratch/acw777/esp/beans-next/test/audio/c8/c8a385c91a7b4237bc0cada28635d151609e68c706cb3d9cb99ec9e2a7f31790.wav", 44100),
    ("noise-60s", "/gpfs/scratch/acw777/beans-next-gaussian/noise-cache/e550744c4879bbf039f29340deec02c0498f5ce824989fe60da2894f216b93c2.wav", 44100),
    ("real-120s", "/gpfs/scratch/acw777/esp/beans-next/test/audio/94/94b7d828d20784e45cdae0695327371ebbcd2379c59b21fbe7208bba5b2eaa73.wav", 96000),
    ("noise-120s", "/gpfs/scratch/acw777/beans-next-gaussian/noise-cache/d7aeefaf3776397ae6e402bbf2afc7158a476a5431c8da72b9c370558bb0dd75.wav", 96000),
]
for sample_id, path, sample_rate in cases:
    payload = {
        "schema_version": "predictions_v1",
        "requests": [{
            "sample_id": sample_id,
            "messages": [{
                "role": "user",
                "content": "<Audio><AudioHere></Audio>Answer briefly.",
            }],
            "audio_inputs": [{
                "payload_type": "file_path",
                "data": path,
                "sample_rate": sample_rate,
            }],
            "generation_config": {"max_tokens": 4, "temperature": 0},
        }],
    }
    request = urllib.request.Request(
        "$PREDICT_URL",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)["responses"][0]
    print(sample_id, json.dumps(result, sort_keys=True))
    if result.get("error"):
        raise SystemExit(f"Boundary pilot failed for {sample_id}: {result['error']}")
PY
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
  "beans_zero_dataset_revision": "$BEANS_ZERO_DATASET_REVISION",
  "code_commit": "$BEANS_NEXT_CODE_COMMIT",
  "global_seed": "$GLOBAL_SEED",
  "protocol_version": "$PROTOCOL_VERSION",
  "predict_url": "$PREDICT_URL",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

case "$RUN_PHASE" in
  af-boundary-pilot)
    run_af_boundary_pilot
    ;;
  pilots)
    run_eval noise_smoke beans_next_all_tiers_hf 2 -20
    run_eval beans_zero_noise_smoke beans_zero_smoke 2 -20
    run_eval noise_sensitivity_m30 beans_next_all_tiers_hf 5 -30
    run_eval noise_sensitivity_m20 beans_next_all_tiers_hf 5 -20
    run_eval noise_sensitivity_m10 beans_next_all_tiers_hf 5 -10
    run_eval noise_throughput beans_next_all_tiers_hf 20 -20
    ;;
  sensitivity-throughput)
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
