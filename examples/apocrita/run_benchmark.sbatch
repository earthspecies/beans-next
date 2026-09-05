#!/usr/bin/env bash
# Run the CPU-side BEANS-Next benchmark client on Apocrita. The model server is
# submitted separately and must expose the predictions_v1 HTTP contract.
#
# Before submitting, create the log directory once:
#   mkdir -p /gpfs/scratch/acw777/beans-next-runs/logs
#
# Example:
#   BEANS_NEXT_PREDICT_URL=http://<model-node>:8000/predict \
#   sbatch examples/apocrita/run_benchmark.sbatch

#SBATCH --job-name=beans-next-eval
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit this job from the beans-next repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly DATA_ROOT="${BEANS_NEXT_HF_BEANS_NEXT_ROOT:-$SCRATCH_ROOT/esp/beans-next}"
readonly RUNTIME_ROOT="${BEANS_NEXT_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-runtime}"
readonly RESULTS_ROOT="${BEANS_NEXT_RESULTS_ROOT:-$SCRATCH_ROOT/beans-next-runs/results}"
readonly PREDICT_URL="${BEANS_NEXT_PREDICT_URL:?Set BEANS_NEXT_PREDICT_URL to the model server /predict URL}"
readonly SUITE="${BEANS_NEXT_SUITE:-beans_next_tiny}"
readonly RUN_ID="${BEANS_NEXT_RUN_ID:-apocrita_${SLURM_JOB_ID}}"
readonly OUT_DIR="${BEANS_NEXT_OUT_DIR:-$RESULTS_ROOT/$RUN_ID}"

if [[ ! -f "$DATA_ROOT/test/metadata.parquet" ]]; then
  echo "Missing dataset metadata: $DATA_ROOT/test/metadata.parquet" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/test/audio" ]]; then
  echo "Missing dataset audio directory: $DATA_ROOT/test/audio" >&2
  exit 1
fi

mkdir -p "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/tmp" "$OUT_DIR"
cd "$REPO"

module use /data/home/acw777/privatemodules
module load uv/latest

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$RUNTIME_ROOT/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNTIME_ROOT/cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_ROOT/cache/xdg}"
export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-$RUNTIME_ROOT/tmp}"
export BEANS_NEXT_HF_BEANS_NEXT_ROOT="$DATA_ROOT"
export BEANS_NEXT_HF_WORKERS="${BEANS_NEXT_HF_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
export BEANS_NEXT_OUT_DIR_RESOLVED="$OUT_DIR"
export BEANS_NEXT_SUITE_RESOLVED="$SUITE"

if [[ "${BEANS_NEXT_SKIP_UV_SYNC:-0}" != "1" ]]; then
  uv sync --frozen
fi

export BEANS_NEXT_CODE_COMMIT="$(git rev-parse HEAD)"
snapshot_metadata="$DATA_ROOT/.cache/huggingface/download/test/metadata.parquet.metadata"
export BEANS_NEXT_DATASET_COMMIT=""
export BEANS_NEXT_DATASET_METADATA_SHA256=""
if [[ -f "$snapshot_metadata" ]]; then
  export BEANS_NEXT_DATASET_COMMIT="$(sed -n '1p' "$snapshot_metadata")"
  export BEANS_NEXT_DATASET_METADATA_SHA256="$(sed -n '2p' "$snapshot_metadata")"
fi

uv run --frozen python - <<'PY'
import json
import os
from pathlib import Path

context = {
    "schema_version": "beans_next.apocrita_context.v1",
    "code_commit": os.environ["BEANS_NEXT_CODE_COMMIT"],
    "dataset_root": os.environ["BEANS_NEXT_HF_BEANS_NEXT_ROOT"],
    "dataset_commit": os.environ.get("BEANS_NEXT_DATASET_COMMIT") or None,
    "dataset_metadata_sha256": (
        os.environ.get("BEANS_NEXT_DATASET_METADATA_SHA256") or None
    ),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_node": os.environ.get("SLURMD_NODENAME"),
    "suite": os.environ["BEANS_NEXT_SUITE_RESOLVED"],
}
out = Path(os.environ["BEANS_NEXT_OUT_DIR_RESOLVED"])
(out / "apocrita_context.json").write_text(
    json.dumps(context, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

args=(
  run
  --predict-url "$PREDICT_URL"
  --backend huggingface
  --suite "$SUITE"
  --run-id "$RUN_ID"
  --output-dir "$OUT_DIR"
  --workers "${BEANS_NEXT_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
)

# Both jobs on Apocrita can read GPFS. Avoid base64 expansion and repeated audio
# reads unless the endpoint is outside Apocrita.
if [[ "${BEANS_NEXT_SHARED_FILESYSTEM:-1}" == "1" ]]; then
  args+=(--preserve-file-paths)
fi
if [[ -n "${BEANS_NEXT_LIMIT:-}" ]]; then
  args+=(--limit "$BEANS_NEXT_LIMIT")
fi
if [[ "${BEANS_NEXT_RESUME:-1}" == "1" ]]; then
  args+=(--resume-from "$OUT_DIR")
fi

echo "repo_commit=$(git rev-parse HEAD)"
echo "dataset_root=$DATA_ROOT"
echo "suite=$SUITE run_id=$RUN_ID output_dir=$OUT_DIR"
uv run --frozen beans-next "${args[@]}"
