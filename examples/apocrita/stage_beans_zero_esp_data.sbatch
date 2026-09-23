#!/usr/bin/env bash
# Mirror BEANS-Zero's esp_data (16 kHz) subtree to local scratch so the
# chained Audex evaluation reads local files instead of hitting the public
# GCS bucket on every job. Public bucket, plain HTTPS, no auth/gcloud needed.
# Uses stdlib only, so the existing runner venv's python is sufficient.

#SBATCH --job-name=beans-zero-stage
#SBATCH --output=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-runs/logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next-audex repository root}"
readonly SCRATCH_ROOT="${BEANS_NEXT_SCRATCH_ROOT:-/gpfs/scratch/acw777}"
readonly RUNNER_RUNTIME="${BEANS_NEXT_RUNTIME_ROOT:-$SCRATCH_ROOT/beans-next-runtime}"
readonly RUNNER_PYTHON="$RUNNER_RUNTIME/.venv/bin/python"
readonly OUTPUT_ROOT="${BEANS_ZERO_ESP_DATA_ROOT:-$SCRATCH_ROOT/beans-zero-esp-data}"
readonly MANIFEST="${BEANS_ZERO_STAGE_MANIFEST:-$OUTPUT_ROOT/stage-manifest.${SLURM_JOB_ID}.json}"

if [[ ! -x "$RUNNER_PYTHON" ]]; then
  echo "Missing runner Python: $RUNNER_PYTHON" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

"$RUNNER_PYTHON" "$REPO/examples/apocrita/stage_beans_zero_esp_data.py" \
  --output-root "$OUTPUT_ROOT" \
  --manifest "$MANIFEST" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"

echo "beans_zero_stage_manifest=$MANIFEST"
echo "beans_zero_esp_data_root=$OUTPUT_ROOT"
