#!/usr/bin/env bash
# Stage exact Gaussian-ablation model snapshots and reusable runtimes on a CPU node.
#SBATCH --job-name=beans-gaussian-stage
#SBATCH --output=/gpfs/scratch/acw777/beans-next-gaussian/logs/%x.%j.out
#SBATCH --error=/gpfs/scratch/acw777/beans-next-gaussian/logs/%x.%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

readonly REPO="${SLURM_SUBMIT_DIR:?Submit from the beans-next repository root}"
readonly ROOT="/gpfs/scratch/acw777/beans-next-gaussian"
readonly RUNTIME_ROOT="$ROOT/runtime"
readonly MODEL_ROOT="$ROOT/models"
readonly QWEN_HUB_CACHE="$ROOT/cache/qwen-huggingface/hub"
readonly NATURE_CODE_COMMIT="c708df7a4cc294ca8d4aaf0498794b5674ce20b1"
readonly NATURE_MODEL_REVISION="77855d9b0acd039f90798dcf0861f626bfefbd6f"
readonly LLAMA_REVISION="0e9e39f249a16976918f6564b8830bc894c89659"
readonly QWEN_REVISION="26291f793822fb6be9555850f06dfe95f2d7e695"
readonly AFNEXT_REVISION="5634886e2615c2f587dcf8b93c6edfe9907930ca"

mkdir -p "$ROOT/logs" "$RUNTIME_ROOT" "$MODEL_ROOT" "$ROOT/cache" "$QWEN_HUB_CACHE"

module use /data/home/acw777/privatemodules
module load uv/latest

export UV_CACHE_DIR="$ROOT/cache/uv"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$ROOT/cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

link_cached_snapshot() {
  local source_snapshot="$1"
  local target_repo_cache="$2"
  local revision="$3"
  test -d "$source_snapshot"
  mkdir -p "$target_repo_cache/snapshots"
  if [[ ! -e "$target_repo_cache/snapshots/$revision" ]]; then
    ln -s "$source_snapshot" "$target_repo_cache/snapshots/$revision"
  fi
  test -d "$target_repo_cache/snapshots/$revision"
  mkdir -p "$target_repo_cache/refs"
  printf '%s\n' "$revision" >"$target_repo_cache/refs/main"
}

link_cached_snapshot \
  "/gpfs/scratch/acw777/beans-next-models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/$LLAMA_REVISION" \
  "$HUGGINGFACE_HUB_CACHE/models--meta-llama--Meta-Llama-3.1-8B-Instruct" \
  "$LLAMA_REVISION"
link_cached_snapshot \
  "/gpfs/scratch/acw777/alm-eval/cache/huggingface/hub/models--nvidia--audio-flamingo-next-hf/snapshots/$AFNEXT_REVISION" \
  "$HUGGINGFACE_HUB_CACHE/models--nvidia--audio-flamingo-next-hf" \
  "$AFNEXT_REVISION"

cd "$REPO"
UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/core" uv sync --frozen --python 3.12

cd "$REPO/examples/servers/af3"
UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/afnext" uv sync --frozen --group gpu --python 3.12

cd "$REPO/examples/servers/vllm"
if [[ ! -x "$RUNTIME_ROOT/qwen/bin/python" ]]; then
  uv venv --python 3.12 --seed "$RUNTIME_ROOT/qwen"
fi
uv pip install --python "$RUNTIME_ROOT/qwen/bin/python" \
  fastapi==0.115.12 'uvicorn[standard]==0.34.2' pydantic==2.11.7 httpx==0.28.1
uv pip install --python "$RUNTIME_ROOT/qwen/bin/python" \
  'vllm==0.18.0' --torch-backend=cu128
uv pip install --python "$RUNTIME_ROOT/qwen/bin/python" \
  'vllm-omni==0.18.0' qwen-omni-utils

readonly NATURE_CODE="$MODEL_ROOT/NatureLM-audio-$NATURE_CODE_COMMIT"
if [[ ! -d "$NATURE_CODE/.git" ]]; then
  git clone https://github.com/earthspecies/naturelm-audio.git "$NATURE_CODE"
fi
cd "$NATURE_CODE"
git fetch origin "$NATURE_CODE_COMMIT"
git checkout --detach "$NATURE_CODE_COMMIT"
test "$(git rev-parse HEAD)" = "$NATURE_CODE_COMMIT"

cd "$REPO/examples/servers/naturelm-v1.0"
# This launcher does not contain a lockfile. Its direct model-runtime packages
# are pinned in the local pyproject.toml.
UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/naturelm" uv sync --group real --python 3.12
uv pip install --python "$RUNTIME_ROOT/naturelm/bin/python" --no-deps -e "$NATURE_CODE"
uv pip install --python "$RUNTIME_ROOT/naturelm/bin/python" \
  'cloudpathlib[gs]>=0.20.0' 'einops>=0.8.0' 'librosa>=0.9.2' \
  'resampy>=0.3.1' 'scipy>=1.14.0' 'pyyaml>=6.0.0' \
  'pydantic-settings>=2.7.1' 'datasets>=2.20.0' 'peft==0.11.1' \
  'tqdm>=4.66.4' 'click>=8.1.7' \
  'beans-zero @ git+https://github.com/earthspecies/beans-zero.git@31d4487ee6452ae6c31853d45fd38b7d4150372d'

export NATURE_MODEL_REVISION LLAMA_REVISION QWEN_REVISION AFNEXT_REVISION QWEN_HUB_CACHE
"$RUNTIME_ROOT/core/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

targets = (
    ("EarthSpeciesProject/NatureLM-audio", os.environ["NATURE_MODEL_REVISION"], False, None),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", os.environ["LLAMA_REVISION"], True, None),
    (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        os.environ["QWEN_REVISION"],
        False,
        os.environ["QWEN_HUB_CACHE"],
    ),
    ("nvidia/audio-flamingo-next-hf", os.environ["AFNEXT_REVISION"], True, None),
)
for repo_id, revision, local_only, cache_dir in targets:
    path = snapshot_download(
        repo_id,
        revision=revision,
        local_files_only=local_only,
        cache_dir=cache_dir,
    )
    broken = [item for item in Path(path).iterdir() if item.is_symlink() and not item.exists()]
    if broken:
        raise RuntimeError(f"{repo_id}@{revision} contains {len(broken)} broken file links")
    print(f"staged {repo_id}@{revision} -> {path}")
PY

printf '%s\n' \
  "nature_code_commit=$NATURE_CODE_COMMIT" \
  "nature_model_revision=$NATURE_MODEL_REVISION" \
  "llama_revision=$LLAMA_REVISION" \
  "qwen_revision=$QWEN_REVISION" \
  "audio_flamingo_next_revision=$AFNEXT_REVISION" \
  >"$ROOT/model_revisions.env"
