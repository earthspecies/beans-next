#!/usr/bin/env bash
#
# Scratch guard utilities for Slurm serve scripts.
# - Checks free space on /scratch
# - Optionally prunes safe-to-delete per-user scratch artifacts (old job venvs, stale lock dirs)
# - Exits with a clear "disk space blocker" marker if still insufficient
#
# Intended usage (inside a Slurm job on a compute node):
#   source "$REPO/examples/slurm/scratch_guard.sh"
#   beans_next_scratch_guard "<model_kind>" "<model_id>" "<hf_home>"
#
# model_kind: one of "vllm", "qwen", "naturelm", "af3", "unknown"
# model_id: freeform string (used for matching heuristics)
# hf_home: path to HF_HOME (may be shared or per-user scratch)
#
# Environment:
# - BEANS_PRO_SCRATCH_MIN_FREE_GB: override required space (int).
#   Use case: if model weights are already cached from a prior run, the job needs far less
#   free space (mostly venv + runtime overhead, ~15-20 GB). Set this to 20 to bypass the
#   conservative default when you know the cache is warm:
#     BEANS_PRO_SCRATCH_MIN_FREE_GB=20 sbatch --exclude=<full_node> serve_<model>.sh
#   Note: beans_next_scratch_guard() also auto-detects cached weights heuristically (hub dir
#   exists + >= 10 GB) and lowers the threshold automatically — but the env var forces it
#   even if the heuristic misses (e.g. weights stored under a non-standard path).
# - BEANS_PRO_SCRATCH_CLEAN: if "1", allow more aggressive cleanup of old per-user hf_cache dirs
set -euo pipefail

_beans_next_df_free_gb() {
  # df -BG prints free space as e.g. "78G" (integer GiB-ish).
  # shellcheck disable=SC2016
  df -BG /scratch 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}'
}

_beans_next_dir_size_gb() {
  local path="${1:?path required}"
  if [[ ! -e "$path" ]]; then
    echo 0
    return 0
  fi
  # du -sB1 gives bytes; convert to whole GB, round down.
  local bytes
  bytes="$(du -sB1 "$path" 2>/dev/null | awk '{print $1}' || echo 0)"
  if [[ -z "$bytes" ]]; then
    bytes=0
  fi
  echo $((bytes / 1024 / 1024 / 1024))
}

_beans_next_hf_hub_model_dir() {
  # HF hub cache directories are like:
  #   <HF_HOME>/hub/models--ORG--REPO
  local hf_home="${1:?hf_home required}"
  local model_id="${2:?model_id required}" # e.g. "Qwen/Qwen3-Omni-30B-A3B-Instruct"
  if [[ "$model_id" != */* ]]; then
    echo ""
    return 0
  fi
  local safe
  safe="${model_id//\//--}"
  echo "${hf_home%/}/hub/models--${safe}"
}

_beans_next_required_scratch_gb() {
  local model_kind="${1:-unknown}"
  local model_id="${2:-}"

  if [[ -n "${BEANS_PRO_SCRATCH_MIN_FREE_GB:-}" ]]; then
    echo "${BEANS_PRO_SCRATCH_MIN_FREE_GB}"
    return 0
  fi

  # Defaults should be "pragmatic": enough headroom for common cases without causing
  # needless early failure on moderately full nodes. Operator can override per job via
  # BEANS_PRO_SCRATCH_MIN_FREE_GB.
  #
  # Note: thresholds below are the "baseline" thresholds reduced by 10GB per operator request.
  case "$model_kind" in
    qwen|vllm)
      # Baseline was 110 GB → now 100 GB.
      echo 100
      ;;
    naturelm)
      # Baseline was 60 GB → now 50 GB.
      echo 50
      ;;
    af3)
      # Baseline was 55 GB → now 45 GB.
      echo 45
      ;;
    *)
      # Baseline was 50 GB → now 40 GB.
      echo 40
      ;;
  esac
}

_beans_next_required_scratch_gb_when_cached() {
  # When model weights are already present in HF hub cache, we mostly need:
  # - job-scoped venv build
  # - temporary space for runtime (logs, shards, mmap, compile caches)
  local model_kind="${1:-unknown}"

  # Cached-mode thresholds also reduced by 10GB from baseline cached thresholds:
  # - qwen/vllm: 20 → 10
  # - naturelm: 20 → 10
  # - af3: 15 → 5
  # - fallback: 15 → 5
  case "$model_kind" in
    qwen|vllm)
      echo 10
      ;;
    naturelm)
      echo 10
      ;;
    af3)
      echo 5
      ;;
    *)
      echo 5
      ;;
  esac
}

_beans_next_prune_user_venvs() {
  local venv_root="/scratch/${USER}/venvs"
  [[ -d "$venv_root" ]] || return 0

  # Remove old job-scoped envs to recover space. Only touch known naming patterns.
  # Note: keep recent envs to avoid thrash across back-to-back jobs.
  find "$venv_root" -maxdepth 1 -mindepth 1 -type d \
    \( -name 'beans-next-serve-*' -o -name '*-naturelm-v1.*' -o -name '*-vllm*' -o -name '*-af3*' \) \
    -mtime +3 -print0 2>/dev/null \
    | xargs -0r rm -rf
}

_beans_next_empty_dir_contents() {
  # Remove all contents of a directory WITHOUT deleting the directory itself.
  # This is safe for shared cache roots like /scratch/.cache/huggingface where
  # tools expect the directory to exist.
  local dir="${1:?dir required}"
  [[ -d "$dir" ]] || return 0
  find "$dir" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | xargs -0r rm -rf
}

_beans_next_prune_shared_scratch_caches() {
  # Shared cache policy:
  # - NEVER delete /scratch/.cache or cache directories themselves.
  # - If space is low, we may empty known cache contents to recover disk quickly.
  # Targets requested: Hugging Face + uv.
  local root="/scratch/.cache"
  [[ -d "$root" ]] || return 0

  _beans_next_empty_dir_contents "$root/huggingface" || true
  _beans_next_empty_dir_contents "$root/uv" || true
}

_beans_next_prune_hf_locks() {
  local hf_home="${1:-}"
  [[ -n "$hf_home" ]] || return 0
  [[ -d "$hf_home" ]] || return 0

  rm -rf "$hf_home/hub/.locks" 2>/dev/null || true
  rm -rf "$hf_home/transformers/.locks" 2>/dev/null || true

  # If present, xet cache can grow large and is safe to clear (will be refilled).
  if [[ -d "$hf_home/xet" ]]; then
    rm -rf "$hf_home/xet/"* 2>/dev/null || true
  fi
}

_beans_next_prune_old_user_hf_caches() {
  # Aggressive mode: delete *old* per-user hf_cache* dirs on scratch (not shared caches).
  # Only enabled when BEANS_PRO_SCRATCH_CLEAN=1.
  local current_hf_home="${1:-}"
  [[ "${BEANS_PRO_SCRATCH_CLEAN:-0}" == "1" ]] || return 0

  local user_root="/scratch/${USER}"
  [[ -d "$user_root" ]] || return 0

  find "$user_root" -maxdepth 1 -mindepth 1 -type d -name 'hf_cache*' -mtime +7 2>/dev/null \
    | while read -r d; do
        if [[ -n "$current_hf_home" && "$d" == "$current_hf_home" ]]; then
          continue
        fi
        rm -rf "$d" 2>/dev/null || true
      done
}

beans_next_scratch_guard() {
  local model_kind="${1:-unknown}"
  local model_id="${2:-}"
  local hf_home="${3:-}"

  if [[ ! -d "/scratch" ]]; then
    echo "WARNING: /scratch not present; skipping scratch guard."
    return 0
  fi

  local required_gb
  required_gb="$(_beans_next_required_scratch_gb "$model_kind" "$model_id")"

  # If the model appears already cached in HF hub, we can safely lower the requirement.
  # Heuristic: hub dir exists and is "large enough" (>= 10 GB).
  if [[ -n "$hf_home" && -d "${hf_home%/}/hub" && "$model_id" == */* ]]; then
    local model_dir
    model_dir="$(_beans_next_hf_hub_model_dir "$hf_home" "$model_id")"
    if [[ -n "$model_dir" && -d "$model_dir" ]]; then
      local model_gb
      model_gb="$(_beans_next_dir_size_gb "$model_dir")"
      if [[ "$model_gb" -ge 10 ]]; then
        local cached_required_gb
        cached_required_gb="$(_beans_next_required_scratch_gb_when_cached "$model_kind")"
        if [[ "$cached_required_gb" -lt "$required_gb" ]]; then
          echo "Scratch guard: model cache detected ($model_id ~${model_gb}GB at $model_dir); lowering required free space to ${cached_required_gb}GB."
          required_gb="$cached_required_gb"
        fi
      fi
    fi
  fi

  local free_gb
  free_gb="$(_beans_next_df_free_gb || echo 0)"

  if [[ -z "$free_gb" ]]; then
    free_gb=0
  fi

  if [[ "$free_gb" -ge "$required_gb" ]]; then
    echo "Scratch guard: OK (/scratch free=${free_gb}GB, required=${required_gb}GB)."
    return 0
  fi

  echo "Scratch guard: LOW SPACE (/scratch free=${free_gb}GB, required=${required_gb}GB)."
  echo "Scratch guard: attempting safe cleanup (user=$USER, hf_home=${hf_home:-<unset>})."

  _beans_next_prune_user_venvs || true
  _beans_next_prune_hf_locks "$hf_home" || true
  _beans_next_prune_old_user_hf_caches "$hf_home" || true
  _beans_next_prune_shared_scratch_caches || true

  free_gb="$(_beans_next_df_free_gb || echo 0)"
  if [[ -z "$free_gb" ]]; then
    free_gb=0
  fi

  if [[ "$free_gb" -ge "$required_gb" ]]; then
    echo "Scratch guard: recovered space (/scratch free=${free_gb}GB, required=${required_gb}GB)."
    return 0
  fi

  echo "BEANS_PRO_DISK_SPACE_BLOCKER: insufficient /scratch after cleanup."
  echo "BEANS_PRO_DISK_SPACE_BLOCKER: model_kind=${model_kind} model_id=${model_id}"
  echo "BEANS_PRO_DISK_SPACE_BLOCKER: free_gb=${free_gb} required_gb=${required_gb}"
  df -h /scratch || true
  exit 86
}

