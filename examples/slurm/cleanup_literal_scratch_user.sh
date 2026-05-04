#!/usr/bin/env bash
#
# Delete the mistakenly-created literal path: /scratch/$USER
#
# This path can be created when scripts pass '/scratch/$USER' (single-quoted) to tools that
# create cache directories (HF, vLLM, etc.). This cleanup removes ONLY that literal directory.
#
# Safe to run as the target user on a compute node (no sudo needed).
set -euo pipefail

literal_dir="/scratch/\$USER"

host="$(hostname 2>/dev/null || echo unknown-host)"
if [[ -d "$literal_dir" ]]; then
  echo "[$host] Removing literal scratch dir: $literal_dir"
  rm -rf -- "$literal_dir"
  echo "[$host] Removed."
else
  echo "[$host] OK: literal scratch dir not present: $literal_dir"
fi

