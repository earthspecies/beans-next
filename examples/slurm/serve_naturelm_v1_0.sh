#!/usr/bin/env bash
# SLURM job: start NatureLM-audio v1.0 launcher and write its URL to a file.
#
# Submit:
#   sbatch examples/slurm/serve_naturelm_v1_0.sh
#
# The launcher writes its predict URL to $BEANS_PRO_URL_DIR/<job_id>.url
# once /health passes. Point an inference job at that file with:
#   BEANS_PRO_URL_FILE=$HOME/beans-next-launchers/<job_id>.url sbatch examples/slurm/run_inference.sh

# Partition preference: a100-40 first, h100-80 fallback.
# Check availability before submitting: sinfo; squeue --me
# To use h100-80: sbatch --partition=h100-80 examples/slurm/serve_naturelm_v1_0.sh
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --export=ALL
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next serve naturelm-v1.0"

set -euo pipefail

# Optional debug mode.
# Enable with: BEANS_PRO_DEBUG=1 (or true/yes).
_debug_enabled() {
  case "${BEANS_PRO_DEBUG:-0}" in
    1|true|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

_ts() {
  date -Is 2>/dev/null || date
}

_step() {
  echo "$(_ts) [naturelm-v1.0][serve] $*"
}

# In Slurm, scripts are copied to a spool dir before execution, so $0 does not point at the repo.
# Use the submission directory as the repo root (we submit from the repo root in all runbooks).
REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Fixed port by default (override via BEANS_PRO_PORT).
PORT="${BEANS_PRO_PORT:-8000}"

# Ensure all `uv` operations use a job-scoped environment on node-local scratch.
# This avoids relying on a potentially stale/broken repo-local `.venv/` on the shared filesystem.
# Ensure all `uv` operations use a job-scoped environment on node-local scratch.
# Allow override via BEANS_PRO_UV_PROJECT_ENVIRONMENT, but do not inherit random ambient values.
export UV_PROJECT_ENVIRONMENT="${BEANS_PRO_UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/beans-next-serve-${SLURM_JOB_ID}}"

# Keep TMPDIR job-scoped to avoid collisions; uv cache remains shared.
export TMPDIR="${TMPDIR:-/scratch/$USER/.cache/tmp/beans-next-serve-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR" 2>/dev/null || true

# URL file written once the server is healthy.
URL_DIR="${BEANS_PRO_URL_DIR:-$HOME/beans-next-launchers}"
mkdir -p "$URL_DIR"
URL_FILE="$URL_DIR/${SLURM_JOB_ID}.url"
rm -f "$URL_FILE"

# Optional: pre-downloaded model cache on shared FS.
export HF_HOME="${HF_HOME:-/scratch/shared/hf_cache}"
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.config/huggingface/hf_token" ]]; then
  export HF_TOKEN="$(< "$HOME/.config/huggingface/hf_token")"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

# GitHub token plumbing for non-interactive installs on compute nodes.
# Matches the cluster convention: ~/.config/github/github_token is readable on all nodes.
if [[ -z "${GITHUB_TOKEN:-}" && -f "$HOME/.config/github/github_token" ]]; then
  export GITHUB_TOKEN="$(< "$HOME/.config/github/github_token")"
fi

cd "$REPO/examples/servers/naturelm-v1.0"
export NATURELM_CFG_PATH="${NATURELM_CFG_PATH:-$REPO/examples/servers/naturelm-v1.0/inference.yml}"
export NATURELM_V1_0_EXPECTED_SAMPLE_RATE_HZ="${NATURELM_V1_0_EXPECTED_SAMPLE_RATE_HZ:-16000}"
# v1.0 should use the upstream NatureLM pipeline path by default. The
# transformers AutoProcessor fallback is not compatible with this repo layout.
export NATURELM_V1_0_LOAD_MODE="${NATURELM_V1_0_LOAD_MODE:-pipeline}"

# Prefer Python 3.11+ for this project. Some compute nodes may not have a compatible system
# interpreter, so allow uv to download a managed Python when needed.
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_PRO_UV_PYTHON:-3.11}"

# ---------------------------------------------------------------------------
# Scratch disk guard (check → prune safe caches → fail loudly if still low).
# ---------------------------------------------------------------------------
source "$REPO/examples/slurm/scratch_guard.sh"
beans_next_scratch_guard "naturelm" "naturelm-v1.0" "$HF_HOME"

# Debug: increase uvicorn log verbosity and emit step markers.
if _debug_enabled; then
  export PS4='+[$(_ts)] ${BASH_SOURCE##*/}:${LINENO}: '
  set -x
  _step "DEBUG enabled (BEANS_PRO_DEBUG=1)"
  _step "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
  _step "URL_FILE: $URL_FILE"
  _step "PORT: $PORT"
fi

# Ensure the job-scoped venv exists.
if [[ ! -x "${UV_PROJECT_ENVIRONMENT%/}/bin/python" ]]; then
  uv venv "$UV_PROJECT_ENVIRONMENT"
fi

# Real-mode dependency install (heavy).
#
# `examples/servers/naturelm-v1.0/requirements.txt` is stub-only by design, so for real inference
# we install launcher-local deps here when NATURELM_V1_0_STUB is unset/falsey.
if [[ "${NATURELM_V1_0_STUB:-}" != "1" ]]; then
  echo "NatureLM v1.0 real mode selected (NATURELM_V1_0_STUB not set to 1)."
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN must be set for real NatureLM inference." >&2
    echo "ERROR: You need HuggingFace access to the gated base model:" >&2
    echo "  meta-llama/Meta-Llama-3.1-8B-Instruct" >&2
    exit 1
  fi
  if [[ "${BEANS_PRO_SKIP_REAL_DEPS_INSTALL:-0}" == "1" ]]; then
    echo "BEANS_PRO_SKIP_REAL_DEPS_INSTALL=1 set; skipping real-mode dependency install"
  else
    if [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
      uv sync --group real
    else
      echo "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync --group real'" >&2
      exit 1
    fi
  fi

  # Install NatureLM-audio code so `import NatureLM` works (required for pipeline mode).
  # Prefer a local checkout on the cluster to avoid VCS auth issues.
  if [[ -z "${NATURELM_CODE_DIR:-}" && -d "$HOME/code/NatureLM-audio" ]]; then
    export NATURELM_CODE_DIR="$HOME/code/NatureLM-audio"
  fi
  if [[ -z "${NATURELM_CODE_DIR:-}" ]]; then
    echo "ERROR: NatureLM-audio code is required for v1.0 real inference but NATURELM_CODE_DIR is not set." >&2
    echo "Expected it at $HOME/code/NatureLM-audio, or set NATURELM_CODE_DIR explicitly." >&2
    exit 1
  fi

  # Install NatureLM-audio itself (without letting it re-resolve torch).
  # Then install a minimal set of runtime deps it expects.
  uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -e "$NATURELM_CODE_DIR"
  uv pip install --python "$UV_PROJECT_ENVIRONMENT" \
    "cloudpathlib[gs]>=0.20.0" \
    "einops>=0.8.0" \
    "librosa>=0.9.2" \
    "resampy>=0.3.1" \
    "scipy>=1.14.0" \
    "pyyaml>=6.0.0" \
    "pydantic-settings>=2.7.1" \
    "datasets>=2.20.0" \
    "peft==0.11.1" \
    "tqdm>=4.66.4" \
    "click>=8.1.7"

  # beans-zero is referenced by NatureLM-audio (install via https to avoid ssh).
  uv pip install --python "$UV_PROJECT_ENVIRONMENT" \
    "beans-zero @ git+https://github.com/earthspecies/beans-zero.git@31d4487ee6452ae6c31853d45fd38b7d4150372d"
else
  echo "NatureLM v1.0 stub mode selected (NATURELM_V1_0_STUB=1)."
  if [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
    uv sync
  else
    echo "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync'" >&2
    exit 1
  fi
fi
PORT="$PORT" URL_FILE="$URL_FILE" uv run --python "$UV_PROJECT_ENVIRONMENT" python - <<'PY'
import os
import signal
import socket
import subprocess
import sys
import select
import time
from urllib.error import URLError
from urllib.request import urlopen


def _http_ok(url: str, timeout_sec: float) -> bool:
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except (URLError, TimeoutError):
        return False


def _gcp_internal_ip() -> str | None:
    """Return GCP internal IP from instance metadata, or None if not on GCP."""
    try:
        import urllib.request as _req
        r = _req.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip",
            headers={"Metadata-Flavor": "Google"},
        )
        with _req.urlopen(r, timeout=2) as resp:
            return resp.read().decode().strip() or None
    except Exception:
        return None


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    url_file = os.environ["URL_FILE"]
    url_dir = os.path.dirname(url_file)
    os.makedirs(url_dir, exist_ok=True)

    hostname = os.environ.get("BEANS_PRO_HOSTNAME") or _gcp_internal_ip() or socket.gethostname()
    base_url = f"http://127.0.0.1:{port}"
    predict_url = f"http://{hostname}:{port}/predict"

    debug = os.environ.get("BEANS_PRO_DEBUG", "").lower() in {"1", "true", "yes"}
    log_level = "debug" if debug else "info"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "serve:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--log-level",
            log_level,
        ]
    )

    def _cleanup(*_args: object) -> None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if os.path.exists(url_file):
                os.remove(url_file)
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    health_timeout_sec = float(
        os.environ.get("BEANS_PRO_LAUNCHER_HEALTH_TIMEOUT_SEC", "1800")
    )
    deadline = time.time() + health_timeout_sec
    interval_sec = float(os.environ.get("BEANS_PRO_HEALTH_INTERVAL_SEC", "5"))
    health_url = f"{base_url}/health"
    info_url = f"{base_url}/info"
    while time.time() < deadline:
        if proc.poll() is not None:
            _cleanup()
            raise RuntimeError(f"launcher exited early with code {proc.returncode}")
        if _http_ok(health_url, timeout_sec=1.0):
            break
        select.select([], [], [], interval_sec)
    else:
        _cleanup()
        raise RuntimeError(
            f"launcher did not become healthy within {int(health_timeout_sec)}s"
        )

    tmp_path = url_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(predict_url + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, url_file)

    print(f"Launcher ready (job_id={job_id}). URL written to: {url_file}")
    print(f"Predict URL: {predict_url}")
    if debug:
        # Best-effort: print /info payload to help confirm schema + model identity.
        try:
            with urlopen(info_url, timeout=2.0) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            print(f"/info: {body}")
        except Exception as exc:
            print(f"WARNING: failed to fetch /info: {exc}", file=sys.stderr)

    try:
        return proc.wait()
    finally:
        _cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

PY
