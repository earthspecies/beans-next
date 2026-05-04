#!/usr/bin/env bash
# SLURM job: start NatureLM-audio v1.1 launcher and write its URL to a file.
#
# Weights are loaded from GCS. Set NATURELM_GCS_CHECKPOINT_URI before submitting:
#   NATURELM_GCS_CHECKPOINT_URI=gs://foundation-models/naturelm-audio-1.1/base_model/1290000 \
#     sbatch examples/slurm/serve_naturelm_v1_1.sh
#
# The launcher writes its predict URL to $BEANS_PRO_URL_DIR/<job_id>.url

# Partition preference: a100-40 first, h100-80 fallback.
# Check availability before submitting: sinfo; squeue --me
# To use h100-80: sbatch --partition=h100-80 examples/slurm/serve_naturelm_v1_1.sh
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --export=ALL
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next serve naturelm-v1.1"

set -euo pipefail

# ── GCS CHECKPOINT OVERRIDE ────────────────────────────────────────────────
# To benchmark a specific GCS checkpoint instead of HuggingFace weights, set
# NATURELM_GCS_CHECKPOINT_URI before submitting:
#
#   NATURELM_GCS_CHECKPOINT_URI=gs://foundation-models/naturelm-audio-1.5/all_backup/merged_variations_f0_v5 \
#     sbatch examples/slurm/serve_naturelm_v1_1.sh
#
# The checkpoint is downloaded to scratch and loaded via from_checkpoint_dir().
# The /info endpoint will report the full GCS URI as model and the basename as model_revision.
# ───────────────────────────────────────────────────────────────────────────

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
  echo "$(_ts) [naturelm-v1.1][serve] $*"
}

# In Slurm, scripts are copied to a spool dir before execution, so $0 does not point at the repo.
# Use the submission directory as the repo root (we submit from the repo root in all runbooks).
REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Default port selection:
# - avoid fixed-port collisions on shared GPU nodes
# - keep it deterministic per job for easier debugging
# Override any time with BEANS_PRO_PORT.
if [[ -n "${BEANS_PRO_PORT:-}" ]]; then
  PORT="$BEANS_PRO_PORT"
else
  PORT="$((18000 + (SLURM_JOB_ID % 1000)))"
fi

URL_DIR="${BEANS_PRO_URL_DIR:-$HOME/beans-next-launchers}"
mkdir -p "$URL_DIR"
URL_FILE="$URL_DIR/${SLURM_JOB_ID}.url"
rm -f "$URL_FILE"

# GitHub token plumbing for non-interactive installs on compute nodes.
# Cluster convention: ~/.config/github/github_token is readable on all nodes.
if [[ -z "${GITHUB_TOKEN:-}" && -f "$HOME/.config/github/github_token" ]]; then
  export GITHUB_TOKEN="$(< "$HOME/.config/github/github_token")"
fi

if [[ "${NATURELM_STUB_MODE:-0}" != "1" && -z "${NATURELM_GCS_CHECKPOINT_URI:-}" ]]; then
  echo "ERROR: NATURELM_GCS_CHECKPOINT_URI must be set for real inference." >&2
  echo "Example: NATURELM_GCS_CHECKPOINT_URI=gs://foundation-models/naturelm-audio-1.1/base_model/1290000" >&2
  echo "For conformance-only, set NATURELM_STUB_MODE=1." >&2
  exit 1
fi
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  echo "GitHub token loaded (len=${#GITHUB_TOKEN})."
else
  echo "WARNING: GitHub token not loaded; GitHub installs may fail on compute nodes." >&2
fi

cd "$REPO/examples/servers/naturelm-v1.1"

# Use a job-scoped environment on node-local scratch (do not rely on repo-root venv).
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/${SLURM_JOB_ID}-naturelm-v1.1}"

# Keep TMPDIR job-scoped to avoid collisions; uv cache remains shared.
export TMPDIR="${TMPDIR:-/scratch/$USER/.cache/tmp/beans-next-serve-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR" 2>/dev/null || true

# Prefer Python 3.11+ for this project. Some compute nodes may not have a compatible system
# interpreter, so allow uv to download a managed Python when needed.
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_PRO_UV_PYTHON:-3.12}"

# ---------------------------------------------------------------------------
# Scratch disk guard (check → prune safe caches → fail loudly if still low).
# ---------------------------------------------------------------------------
source "$REPO/examples/slurm/scratch_guard.sh"
# HF_HOME is optional; scratch_guard accepts an empty cache hint.
beans_next_scratch_guard "naturelm" "naturelm-v1.1" "${HF_HOME:-}"

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

# Fully-uv install: this launcher directory is a uv project (pyproject.toml).
if [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
  if [[ "${NATURELM_STUB_MODE:-0}" == "1" ]]; then
    uv sync
  else
    uv sync --group real
  fi
else
  echo "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync'" >&2
  exit 1
fi

# Real NatureLM-audio v1.1 serving requires launcher-local research code.
# Keep this strictly launcher-only: do not add `esp-research` to BEANS-Next core deps.
#
# If your cluster environment needs real (non-stub) inference, install esp-research from GitHub.
# You may override the ref/tag/SHA via ESP_RESEARCH_GIT_REF, or point at a local clone via
# ESP_RESEARCH_LOCAL_PATH (recommended on ESP infra).
#
# Authoritative NatureLM-audio inference notes (including v1.1) live in:
#   ~/code/esp-research__david-multi-audio/projects/NatureLM-audio-v1.5/
#   https://github.com/earthspecies/esp-research/blob/david-multi-audio/projects/NatureLM-audio-v1.5/
if [[ "${NATURELM_STUB_MODE:-0}" != "1" ]]; then
  _git_url_with_token() {
    # If a GitHub token is available, prefer embedding it into https:// URLs at install time
    # (do not print the resulting URL). This avoids SSH agent forwarding issues on compute nodes.
    local url="$1"
    # Allow a more specific token override, but keep GITHUB_TOKEN as the primary pattern
    # (matches serve_naturelm_v1_0.sh guidance).
    local token="${BEANS_PRO_GITHUB_TOKEN:-${GITHUB_TOKEN:-}}"
    if [[ -z "${token}" ]]; then
      printf '%s' "$url"
      return 0
    fi
    if [[ "$url" =~ ^git\\+https://github.com/ ]]; then
      printf '%s' "${url/git+https:\/\/github.com\//git+https:\/\/${token}@github.com\/}"
      return 0
    fi
    if [[ "$url" =~ ^https://github.com/ ]]; then
      printf '%s' "${url/https:\/\/github.com\//https:\/\/${token}@github.com\/}"
      return 0
    fi
    printf '%s' "$url"
  }

  if [[ -z "${ESP_RESEARCH_LOCAL_PATH:-}" && -d "$HOME/code/esp-research" ]]; then
    ESP_RESEARCH_LOCAL_PATH="$HOME/code/esp-research"
  fi

  if [[ -n "${ESP_RESEARCH_LOCAL_PATH:-}" ]]; then
    if [[ "${BEANS_PRO_SKIP_UV_PIP_INSTALL:-0}" != "1" ]]; then
      # We need esp-research to be importable *and* to have package metadata
      # available (some modules call importlib.metadata.version("esp-research")).
      # Install it into the job venv without pulling deps (avoid cluster-wide
      # resolution conflicts).
      uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -e "${ESP_RESEARCH_LOCAL_PATH}"

      # esp-research frequently imports esp-data. Install the matching packaged
      # version from ESP PyPI (avoid local checkout drift).
      ESP_PYPI_INDEX_URL="${BEANS_PRO_ESP_PYPI_INDEX_URL:-https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/}"
      # esp-data is on esp-pypi, but its transitive deps (e.g. gcsfs) are on PyPI.
      uv pip install --python "$UV_PROJECT_ENVIRONMENT" \
        --index-url "${ESP_PYPI_INDEX_URL}" \
        --extra-index-url "https://pypi.org/simple" \
        esp-data

      # esp-research audio encoders depend on avex (flashbeats branch).
      # Install without deps to avoid pulling tensorflow/CUDA-native stacks.
      if [[ "${BEANS_PRO_SKIP_UV_PIP_INSTALL_AVEX:-0}" != "1" ]]; then
        AVEX_URL="git+https://github.com/earthspecies/avex.git@flashbeats"
        uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps "$(_git_url_with_token "$AVEX_URL")"
      fi
    fi
  else
    echo "ERROR: Fully-uv mode requires ESP_RESEARCH_LOCAL_PATH for v1.1 (no VCS installs in job)." >&2
    echo "Expected it at $HOME/code/esp-research, or set ESP_RESEARCH_LOCAL_PATH explicitly." >&2
    exit 1
  fi

  # Use esp-research NatureLM-audio-v1.5 implementation (authoritative).
  # The launcher imports it from:
  #   ${ESP_RESEARCH_LOCAL_PATH}/projects/NatureLM-audio-v1.5
  # Override with ESP_RESEARCH_NATURELM_PROJECT_DIR if needed.
  if [[ -z "${ESP_RESEARCH_NATURELM_PROJECT_DIR:-}" ]]; then
    export ESP_RESEARCH_NATURELM_PROJECT_DIR="${ESP_RESEARCH_LOCAL_PATH}/projects/NatureLM-audio-v1.5"
  fi

  # NatureLM-audio dependencies:
  #
  # The launcher imports NatureLM code via esp-research project dir. In most
  # setups, the launcher venv already contains the runtime deps it needs
  # (torch/transformers/etc). Installing the full NatureLM repo requirements can
  # over-pin versions and break resolution on shared clusters.
  #
  # If you hit import errors in /health, install the repo requirements explicitly:
  #   NATURELM_INSTALL_REQUIREMENTS=1 sbatch examples/slurm/serve_naturelm_v1_1.sh
  if [[ "${NATURELM_INSTALL_REQUIREMENTS:-0}" == "1" ]]; then
    if [[ -f "${ESP_RESEARCH_NATURELM_PROJECT_DIR}/pyproject.toml" ]]; then
      (cd "${ESP_RESEARCH_NATURELM_PROJECT_DIR}" && uv sync --group real)
    elif [[ -f "${ESP_RESEARCH_NATURELM_PROJECT_DIR}/requirements.txt" ]]; then
      uv pip install --python "$UV_PROJECT_ENVIRONMENT" -r "${ESP_RESEARCH_NATURELM_PROJECT_DIR}/requirements.txt"
    else
      echo "WARNING: NatureLM-audio-v1.5 requirements not found; skipping deps install." >&2
    fi
  fi

  # Enable real inference by default when not in stub mode (launcher supports overriding).
  export NATURELM_ENABLE_INFERENCE="${NATURELM_ENABLE_INFERENCE:-1}"
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
    port = int(os.environ.get("PORT", "8001"))
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
