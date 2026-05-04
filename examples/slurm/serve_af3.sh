#!/usr/bin/env bash
# SLURM job: start Audio Flamingo Next launcher and write its URL to a file.
#
# nvidia/audio-flamingo-next-hf: 8B BF16, requires ~20 GB VRAM.
# NVIDIA OneWay Noncommercial License — non-commercial research use only.
#
# Submit:
#   sbatch examples/slurm/serve_af3.sh
#
# Override model:
#   AF3_MODEL=nvidia/audio-flamingo-next-hf sbatch examples/slurm/serve_af3.sh

# Partition preference: a100-40 first, h100-80 fallback.
# Check availability before submitting: sinfo; squeue --me
# To use h100-80: sbatch --partition=h100-80 examples/slurm/serve_af3.sh
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next serve af3"

set -euo pipefail

# In Slurm, scripts are copied to a spool dir before execution, so $0 does not point at the repo.
# Use the submission directory as the repo root (we submit from the repo root in all runbooks).
REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Fixed port by default (override via BEANS_PRO_PORT).
PORT="${BEANS_PRO_PORT:-8002}"

# Ensure all `uv` operations use a job-scoped environment on node-local scratch.
# Note: some cluster images ship Python without `pip`/`ensurepip`. We install packages
# via `uv pip` into this environment, which does not require `pip` to be present.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/${SLURM_JOB_ID}}"

# Keep TMPDIR job-scoped to avoid collisions; uv cache remains shared.
export TMPDIR="${TMPDIR:-/scratch/$USER/.cache/tmp/beans-next-serve-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR" 2>/dev/null || true

URL_DIR="${BEANS_PRO_URL_DIR:-$HOME/beans-next-launchers}"
mkdir -p "$URL_DIR"
URL_FILE="$URL_DIR/${SLURM_JOB_ID}.url"
rm -f "$URL_FILE"

export HF_HOME="${HF_HOME:-/scratch/shared/hf_cache}"
export AF3_MODEL="${AF3_MODEL:-nvidia/audio-flamingo-next-hf}"

cd "$REPO/examples/servers/af3"

# ---------------------------------------------------------------------------
# Scratch disk guard (check → prune safe caches → fail loudly if still low).
# ---------------------------------------------------------------------------
source "$REPO/examples/slurm/scratch_guard.sh"
beans_next_scratch_guard "af3" "$AF3_MODEL" "$HF_HOME"

# Prefer Python 3.11+ for this project. Some compute nodes may not have a compatible system
# interpreter, so allow uv to download a managed Python when needed.
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_PRO_UV_PYTHON:-3.11}"

# On clusters where compute nodes cannot reach PyPI, `uv sync` inside the job may fail.
# Set `BEANS_PRO_SKIP_UV_SYNC=1` if you have pre-built the environment on a shared filesystem.
if [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
  if [[ "${AF3_STUB:-0}" == "1" ]]; then
    uv sync
  else
    uv sync --group gpu
  fi
else
  echo "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync'"
fi

# ---------------------------------------------------------------------------
# GPU sanity check (writes to the Slurm job log)
# ---------------------------------------------------------------------------
if [[ "${AF3_STUB:-0}" != "1" ]]; then
  echo "AF3 GPU sanity check (torch/CUDA):"
  uv run python - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"ERROR: failed to import torch: {exc!r}", file=sys.stderr)
    raise

print(f"torch.__version__={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
    # Force a tiny CUDA allocation so driver/runtime mismatches fail early.
    _ = torch.zeros(1, device="cuda")
    print("cuda_allocation_ok=True")
else:
    print("cuda_allocation_ok=False")
PY
  echo "AF3 GPU sanity check complete."
fi
PORT="$PORT" URL_FILE="$URL_FILE" uv run python - <<'PY'
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
    port = int(os.environ.get("PORT", "8002"))
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    url_file = os.environ["URL_FILE"]
    url_dir = os.path.dirname(url_file)
    os.makedirs(url_dir, exist_ok=True)

    hostname = os.environ.get("BEANS_PRO_HOSTNAME") or _gcp_internal_ip() or socket.gethostname()
    base_url = f"http://127.0.0.1:{port}"
    predict_url = f"http://{hostname}:{port}/predict"

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
            "info",
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

    # AF3 can take several minutes to load.
    health_timeout_sec = float(
        os.environ.get("BEANS_PRO_LAUNCHER_HEALTH_TIMEOUT_SEC", "1800")
    )
    deadline = time.time() + health_timeout_sec
    interval_sec = float(os.environ.get("BEANS_PRO_HEALTH_INTERVAL_SEC", "5"))
    health_url = f"{base_url}/health"
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
