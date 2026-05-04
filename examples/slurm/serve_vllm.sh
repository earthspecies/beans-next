#!/usr/bin/env bash
# SLURM job: start vLLM model server + BEANS-Next adapter sidecar.
#
# The adapter sidecar translates predictions_v1 <-> OpenAI Chat Completions.
# Works with Qwen3-Omni, Qwen2-Audio, or any audio model vLLM supports.
#
# Submit:
#   VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct sbatch examples/slurm/serve_vllm.sh
#   VLLM_OMNI=1 VLLM_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct sbatch examples/slurm/serve_vllm.sh
#
# For multi-GPU tensor parallelism, set VLLM_TENSOR_PARALLEL_SIZE and
# increase --gpus accordingly.

# Partition preference: a100-40 first, h100-80 fallback.
# Check availability before submitting: sinfo; squeue --me
# To use h100-80: sbatch --partition=h100-80 examples/slurm/serve_vllm.sh
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next serve vllm"

set -euo pipefail

# In Slurm, scripts are copied to a spool dir before execution, so $0 does not point at the repo.
# Use the submission directory as the repo root (we submit from the repo root in all runbooks).
REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Fixed ports (do not change for BEANS-Next lanes).
# - Adapter port is the one beans-next talks to (writes to URL file).
# - vLLM port is bound to 127.0.0.1 and used only by the adapter sidecar.
#
# D3 requirement: fixed port ONLY at 19082.
ADAPTER_PORT="19082"
if [[ -n "${BEANS_PRO_PORT:-}" && "${BEANS_PRO_PORT}" != "${ADAPTER_PORT}" ]]; then
  echo "ERROR: BEANS_PRO_PORT must be ${ADAPTER_PORT} (got: ${BEANS_PRO_PORT})." >&2
  exit 2
fi
export BEANS_PRO_PORT="${ADAPTER_PORT}"
VLLM_PORT="${VLLM_PORT:-19083}"

# Ensure all `uv` operations use a job-scoped environment on node-local scratch.
# Note: some cluster images ship Python without `pip`/`ensurepip`. We install packages
# via `uv pip` into this environment, which does not require `pip` to be present.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/${SLURM_JOB_ID}}"

URL_DIR="${BEANS_PRO_URL_DIR:-$HOME/beans-next-launchers}"
mkdir -p "$URL_DIR"
URL_FILE="$URL_DIR/${SLURM_JOB_ID}.url"
rm -f "$URL_FILE"

# NOTE: Use BEANS_PRO_* overrides so callers can safely force caches away from
# `/scratch/.cache` (which is often full on shared nodes) without having to
# stomp cluster-wide HF_* environment variables.
export HF_HOME="${BEANS_PRO_HF_HOME:-/scratch/shared/hf_cache}"
export HF_HUB_CACHE="${BEANS_PRO_HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${BEANS_PRO_HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${BEANS_PRO_TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_XET_CACHE="${BEANS_PRO_HF_XET_CACHE:-$HF_HOME/xet}"
export XDG_CACHE_HOME="${BEANS_PRO_XDG_CACHE_HOME:-$HF_HOME/xdg}"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME"

# Some downstream libraries (notably huggingface_hub) may see unexpanded `$USER`
# when paths are passed via Slurm env-vars. Ensure `$USER` is expanded here.
export HF_HOME="${HF_HOME//\$\{USER\}/$USER}"
export HF_HOME="${HF_HOME//\$USER/$USER}"
export HF_HUB_CACHE="${HF_HUB_CACHE//\$\{USER\}/$USER}"
export HF_HUB_CACHE="${HF_HUB_CACHE//\$USER/$USER}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE//\$\{USER\}/$USER}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE//\$USER/$USER}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE//\$\{USER\}/$USER}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE//\$USER/$USER}"
export HF_XET_CACHE="${HF_XET_CACHE//\$\{USER\}/$USER}"
export HF_XET_CACHE="${HF_XET_CACHE//\$USER/$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME//\$\{USER\}/$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME//\$USER/$USER}"

# Same for free-form vLLM CLI args (commonly includes `--download-dir /home/$USER/...`).
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS//\$\{USER\}/$USER}"
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS//\$USER/$USER}"
  # If the submitter escaped the dollar sign (e.g. "\$USER"), expand that too.
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS//\\\$USER/$USER}"
  export VLLM_EXTRA_ARGS
fi
VLLM_MODEL_ID="${VLLM_MODEL_ID:?VLLM_MODEL_ID must be set (e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct)}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_ALLOWED_LOCAL_MEDIA_PATH="${VLLM_ALLOWED_LOCAL_MEDIA_PATH:-/}"
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-}"
VLLM_OMNI="${VLLM_OMNI:-0}"
VLLM_OMNI_INSTALL="${VLLM_OMNI_INSTALL:-0}"
VLLM_INSTALL_VERSION="${VLLM_INSTALL_VERSION:-0.19.0}"

cd "$REPO/examples/servers/vllm"

# Cap audio duration for Qwen omni models to reduce multimodal token pressure.
# Mirrors the OpenAI proxy's OPENAI_PROXY_MAX_AUDIO_SECONDS behavior.
if [[ "${VLLM_MODEL_ID:-}" == Qwen/Qwen3-Omni-* || "${VLLM_MODEL_ID:-}" == Qwen/Qwen2.5-Omni-* ]]; then
  export VLLM_ADAPTER_MAX_AUDIO_SECONDS="${VLLM_ADAPTER_MAX_AUDIO_SECONDS:-30}"
  export VLLM_ADAPTER_CANONICALIZE_WAV="${VLLM_ADAPTER_CANONICALIZE_WAV:-1}"
fi

# ---------------------------------------------------------------------------
# /tmp guard (best-effort).
#
# vLLM-Omni's Speech API currently uses a hard-coded `/tmp/voice_samples` path
# during server initialization, even for text-only pipelines. On this cluster,
# `/tmp` is sometimes left full by previous jobs and `mkdir("/tmp/voice_samples")`
# fails with Errno 28. We do a conservative cleanup of common per-job temp dirs
# and redirect as much as possible to node-local scratch.
# ---------------------------------------------------------------------------
rm -rf /tmp/voice_samples 2>/dev/null || true

# Prefer node-local scratch for tmp/cache when available.
SCRATCH_CACHE_ROOT=""
if [[ -d "/scratch" ]]; then
  SCRATCH_CACHE_ROOT="/scratch/.cache"
  mkdir -p "$SCRATCH_CACHE_ROOT" 2>/dev/null || true
fi

# TMPDIR is respected by many libraries (uv, HF hub temp files, etc.).
if [[ -n "$SCRATCH_CACHE_ROOT" ]]; then
  export TMPDIR="${TMPDIR:-$SCRATCH_CACHE_ROOT/tmp/beans-next-${SLURM_JOB_ID}}"
else
  export TMPDIR="${TMPDIR:-/home/$USER/tmp/beans-next-${SLURM_JOB_ID}}"
fi
mkdir -p "$TMPDIR" 2>/dev/null || true

# vLLM-Omni Speech API currently hard-codes /tmp/voice_samples.
# Workaround: create the directory entry in /tmp, but back it by scratch via symlink.
VOICE_SAMPLES_BACKING=""
if [[ -n "$SCRATCH_CACHE_ROOT" ]]; then
  VOICE_SAMPLES_BACKING="$SCRATCH_CACHE_ROOT/voice_samples/${SLURM_JOB_ID}"
else
  VOICE_SAMPLES_BACKING="/home/$USER/tmp/voice_samples/${SLURM_JOB_ID}"
fi
mkdir -p "$VOICE_SAMPLES_BACKING" 2>/dev/null || true

if ! ln -s "$VOICE_SAMPLES_BACKING" /tmp/voice_samples 2>/dev/null; then
  # If symlink fails (e.g. /tmp full), try a plain directory as a fallback.
  if ! mkdir -p /tmp/voice_samples 2>/dev/null; then
    echo "BEANS_PRO_TMP_BLOCKER: cannot create /tmp/voice_samples (disk full?)." >&2
    echo "BEANS_PRO_TMP_BLOCKER: TMPDIR=$TMPDIR" >&2
    echo "BEANS_PRO_TMP_BLOCKER: VOICE_SAMPLES_BACKING=$VOICE_SAMPLES_BACKING" >&2
    df -h /tmp 2>/dev/null || true
    [[ -d /scratch ]] && df -h /scratch 2>/dev/null || true
    exit 86
  fi
fi

# Ensure multiple allocated GPUs are visible to subprocesses.
# Some Slurm setups allocate multiple GPUs (AllocTRES) but do not set
# CUDA_VISIBLE_DEVICES consistently; vLLM/vllm-omni rely on it for local_rank.
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}"
fi

# ---------------------------------------------------------------------------
# Scratch disk guard (check → prune safe caches → fail loudly if still low).
# ---------------------------------------------------------------------------
source "$REPO/examples/slurm/scratch_guard.sh"
if [[ "$VLLM_MODEL_ID" == Qwen/* ]]; then
  beans_next_scratch_guard "qwen" "$VLLM_MODEL_ID" "$HF_HOME"
else
  beans_next_scratch_guard "vllm" "$VLLM_MODEL_ID" "$HF_HOME"
fi

# Prefer Python 3.11+ for this project. Some compute nodes may not have a compatible system
# interpreter, so allow uv to download a managed Python when needed.
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_PRO_UV_PYTHON:-3.11}"

# On clusters where compute nodes cannot reach PyPI, installs inside the job may fail.
# Set `BEANS_PRO_SKIP_UV_SYNC=1` if you have pre-built the environment on a shared filesystem.
if [[ "$VLLM_OMNI_INSTALL" == "1" ]]; then
  uv venv --python 3.12 --seed "$UV_PROJECT_ENVIRONMENT"
  uv pip install --python "${UV_PROJECT_ENVIRONMENT%/}/bin/python" \
    fastapi==0.115.12 \
    'uvicorn[standard]==0.34.2' \
    pydantic==2.11.7 \
    httpx==0.28.1
  uv pip install --python "${UV_PROJECT_ENVIRONMENT%/}/bin/python" \
    "vllm==${VLLM_INSTALL_VERSION}" --torch-backend=auto
  uv pip install --python "${UV_PROJECT_ENVIRONMENT%/}/bin/python" \
    "vllm-omni==${VLLM_OMNI_VERSION:-0.18.0}" qwen-omni-utils
elif [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
  uv sync --group upstream
else
  echo "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync'"
fi

PYTHON_EXE="${UV_PROJECT_ENVIRONMENT%/}/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "ERROR: Python executable not found in UV_PROJECT_ENVIRONMENT: $PYTHON_EXE" >&2
  exit 1
fi

REPO="$REPO" \
ADAPTER_PORT="$ADAPTER_PORT" \
VLLM_PORT="$VLLM_PORT" \
URL_FILE="$URL_FILE" \
VLLM_MODEL_ID="$VLLM_MODEL_ID" \
VLLM_TENSOR_PARALLEL_SIZE="$VLLM_TENSOR_PARALLEL_SIZE" \
VLLM_DTYPE="$VLLM_DTYPE" \
VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
VLLM_ALLOWED_LOCAL_MEDIA_PATH="$VLLM_ALLOWED_LOCAL_MEDIA_PATH" \
VLLM_LIMIT_MM_PER_PROMPT="$VLLM_LIMIT_MM_PER_PROMPT" \
VLLM_OMNI="$VLLM_OMNI" \
"$PYTHON_EXE" - <<'PY'
import os
import signal
import shlex
import shutil
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


def _pick_free_local_port(preferred: int, *, host: str = "127.0.0.1") -> int:
    """Return a free TCP port on (host, *) starting at preferred.

    We must avoid vLLM's internal "port already in use, trying port+1" behavior,
    because this launcher wires multiple components (health checks + adapter
    upstream URL) to the configured port. If vLLM silently bumps, the adapter
    will poll the wrong port and the job never becomes healthy.
    """
    for port in range(preferred, preferred + 1000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Don't set SO_REUSEADDR: we want an exclusive bind check. If another
            # process is listening on the port, bind() must fail.
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No free port found in range [{preferred}, {preferred + 999}] on {host}."
    )


def main() -> int:
    repo = os.environ["REPO"]
    adapter_port = int(os.environ["ADAPTER_PORT"])
    vllm_port = _pick_free_local_port(int(os.environ["VLLM_PORT"]))
    url_file = os.environ["URL_FILE"]
    model_id = os.environ["VLLM_MODEL_ID"]
    tp = os.environ["VLLM_TENSOR_PARALLEL_SIZE"]
    dtype = os.environ["VLLM_DTYPE"]
    max_model_len = os.environ["VLLM_MAX_MODEL_LEN"]
    allowed_media_path = os.environ["VLLM_ALLOWED_LOCAL_MEDIA_PATH"]
    limit_mm_per_prompt = os.environ.get("VLLM_LIMIT_MM_PER_PROMPT", "").strip()
    use_omni = os.environ.get("VLLM_OMNI", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")

    hostname = os.environ.get("BEANS_PRO_HOSTNAME") or _gcp_internal_ip() or socket.gethostname()
    adapter_base = f"http://127.0.0.1:{adapter_port}"
    vllm_base = f"http://127.0.0.1:{vllm_port}"
    predict_url = f"http://{hostname}:{adapter_port}/predict"

    venv_bin = os.path.join(os.environ["UV_PROJECT_ENVIRONMENT"], "bin")
    vllm_exe = os.environ.get("VLLM_EXECUTABLE") or os.path.join(venv_bin, "vllm")
    if not os.path.exists(vllm_exe):
        vllm_exe = shutil.which("vllm") or ""
    if not vllm_exe:
        raise RuntimeError(
            "vLLM console script was not found. Do not use 'python -m vllm'; "
            "installed vLLM packages commonly lack vllm.__main__. Check the "
            "launcher environment or set VLLM_EXECUTABLE."
        )

    extra_args = shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))
    vllm_cmd = [
        vllm_exe,
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(vllm_port),
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--allowed-local-media-path",
        allowed_media_path,
    ]
    if use_omni:
        vllm_cmd.append("--omni")
    if limit_mm_per_prompt:
        vllm_cmd.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])
    vllm_cmd.extend(extra_args)
    vllm_proc = subprocess.Popen(vllm_cmd)

    adapter_env = dict(os.environ)
    adapter_env.update(
        {
            "VLLM_ADAPTER_STUB": "0",
            "VLLM_UPSTREAM_BASE_URL": vllm_base,
            "VLLM_MODEL_ID": model_id,
            "PORT": str(adapter_port),
        }
    )
    adapter_proc: subprocess.Popen[str] | None = None

    def _cleanup(*_args: object) -> None:
        nonlocal adapter_proc
        for proc in [adapter_proc, vllm_proc]:
            if proc is None:
                continue
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in [adapter_proc, vllm_proc]:
            if proc is None:
                continue
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

    # Wait for vLLM health first (load can take a long time).
    vllm_health = f"{vllm_base}/health"
    vllm_deadline = time.time() + 30 * 60
    vllm_interval_sec = float(os.environ.get("BEANS_PRO_HEALTH_INTERVAL_SEC", "5"))
    while time.time() < vllm_deadline:
        if vllm_proc.poll() is not None:
            _cleanup()
            raise RuntimeError(f"vLLM exited early with code {vllm_proc.returncode}")
        if _http_ok(vllm_health, timeout_sec=1.0):
            break
        select.select([], [], [], vllm_interval_sec)
    else:
        _cleanup()
        raise RuntimeError("vLLM did not become healthy within 1800s")

    adapter_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "adapter:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(adapter_port),
            "--log-level",
            "info",
        ],
        cwd=os.path.join(repo, "examples", "servers", "vllm"),
        env=adapter_env,
    )

    adapter_health = f"{adapter_base}/health"
    adapter_deadline = time.time() + 2 * 60
    adapter_interval_sec = float(os.environ.get("BEANS_PRO_HEALTH_INTERVAL_SEC", "2"))
    while time.time() < adapter_deadline:
        if adapter_proc.poll() is not None:
            _cleanup()
            raise RuntimeError(
                f"adapter exited early with code {adapter_proc.returncode}"
            )
        if _http_ok(adapter_health, timeout_sec=1.0):
            break
        select.select([], [], [], adapter_interval_sec)
    else:
        _cleanup()
        raise RuntimeError("adapter did not become healthy within 120s")

    os.makedirs(os.path.dirname(url_file), exist_ok=True)
    tmp_path = url_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(predict_url + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, url_file)

    print(f"Adapter ready (job_id={job_id}). URL written to: {url_file}")
    print(f"Predict URL: {predict_url}")

    try:
        # Wait on adapter; if it exits, the job ends and cleanup runs.
        return adapter_proc.wait()
    finally:
        _cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

PY
