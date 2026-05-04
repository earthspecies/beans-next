#!/usr/bin/env bash
# SLURM job: start OpenAI-compatible proxy (OpenAI GPT) on a CPU node.
#
# This launcher implements the BEANS-Next `predictions_v1` HTTP contract by proxying
# to an OpenAI-compatible upstream `POST /v1/chat/completions`.
#
# Submit (from repo root):
#   sbatch examples/slurm/serve_openai_proxy_gpt4o.sh
#
# The script writes its predict URL to:
#   $HOME/beans-next-launchers/<job_id>.url
# once `/health` passes and a conformance smoke-check succeeds.
#
# Notes:
# - Secrets: this script does not echo API keys. Store keys in `~/.config/openai/cfg`.
# - Network: requires egress from the compute node to `api.openai.com`.
#
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next serve openai-proxy gpt"

set -euo pipefail

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
  echo "$(_ts) [openai-proxy][serve] $*"
}

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this job from the repo root." >&2
  exit 1
fi

# Fixed port by default; override with BEANS_PRO_PORT.
PORT="${BEANS_PRO_PORT:-19085}"

# Job-scoped uv environment on node-local scratch.
export UV_PROJECT_ENVIRONMENT="${BEANS_PRO_UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/beans-next-openai-proxy-${SLURM_JOB_ID}}"

# URL file written once healthy + checked.
URL_DIR="${BEANS_PRO_URL_DIR:-$HOME/beans-next-launchers}"
mkdir -p "$URL_DIR"
URL_FILE="$URL_DIR/${SLURM_JOB_ID}.url"
rm -f "$URL_FILE"

# Prefer Python 3.11+ for this repo.
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"
export UV_PYTHON="${BEANS_PRO_UV_PYTHON:-3.11}"

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

# Install minimal deps (FastAPI + Uvicorn + httpx live in the dev group).
if [[ "${BEANS_PRO_SKIP_UV_SYNC:-0}" != "1" ]]; then
  (cd "$REPO" && uv sync --group dev)
else
  _step "BEANS_PRO_SKIP_UV_SYNC=1 set; skipping 'uv sync --group dev'"
fi

# ---------------------------------------------------------------------------
# Match local `examples/servers/openai_compatible_proxy/serve.sh` behavior:
# - load API keys from protected cfg files if not already set
# - keep defaults in the Python launcher unless explicitly overridden
# ---------------------------------------------------------------------------
_trim_ws() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

_read_cfg_api_key() {
  local path="$1"
  shift
  [[ -f "$path" ]] || return 1

  local line key val accepted
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(_trim_ws "$line")"
    [[ -n "$line" && "${line:0:1}" != "#" ]] || continue

    if [[ "${line,,}" == export\ * ]]; then
      line="$(_trim_ws "${line:7}")"
    fi

    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*[:=][[:space:]]*(.+)[[:space:]]*$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="$(_trim_ws "${BASH_REMATCH[2]}")"
      val="${val%\'}"
      val="${val#\'}"
      val="${val%\"}"
      val="${val#\"}"
      [[ -n "$val" ]] || continue

      if [[ "${key,,}" == "api_key" ]]; then
        printf '%s' "$val"
        return 0
      fi

      for accepted in "$@"; do
        if [[ "$key" == "$accepted" ]]; then
          printf '%s' "$val"
          return 0
        fi
      done
    elif [[ "$line" != *":"* && "$line" != *"="* ]]; then
      val="$line"
      val="${val%\'}"
      val="${val#\'}"
      val="${val%\"}"
      val="${val#\"}"
      if [[ -n "$val" ]]; then
        printf '%s' "$val"
        return 0
      fi
    fi
  done < "$path"

  return 1
}

_is_gemini_upstream() {
  local base_url="${OPENAI_BASE_URL:-}"
  local model="${OPENAI_MODEL:-}"
  [[ "${base_url,,}" == *"generativelanguage.googleapis.com"* || "${model,,}" == gemini-* ]]
}

# OpenAI proxy settings.
# Slurm-specific: bind externally on the node.
export OPENAI_PROXY_BIND_HOST="${OPENAI_PROXY_BIND_HOST:-0.0.0.0}"
export OPENAI_PROXY_STUB="${OPENAI_PROXY_STUB:-0}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-audio-preview}"
# Canonicalize WAV to avoid upstream strictness on edge-case headers.
export OPENAI_PROXY_CANONICALIZE_WAV="${OPENAI_PROXY_CANONICALIZE_WAV:-1}"
# Default batch size (match local proxy defaults).
export OPENAI_PROXY_MAX_BATCH_SIZE="${OPENAI_PROXY_MAX_BATCH_SIZE:-8}"
export OPENAI_PROXY_MAX_CONCURRENCY="${OPENAI_PROXY_MAX_CONCURRENCY:-1}"

# Match local defaults unless overridden.
export OPENAI_PROXY_TIMEOUT_SEC="${OPENAI_PROXY_TIMEOUT_SEC:-120}"
export OPENAI_PROXY_RETRIES="${OPENAI_PROXY_RETRIES:-2}"
export OPENAI_AUTH_HEADER="${OPENAI_AUTH_HEADER:-Authorization}"

# Load API key from protected cfg (matches local serve.sh).
if [[ -z "${OPENAI_API_KEY:-}" ]] && _is_gemini_upstream; then
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$GEMINI_API_KEY"
  elif [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$GOOGLE_API_KEY"
  elif key="$(_read_cfg_api_key "${GEMINI_CFG_PATH:-$HOME/.config/gemini/cfg}" GEMINI_API_KEY GOOGLE_API_KEY)"; then
    export OPENAI_API_KEY="$key"
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]] && key="$(_read_cfg_api_key "${OPENAI_CFG_PATH:-$HOME/.config/openai/cfg}" OPENAI_API_KEY)"; then
  export OPENAI_API_KEY="$key"
fi

PORT="$PORT" URL_FILE="$URL_FILE" REPO="$REPO" uv run --python "$UV_PROJECT_ENVIRONMENT" python - <<'PY'
import os
import signal
import socket
import subprocess
import sys
import time
import json
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


def _http_ok(url: str, timeout_sec: float) -> bool:
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except (URLError, TimeoutError):
        return False


def _gcp_internal_ip() -> str | None:
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
    repo = os.environ["REPO"]
    port = int(os.environ.get("PORT", "19085"))
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    url_file = os.environ["URL_FILE"]
    os.makedirs(os.path.dirname(url_file), exist_ok=True)

    hostname = (
        os.environ.get("BEANS_PRO_HOSTNAME")
        or _gcp_internal_ip()
        or socket.gethostname()
    )
    base_url = f"http://127.0.0.1:{port}"
    predict_url = f"http://{hostname}:{port}/predict"

    env = dict(os.environ)
    env["PORT"] = str(port)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "examples.servers.openai_compatible_proxy.serve:app",
            "--app-dir",
            repo,
            "--host",
            env.get("OPENAI_PROXY_BIND_HOST", "0.0.0.0"),
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        env=env,
    )

    def _term(signum: int, _frame: object) -> None:
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    health_url = f"{base_url}/health"
    info_url = f"{base_url}/info"

    deadline = time.time() + 300.0
    while time.time() < deadline:
        if _http_ok(health_url, timeout_sec=2.0):
            break
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early with code={proc.returncode}")
        time.sleep(0.5)
    else:
        raise TimeoutError(f"timed out waiting for /health: {health_url}")

    try:
        with urlopen(info_url, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"/info: {body}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to fetch /info: {exc}", file=sys.stderr)

    # Real-mode prediction smoke-check (longer timeout than scripts/check_launcher.py).
    #
    # We require at least one non-empty prediction and empty `error`.
    import base64
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1 second of silence (some upstreams reject ultra-short WAVs).
        wf.writeframes(b"\x00\x00" * 16000)
    good_wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    smoke_timeout = float(os.environ.get("BEANS_PRO_SMOKE_TIMEOUT_SEC", "120"))
    predict_body = {
        "schema_version": "predictions_v1",
        "requests": [
            {
                "sample_id": "slurm_smoke_ok",
                "messages": [{"role": "user", "content": "x"}],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": good_wav_b64,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 8, "temperature": 0.0},
            }
        ],
    }
    req = Request(
        f"{base_url}/predict",
        data=json.dumps(predict_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=smoke_timeout) as resp:
            raw = resp.read()
            status = int(getattr(resp, "status", 200))
    except Exception as exc:  # noqa: BLE001
        print(f"Launcher smoke-check failed: /predict request error: {exc!r}", file=sys.stderr)
        return 1
    if status != 200:
        print(
            f"Launcher smoke-check failed: /predict status={status} body={raw[:2000]!r}",
            file=sys.stderr,
        )
        return 1
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"Launcher smoke-check failed: invalid JSON: {exc!r}", file=sys.stderr)
        return 1
    responses = parsed.get("responses")
    if not isinstance(responses, list) or not responses:
        print(f"Launcher smoke-check failed: bad responses: {parsed!r}", file=sys.stderr)
        return 1
    item = responses[0] if isinstance(responses[0], dict) else {}
    preds = item.get("predictions")
    err = item.get("error")
    if not isinstance(preds, list) or not preds or not all(isinstance(x, str) for x in preds):
        print(f"Launcher smoke-check failed: empty/invalid predictions: {item!r}", file=sys.stderr)
        return 1
    if err is not None and str(err).strip():
        print(f"Launcher smoke-check failed: non-empty error: {item!r}", file=sys.stderr)
        return 1

    with open(url_file, "w", encoding="utf-8") as f:
        f.write(predict_url + "\n")

    print(f"Wrote URL file: {url_file}")
    print(f"Predict URL: {predict_url}")

    # Block forever until Slurm cancels the job.
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited with code={proc.returncode}")
        time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())
PY

