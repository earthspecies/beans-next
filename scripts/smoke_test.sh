#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP_UTC="$(date -u +"%Y-%m-%d_%H%M")"
AGENT_LOG_DIR="$REPO_ROOT/logs/agents"
AGENT_LOG_PATH="$AGENT_LOG_DIR/${TIMESTAMP_UTC}_i4-c-smoke-test_I4-C.md"

mkdir -p "$AGENT_LOG_DIR"

exec > >(tee -a "$AGENT_LOG_PATH") 2>&1

echo "# Agent log — I4-C Smoke test"
echo
echo "- Task id: I4-C"
echo "- Start (UTC): ${TIMESTAMP_UTC}"
echo "- Host: $(hostname 2>/dev/null || echo unknown)"
echo
echo "## Plan"
echo
echo "- Start dummy launcher"
echo "- Wait for /health"
echo "- Run launcher conformance check"
echo "- Run 5-sample CPU-only benchmark run and write artifacts"
echo

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1

DUMMY_PORT="${DUMMY_PORT:-8000}"
DUMMY_HOST="${DUMMY_HOST:-127.0.0.1}"
BASE_URL="http://${DUMMY_HOST}:${DUMMY_PORT}"
PREDICT_URL="${BASE_URL}/predict"

RUN_ID="${RUN_ID:-smoke-test-${TIMESTAMP_UTC}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/${RUN_ID}}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-20}"
SUITE_ID="${SUITE_ID:-beans_zero_smoke}"

DUMMY_DIR="$REPO_ROOT/examples/servers/dummy"
DUMMY_LAUNCHER_SH="$DUMMY_DIR/serve.sh"

dummy_pid=""
cleanup() {
  set +e
  echo
  echo "## Cleanup"
  if [[ -n "${dummy_pid}" ]] && kill -0 "$dummy_pid" >/dev/null 2>&1; then
    echo "- Stopping dummy launcher (pid=$dummy_pid)"
    kill "$dummy_pid" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if kill -0 "$dummy_pid" >/dev/null 2>&1; then
        sleep 0.2
      else
        break
      fi
    done
    if kill -0 "$dummy_pid" >/dev/null 2>&1; then
      echo "- Forcing dummy launcher shutdown (pid=$dummy_pid)"
      kill -9 "$dummy_pid" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

echo "## Environment"
echo
echo "- BASE_URL: ${BASE_URL}"
echo "- PREDICT_URL: ${PREDICT_URL}"
echo "- OUT_DIR: ${OUT_DIR}"
echo "- RUN_ID: ${RUN_ID}"
echo

echo "## Start dummy launcher"
echo
if [[ ! -f "$DUMMY_LAUNCHER_SH" ]]; then
  echo "ERROR: dummy launcher not found at ${DUMMY_LAUNCHER_SH}" >&2
  exit 1
fi

(
  cd "$DUMMY_DIR"
  PORT="$DUMMY_PORT" DUMMY_BIND_HOST="$DUMMY_HOST" bash ./serve.sh
) &
dummy_pid="$!"
echo "- Dummy launcher pid: $dummy_pid"

echo
echo "## Wait for launcher /health"
echo

uv run python -c "
import sys, time
from urllib.error import URLError
from urllib.request import urlopen

url = '${BASE_URL}/health'
deadline = time.time() + 10.0
last_err = None
while time.time() < deadline:
    try:
        with urlopen(url, timeout=1.0) as resp:
            if int(getattr(resp, 'status', 200)) == 200:
                print('OK: launcher is healthy')
                raise SystemExit(0)
            last_err = f'HTTP {getattr(resp, \"status\", None)}'
    except Exception as e:
        last_err = repr(e)
    time.sleep(0.2)
print(f'FAIL: launcher not healthy within timeout; last_err={last_err}', file=sys.stderr)
raise SystemExit(1)
"

echo
echo "## Conformance check"
echo

uv run bash "$REPO_ROOT/scripts/check_launcher.sh" "$BASE_URL"

echo
echo "## 5-sample benchmark run (CPU-only)"
echo
echo "- Command: uv run beans-next run --suite ${SUITE_ID} --limit 5 --predict-url ${PREDICT_URL} --run-id ${RUN_ID} -o ${OUT_DIR}"
echo "- Timeout: ${BENCH_TIMEOUT_S}s (best-effort; conformance is the hard requirement)"
echo

mkdir -p "$OUT_DIR"

run_benchmark() {
  # Prefer suite-based path (I4-D). If suite wiring is not present yet, fall back
  # to the built-in HF runner (still CPU-only).
  #
  # This step is best-effort: if HF dataset access is slow/unavailable, we still
  # want the smoke test to validate launcher endpoints + conformance.
  local rc
  if command -v timeout >/dev/null 2>&1; then
    timeout "${BENCH_TIMEOUT_S}"s uv run beans-next run \
      --suite "${SUITE_ID}" \
      --limit 5 \
      --predict-url "$PREDICT_URL" \
      --run-id "$RUN_ID" \
      -o "$OUT_DIR"
    rc=$?
    if [[ "$rc" -eq 124 ]]; then
      echo "warning: benchmark timed out after ${BENCH_TIMEOUT_S}s; skipping artifacts validation."
      echo "benchmark_status=timeout" >"$OUT_DIR/smoke_test_benchmark_status.txt"
      run_local_benchmark || true
      return 0
    fi
  else
    uv run beans-next run \
      --suite "${SUITE_ID}" \
      --limit 5 \
      --predict-url "$PREDICT_URL" \
      --run-id "$RUN_ID" \
      -o "$OUT_DIR"
    rc=$?
  fi
  return "$rc"
}

run_local_benchmark() {
  echo
  echo "## Local synthetic benchmark fallback"
  echo
  echo "- Purpose: validate BenchmarkRunner artifacts without HF network I/O."
  echo
  uv run python -c "
from pathlib import Path

from beans_next.api.types import DatasetExample
from beans_next.models.http import HttpClient
from beans_next.prompts.renderer import PromptRenderer, load_builtin_prompt_yaml
from beans_next.runner.runner import BenchmarkRunner, RunnerConfig
from beans_next.post_process.pipeline import StepSpec

out_dir = Path(r'''${OUT_DIR}''').resolve()
run_id = r'''${RUN_ID}'''

examples = []
for i in range(5):
    examples.append(
        DatasetExample(
            sample_id=f'local_smoke:{i:04d}',
            task_id='local_smoke',
            split='local',
            labels=['label_0', 'label_1'],
            metadata={'audio_path': '/dev/null'},
        )
    )

spec = load_builtin_prompt_yaml('classification_bioacoustic_v1.yaml')
renderer = PromptRenderer(spec)
cfg = RunnerConfig(
    output_dir=out_dir,
    run_id=run_id + '__local',
    parser_steps=(StepSpec('parse_labels_comma', {}),),
    cleaner_steps=(StepSpec('normalize_whitespace', {}), StepSpec('strip_eos', {})),
)
with HttpClient(r'''${PREDICT_URL}''', probe_on_init=True) as client:
    BenchmarkRunner(client, renderer, cfg).run(examples)
print('OK: local synthetic run complete')
"
  echo "benchmark_status=ok_local" >"$OUT_DIR/smoke_test_benchmark_status.txt"
}

set +e
run_benchmark
run_rc="$?"
set -e

if [[ "$run_rc" -ne 0 ]]; then
  echo
  echo "warning: suite-based run failed (rc=${run_rc}); falling back to built-in HF runner args."
  echo
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout "${BENCH_TIMEOUT_S}"s uv run beans-next run \
      --limit 5 \
      --predict-url "$PREDICT_URL" \
      --run-id "$RUN_ID" \
      -o "$OUT_DIR"
    fallback_rc=$?
  else
    uv run beans-next run \
      --limit 5 \
      --predict-url "$PREDICT_URL" \
      --run-id "$RUN_ID" \
      -o "$OUT_DIR"
    fallback_rc=$?
  fi
  set -e
  if [[ "${fallback_rc}" -ne 0 ]]; then
    echo "warning: benchmark fallback failed (rc=${fallback_rc}); continuing after conformance-only validation."
    run_local_benchmark || echo "benchmark_status=failed" >"$OUT_DIR/smoke_test_benchmark_status.txt"
  else
    echo "benchmark_status=ok_fallback" >"$OUT_DIR/smoke_test_benchmark_status.txt"
  fi
else
  echo "benchmark_status=ok" >"$OUT_DIR/smoke_test_benchmark_status.txt"
fi

echo
echo "## Artifacts"
echo
echo "- Listing: ${OUT_DIR}"
ls -la "$OUT_DIR"

echo
echo "## Completion"
echo
echo "- End (UTC): $(date -u +"%Y-%m-%d_%H%M")"
echo "- Status: success"

