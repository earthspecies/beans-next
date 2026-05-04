#!/usr/bin/env bash
# SLURM job: debug curl vs python urllib connectivity to a launcher.
#
# Usage:
#   SERVE_JOB_ID=57281 sbatch examples/slurm/proxydebug.sh
#
# Output:
#   /home/%u/logs/%A.log
#
# Notes:
# - This is a diagnostic helper script; it does not print any secrets.
# - It intentionally runs both curl and python urllib against /info to surface
#   proxy and resolver differences on compute nodes.
#
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next proxydebug"

set -euo pipefail

SERVE_JOB_ID="${SERVE_JOB_ID:?SERVE_JOB_ID must be set (serving job id whose .url to read)}"

URL_FILE="$HOME/beans-next-launchers/${SERVE_JOB_ID}.url"
echo "URL_FILE=${URL_FILE}"
if [[ ! -s "$URL_FILE" ]]; then
  echo "ERROR: missing or empty URL file: ${URL_FILE}" >&2
  exit 1
fi

PREDICT_URL="$(head -1 "$URL_FILE" | tr -d '[:space:]')"
BASE_URL="${PREDICT_URL%/predict}"
export BASE_URL

echo "PREDICT_URL=${PREDICT_URL}"
echo "BASE_URL=${BASE_URL}"

echo "--- env proxy vars (if any) ---"
env | grep -i proxy || true

echo "--- curl /health ---"
curl -fsS "${BASE_URL%/}/health"
echo

echo "--- curl /info ---"
curl -fsS "${BASE_URL%/}/info"
echo

echo "--- python urllib /info ---"
python3 - <<'PY'
import os
import sys
from urllib.parse import urljoin
from urllib.request import getproxies, urlopen

base = os.environ["BASE_URL"]
print("getproxies=", getproxies())
url = urljoin(base + "/", "info")
print("url=", url)
try:
    with urlopen(url, timeout=10) as resp:
        body = resp.read(2000)
        print("status=", getattr(resp, "status", None))
        print("body_head=", body.decode("utf-8", "replace"))
except Exception as exc:
    print("ERROR", repr(exc))
    sys.exit(2)
PY

