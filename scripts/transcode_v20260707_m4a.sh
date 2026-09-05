#!/usr/bin/env bash
# Transcode all .m4a audio URIs referenced by v20260707 JSONLs to .wav and
# upload to a sibling GCS path, then emit a v20260707_m4afix/ directory of
# rewritten JSONLs.
#
# Uses fsspec/gcsfs (ADC via GCE metadata server) — no gcloud CLI, no user
# auth needed. Runs ffmpeg per file via Python subprocess.
#
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="beans-next m4a-transcode v20260707"

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-}"
[[ -z "$REPO" ]] && { echo "ERROR: SLURM_SUBMIT_DIR unset — submit from repo root."; exit 1; }

command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg missing on compute node"; exit 1; }

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/m4a-transcode-${SLURM_JOB_ID}}"
export UV_PYTHON="${UV_PYTHON:-3.11}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-auto}"

if [[ ! -x "${UV_PROJECT_ENVIRONMENT%/}/bin/python" ]]; then
  uv venv "$UV_PROJECT_ENVIRONMENT"
  uv pip install --python "$UV_PROJECT_ENVIRONMENT" fsspec gcsfs
fi
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

WORK_DIR="/scratch/$USER/m4a_transcode_${SLURM_JOB_ID}"
PERSIST_LOG="/home/$USER/m4a_transcode/results_${SLURM_JOB_ID}.log"
mkdir -p "$WORK_DIR" "$(dirname "$PERSIST_LOG")"

echo "[transcode] running Python transcoder (fsspec/gcsfs, no gcloud CLI)"
"$PY" - "$WORK_DIR" "$PERSIST_LOG" <<'PYEOF'
import fsspec, json, os, sys, tempfile, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

WORK_DIR, PERSIST_LOG = sys.argv[1], sys.argv[2]
TASKS = ["t1-caption","t1-description-mcq","t1-snr-mcq","t1-snr-regression",
         "t2-behavior","t2-captioning","alarm-call-presence","amphibian-presence",
         "begging-call-presence","bird-presence","call-type-fixed-vocab",
         "flight-call-presence","insect-presence","mammal-presence"]
SRC_JSONL_ROOT = "gs://foundation-model-data/synthetic/beanspro/v20260707"
DEST_WAV_ROOT = "gs://foundation-model-data/synthetic/beanspro/v20260707_m4afix/audio"
DEST_JSONL_ROOT = "gs://foundation-model-data/synthetic/beanspro/v20260707_m4afix"

fs = fsspec.filesystem("gs")

# 1. Enumerate unique .m4a URIs
seen = set()
manifest = []
for t in TASKS:
    with fsspec.open(f"{SRC_JSONL_ROOT}/{t}.jsonl", "rt") as f:
        for row in (json.loads(l) for l in f):
            uri = row.get("audio_path_original_sample_rate", "")
            if isinstance(uri, str) and uri.lower().endswith(".m4a") and uri not in seen:
                seen.add(uri)
                wav_name = os.path.basename(uri)[:-4] + ".wav"
                manifest.append((uri, f"{DEST_WAV_ROOT}/{wav_name}"))
print(f"[transcode] {len(manifest)} unique .m4a URIs to convert")

def process_one(pair):
    m4a_uri, wav_uri = pair
    wav_path_in_bucket = wav_uri[5:]
    try:
        if fs.exists(wav_path_in_bucket):
            return ("SKIP", m4a_uri, wav_uri, None)
    except Exception as exc:
        return ("FAIL_EXISTS_CHECK", m4a_uri, wav_uri, str(exc))
    with tempfile.TemporaryDirectory() as td:
        local_m4a = os.path.join(td, "in.m4a")
        local_wav = os.path.join(td, "out.wav")
        try:
            with fs.open(m4a_uri[5:], "rb") as src, open(local_m4a, "wb") as dst:
                dst.write(src.read())
        except Exception as exc:
            return ("FAIL_DOWNLOAD", m4a_uri, wav_uri, str(exc))
        r = subprocess.run(
            ["ffmpeg","-y","-nostdin","-loglevel","error",
             "-i",local_m4a,"-acodec","pcm_s16le","-ac","1",local_wav],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return ("FAIL_FFMPEG", m4a_uri, wav_uri, r.stderr[:200])
        try:
            with fs.open(wav_path_in_bucket, "wb") as dst, open(local_wav, "rb") as src:
                dst.write(src.read())
        except Exception as exc:
            return ("FAIL_UPLOAD", m4a_uri, wav_uri, str(exc))
        return ("OK", m4a_uri, wav_uri, None)

# 2. Parallel transcode
t0 = time.time()
ok = fail = skip = 0
fail_by_kind = {}
transcoded_uris = set()
with open(PERSIST_LOG, "w") as logf, ThreadPoolExecutor(max_workers=12) as ex:
    futures = [ex.submit(process_one, p) for p in manifest]
    for i, fut in enumerate(as_completed(futures), 1):
        status, m4a, wav, err = fut.result()
        logf.write(f"{status}\t{m4a}\t{wav}\t{err or ''}\n")
        logf.flush()
        if status == "OK":
            ok += 1; transcoded_uris.add(wav)
        elif status == "SKIP":
            skip += 1; transcoded_uris.add(wav)
        else:
            fail += 1
            fail_by_kind[status] = fail_by_kind.get(status, 0) + 1
        if i % 50 == 0 or i == len(futures):
            print(f"[transcode] {i}/{len(futures)}  ok={ok} skip={skip} fail={fail}  elapsed={time.time()-t0:.0f}s", flush=True)
print(f"[transcode] transcode complete: ok={ok} skip={skip} fail={fail} total={len(manifest)}  elapsed={time.time()-t0:.0f}s")
if fail_by_kind:
    print("[transcode] failures by kind:", fail_by_kind)

# 3. Rewrite JSONLs
def rewrite(uri):
    if not (isinstance(uri, str) and uri.lower().endswith(".m4a")):
        return uri
    new = f"{DEST_WAV_ROOT}/{os.path.basename(uri)[:-4]}.wav"
    return new if new in transcoded_uris else uri

for t in TASKS:
    src = f"{SRC_JSONL_ROOT}/{t}.jsonl"
    dst = f"{DEST_JSONL_ROOT}/{t}.jsonl"
    with fsspec.open(src, "rt") as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        for k in ("audio_path_original_sample_rate", "audio_path_16KHz", "audio_path_32KHz"):
            if k in r:
                r[k] = rewrite(r[k])
    with fsspec.open(dst, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[transcode] wrote {dst} (n={len(rows)})", flush=True)
print("[transcode] all done", flush=True)
PYEOF

echo "[transcode] slurm job complete"
