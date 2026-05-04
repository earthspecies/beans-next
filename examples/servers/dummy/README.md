# BEANS-Next dummy launcher

CPU-only reference server implementing the `predictions_v1` HTTP contract (`POST /predict`, `GET /info`, `GET /health`). It does not load weights or decode audio; predictions are deterministic strings derived from each request item.

## Setup

```bash
cd examples/servers/dummy
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Using **uv** (no `pyproject.toml` here; only this folder’s `requirements.txt`):

```bash
cd examples/servers/dummy
uv venv
uv pip install -r requirements.txt
. .venv/bin/activate
./serve.sh
```

Environment:

- `PORT` — listen port (default `8000`)
- `DUMMY_BIND_HOST` — bind address (default `127.0.0.1`)

## Run

```bash
export PORT=8000
./serve.sh
```

Or directly:

```bash
python -m uvicorn serve:app --host 127.0.0.1 --port 8000
```

## Contract summary

- **`max_batch_size`:** `32` (see `serve.py` → `MAX_BATCH_SIZE`)
- **`audio_payload_types`:** `base64_wav`, `file_path`, `file_url`
- **Batching:** one response per request `sample_id`; response order may differ from request order.
- **413:** batch larger than `max_batch_size`.
- **Per-sample errors:** HTTP 200 with `error` set (e.g. empty `messages` or invalid item shape).

## Example `curl` commands

Assuming `BASE=http://127.0.0.1:8000`:

```bash
curl -sS "$BASE/health"
```

```bash
curl -sS "$BASE/info"
```

```bash
curl -sS -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "schema_version": "predictions_v1",
    "requests": [
      {
        "sample_id": "example:0000",
        "messages": [
          {"role": "user", "content": "<Audio><AudioHere></Audio>\nWhat species?"}
        ],
        "audio_inputs": [
          {"payload_type": "base64_wav", "data": "UklGRg==", "sample_rate": 16000}
        ],
        "generation_config": {"max_tokens": 32, "temperature": 0.0}
      }
    ]
  }'
```

Partial failure (still HTTP 200):

```bash
curl -sS -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "schema_version": "predictions_v1",
    "requests": [
      {
        "sample_id": "bad:0001",
        "messages": [],
        "audio_inputs": [],
        "generation_config": {}
      }
    ]
  }'
```

Batch too large (expect HTTP 413): send more than `max_batch_size` items in `requests`.
