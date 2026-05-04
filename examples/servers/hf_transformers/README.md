## Tier-2 reference launcher: `hf_transformers`

This directory contains a **Tier-2 reference implementation** of the BEANS-Next HTTP contract (`predictions_v1`). It is intended as a minimal starting point for serving HuggingFace Transformers models that aren't covered by Tier‑1 launchers.

- **Maintenance expectation**: best-effort. It may drift as upstream APIs change.
- **Primary goal**: contract conformance (`POST /predict`, `GET /info`, `GET /health`) and clear documentation.

### Endpoints

- **`GET /health`**: readiness probe (HTTP 200 when ready)
- **`GET /info`**: capability document; must advertise `predictions_v1`
- **`POST /predict`**: batched inference; returns exactly one response per `sample_id`; returns **HTTP 413** if batch size exceeds `max_batch_size`

### Stub mode (CPU-only conformance)

This launcher defaults to a **stub mode** intended to pass `scripts/check_launcher.sh` without GPU or model weights:

- Enable with `HF_TRANSFORMERS_STUB=1`
- Returns deterministic placeholder predictions derived from the request content

### Run locally

From this directory (in an environment with the pinned deps installed):

```bash
PORT=19083 HF_TRANSFORMERS_STUB=1 ./serve.sh
```

### Conformance check (repo root)

```bash
uv run python scripts/with_uvicorn.py --cwd examples/servers/hf_transformers --cmd-cwd . --app serve:app --host 127.0.0.1 --port 19083 --env HF_TRANSFORMERS_STUB=1 -- uv run bash scripts/check_launcher.sh "http://127.0.0.1:19083"
```

### Real inference (not implemented in this starter)

This Tier-2 starter intentionally does **not** implement real inference (it would require choosing a specific model/processor and decoding strategy). To use it for real inference:

- Add model-loading and inference logic in `serve.py` when `HF_TRANSFORMERS_STUB` is not set.
- Add the required ML dependencies to `requirements.txt` and install them in this directory’s venv.

