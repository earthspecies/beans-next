## Slurm model server endpoints (authoritative)

This doc is the **single source of truth** for currently-running BEANS-Next model servers
on Slurm and their `/predict` endpoints.

### How endpoints are published

Each Slurm *serve* job writes a URL file after it passes `GET /health`:

- **URL dir**: `$HOME/beans-next-launchers/`
- **URL file**: `$HOME/beans-next-launchers/<job_id>.url`
- **File contents**: one line, the full `/predict` URL

### Current endpoints (started 2026-04-29)

- **OpenAI proxy (gpt-4o-audio-preview)**:
  - **job_id**: `57486`
  - **predict_url**: `http://10.128.0.66:19085/predict`
  - **url_file**: `$HOME/beans-next-launchers/57486.url`

- **Gemini proxy (gemini-2.5-pro)**:
  - **job_id**: `57487`
  - **predict_url**: `http://10.128.0.66:19086/predict`
  - **url_file**: `$HOME/beans-next-launchers/57487.url`

- **NatureLM v1.0**:
  - **job_id**: `57512`
  - **predict_url**: `http://10.128.0.117:8000/predict`
  - **url_file**: `$HOME/beans-next-launchers/57512.url`

- **NatureLM v1.1**:
  - **job_id**: `57483`
  - **predict_url**: `http://10.128.0.117:18483/predict`
  - **url_file**: `$HOME/beans-next-launchers/57483.url`

- **AF3**:
  - **job_id**: `57484`
  - **predict_url**: `http://10.128.0.117:8002/predict`
  - **url_file**: `$HOME/beans-next-launchers/57484.url`

- **Qwen3-Omni (vLLM adapter)**:
  - **job_id**: `57519`
  - **predict_url**: `http://10.128.0.78:19082/predict`
  - **url_file**: `$HOME/beans-next-launchers/57519.url`

### Updating this doc

To refresh endpoints after restarting servers, replace the `job_id` + `predict_url`
entries above with the current values from the corresponding `.url` files.
