## BEANS-Next HTTP contract (v1)

This document specifies the **language-neutral** HTTP contract for BEANS-Next model launchers.
The contract is **schema_version `predictions_v1`** plus two mandatory introspection endpoints:
`GET /info` and `GET /health`.

Ground truth for wire shapes lives in `beans_next/api/http_schemas.py`. Conformance expectations
are enforced by `scripts/check_launcher.sh` / `scripts/check_launcher.py`.

### Endpoints (required)

- **`GET /health`**
  - **Success**: HTTP 200.
  - **Body**: JSON object (recommended) but not semantically interpreted by the client/conformance;
    the dummy launcher returns `{"status":"ok"}`.
  - **Purpose**: readiness probe (server is up and ready to accept `POST /predict`).

- **`GET /info`**
  - **Success**: HTTP 200 with a JSON object containing the required fields below.
  - **Purpose**: capability discovery and reproducibility identity.

- **`POST /predict`**
  - **Success**: HTTP 200 with a `predictions_v1` response envelope.
  - **Purpose**: batched inference.

---

## `predictions_v1` (POST `/predict`)

### Request

Content-Type: `application/json`

Top-level JSON object:

- **`schema_version`** (string, required): must be **`"predictions_v1"`**
- **`requests`** (array, required): batch of request items

Each request item is a JSON object:

- **`sample_id`** (string, required): unique identifier for the sample within the batch
- **`messages`** (array, required): chat messages (see below)
- **`audio_inputs`** (array, required): audio payload slots (see below)
- **`generation_config`** (object, required): generation parameters; launchers must tolerate
  extra keys for forward compatibility

`messages[]` item:

- **`role`** (string, required)
- **`content`** (string, required)

`audio_inputs[]` item:

- **`payload_type`** (string, required): one of:
  - `"base64_wav"`
  - `"file_path"`
  - `"file_url"`
- **`data`** (string, required): interpretation depends on `payload_type`
  - `"base64_wav"`: base64-encoded WAV bytes (no data URI prefix)
  - `"file_path"`: filesystem path string readable by the launcher
  - `"file_url"`: URL string fetchable by the launcher
- **`sample_rate`** (integer, required): nominal sample rate in Hz

#### Audio placeholder alignment

Prompts may include `<Audio><AudioHere></Audio>` placeholders (see `DESIGN.md`).
`audio_inputs[i]` aligns with the i-th `<AudioHere>` placeholder in the rendered prompt.

### Response

Content-Type: `application/json`

Top-level JSON object:

- **`schema_version`** (string, required): must be **`"predictions_v1"`**
- **`responses`** (array, required): one response item per request `sample_id`

Each response item is a JSON object:

- **`sample_id`** (string, required): must match a `sample_id` from the request batch
- **`predictions`** (array of strings, required)
  - **On success** (see `error`): must be a **non-empty** list of strings
  - **On per-item failure**: may be empty
- **`error`** (string or null, optional)
  - **Success**: `null` (preferred) or empty/whitespace string
  - **Failure**: non-empty string describing the per-item failure
- **`finish_reason`** (string, optional)
- **`usage`** (object, optional): shape is launcher-defined; may include `prompt_tokens`,
  `completion_tokens`
- **`latency_sec`** (number, optional)

The response may include additional fields only where the schema allows it (see
`beans_next/api/http_schemas.py`); launchers must not add arbitrary top-level keys.

---

## Batching + error semantics (required behavior)

### Identity and matching

- **Match requests to responses by `sample_id` only.** Do not rely on array order.
- **Exactly one response item per request item.**
  - No missing `sample_id`s.
  - No extra `sample_id`s not present in the request.
- **`sample_id` values in a request batch must be unique.**
  - Duplicate `sample_id` is a **batch-scope** contract violation (launcher may return HTTP 400).

### Partial failure (per-item `error`)

- **Partial failure must be represented per item** via `responses[i].error` (non-empty string).
- A partially-failing batch must still return **HTTP 200** with a full `responses[]` list.
- A launcher must not turn an individual bad request item into a batch HTTP error if other items
  are valid; instead, return a per-item `error` for the bad item.

### Batch-size limit and HTTP 413

- `/info.max_batch_size` declares the maximum number of request items supported in one call.
- If `len(requests) > max_batch_size`, the launcher must return **HTTP 413**.
  - The 413 response body may be JSON; its exact shape is not currently enforced, but it should
    clearly communicate the limit.

---

## `/info` (GET `/info`)

`GET /info` returns a JSON object with **all** of the following required fields:

- **`name`** (string): launcher name / implementation identifier (e.g., `beans-next-dummy`)
- **`model`** (string): model identifier (e.g., HF repo id); recorded for reproducibility
- **`model_revision`** (string): model weights/code revision identifier; recorded for reproducibility
- **`audio_payload_types`** (array of strings): supported audio payload types for `audio_inputs[].payload_type`
- **`max_batch_size`** (integer ≥ 1): maximum batch size supported by `POST /predict`
- **`supports_batching`** (boolean): whether the launcher supports `requests` with length > 1
- **`schema_versions`** (array of strings): supported schema versions; must include **`"predictions_v1"`**

Clients may use `/info` to:

- Validate that `predictions_v1` is supported before sending inference traffic.
- Chunk batches to avoid 413.
- Record `name` / `model` / `model_revision` into run artifacts for reproducibility.

---

## `/health` (GET `/health`)

`GET /health` must return **HTTP 200** when the launcher is ready to serve requests.
The response body is not contractually significant today; returning a JSON object is recommended.

