This fixture tree supports offline-ingestion script tests in `tests/test_ingestion_scripts.py`.

It intentionally contains **tiny** BEANS-Next artifact directories suitable for:

- `bash scripts/validate_run_dir.sh <run_dir>` (fast string checks)
- `bash scripts/ingest_and_rescore.sh <run_dir>` (calls `uv run beans-next score-from-file`)

Directories:

- `good_input_leaf/`: minimal offline-rescore **input** leaf dir
- `bad_missing_sidecar/`: missing `processed_predictions.jsonl`
- `bad_missing_predictions_field/`: `predictions.jsonl` lacks `"predictions": [...]`

