Small example result directories for offline rescoring.
The files use the evaluator's prediction, processed-prediction, and summary formats.

Directories:

- `good_input_leaf/`: minimal offline-rescore **input** leaf dir
- `bad_missing_sidecar/`: missing `processed_predictions.jsonl`
- `bad_missing_predictions_field/`: `predictions.jsonl` lacks `"predictions": [...]`
