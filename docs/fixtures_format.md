# Test fixtures

Fixtures contain small dataset rows, requests, predictions, and expected scores.
They support CPU regression tests without downloading benchmark audio or model weights.

Each bundle has a `manifest.yaml` with relative paths and model identity.
Expected artifacts use the same JSON and JSONL schemas as evaluator output.

The `scripts/fixtures/` utilities generate synthetic inputs or capture outputs from a running server.
Keep fixture model identities generic. Do not include personal paths, private endpoints, credentials, or private checkpoint locations.

Regenerate expected scores only when the benchmark behavior changes deliberately.
Run the fixture tests after regeneration.
