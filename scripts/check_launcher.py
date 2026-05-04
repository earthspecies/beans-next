"""Launcher contract conformance check for BEANS-Next (``predictions_v1``).

This script validates a running launcher against the minimal contract:

- ``GET /health`` returns HTTP 200
- ``GET /info`` returns required fields and advertises ``predictions_v1``
- ``POST /predict`` supports batching, per-item errors, and 413 on oversize

It uses stdlib HTTP (no `curl`) so it can run in restricted environments.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from scripts._check_launcher import _minimal_wav_b64, build_envelope  # type: ignore


def _die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _http_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read()
    except HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
    except (URLError, OSError, TimeoutError) as exc:
        _die(f"{method} {url}: network error {exc!r}")

    if not raw.strip():
        return status, {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"{method} {url}: invalid JSON ({exc})")
    if not isinstance(parsed, dict):
        _die(f"{method} {url}: top-level JSON must be an object")
    return status, parsed


def _check_info(doc: dict[str, Any]) -> int:
    required = (
        "name",
        "model",
        "model_revision",
        "audio_payload_types",
        "max_batch_size",
        "supports_batching",
        "schema_versions",
    )
    missing = [k for k in required if k not in doc]
    if missing:
        _die(f"/info missing required keys: {missing!r}")

    schema_versions = doc.get("schema_versions")
    if not isinstance(schema_versions, list) or not all(
        isinstance(x, str) for x in schema_versions
    ):
        _die("/info schema_versions must be a list of strings")
    if "predictions_v1" not in schema_versions:
        _die("/info must advertise 'predictions_v1' in schema_versions")

    mbs = doc.get("max_batch_size")
    if not isinstance(mbs, int) or isinstance(mbs, bool) or mbs < 1:
        _die(f"/info max_batch_size must be int >= 1, got {mbs!r}")
    return mbs


def _validate_predict_two(doc: dict[str, Any], expected: set[str]) -> None:
    if doc.get("schema_version") != "predictions_v1":
        _die(f"/predict schema_version mismatch: {doc.get('schema_version')!r}")
    responses = doc.get("responses")
    if not isinstance(responses, list):
        _die("/predict responses must be a list")
    seen: set[str] = set()
    for i, item in enumerate(responses):
        if not isinstance(item, dict):
            _die(f"/predict responses[{i}] must be an object")
        sid = item.get("sample_id")
        if not isinstance(sid, str) or not sid:
            _die(f"/predict responses[{i}].sample_id must be non-empty string")
        preds = item.get("predictions")
        if not isinstance(preds, list) or not preds or not all(
            isinstance(x, str) for x in preds
        ):
            _die(f"/predict responses[{sid!r}].predictions must be non-empty list[str]")
        err = item.get("error", None)
        if err is not None and (not isinstance(err, str) or err.strip()):
            _die(f"/predict responses[{sid!r}].error must be null/empty; got {err!r}")
        seen.add(sid)
    if seen != expected:
        _die(
            "/predict sample_id set mismatch: "
            f"expected={sorted(expected)} got={sorted(seen)}"
        )


def _validate_predict_partial(doc: dict[str, Any]) -> None:
    if doc.get("schema_version") != "predictions_v1":
        _die(f"/predict schema_version mismatch: {doc.get('schema_version')!r}")
    responses = doc.get("responses")
    if not isinstance(responses, list) or len(responses) != 2:
        _die("/predict partial test must return exactly 2 responses")
    by_id: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(responses):
        if not isinstance(item, dict):
            _die(f"/predict responses[{i}] must be an object")
        sid = item.get("sample_id")
        if not isinstance(sid, str) or not sid:
            _die(f"/predict responses[{i}].sample_id invalid")
        by_id[sid] = item
    need = {"conformance_ok_partial", "conformance_bad_item"}
    if set(by_id) != need:
        _die(
            "/predict partial sample_ids mismatch: "
            f"expected={sorted(need)} got={sorted(by_id)}"
        )

    ok_err = by_id["conformance_ok_partial"].get("error", None)
    if ok_err is not None and str(ok_err).strip():
        _die(f"ok sample must have empty error; got {ok_err!r}")
    bad_err = by_id["conformance_bad_item"].get("error", None)
    if not isinstance(bad_err, str) or not bad_err.strip():
        _die(f"bad sample must have non-empty error string; got {bad_err!r}")


def main(argv: list[str]) -> None:
    """Run the conformance checks.

    Parameters
    ----------
    argv
        CLI arguments. Expects exactly one argument: `<base_url>`.

    Raises
    ------
    SystemExit
        If any check fails or if CLI usage is invalid.
    """
    if len(argv) != 2:
        print("usage: scripts/check_launcher.py <base_url>", file=sys.stderr)
        raise SystemExit(2)
    base_url = argv[1].rstrip("/")
    health_url = urljoin(f"{base_url}/", "health")
    info_url = urljoin(f"{base_url}/", "info")
    predict_url = urljoin(f"{base_url}/", "predict")

    status, _ = _http_json("GET", health_url, body=None)
    if status != 200:
        _die(f"GET /health expected 200, got {status}")

    status, info = _http_json("GET", info_url, body=None)
    if status != 200:
        _die(f"GET /info expected 200, got {status}")
    max_batch = _check_info(info)

    # Two-ok batch.
    good = _minimal_wav_b64()
    body_two = build_envelope(
        [
            {
                "sample_id": "conformance_ok_a",
                "messages": [{"role": "user", "content": "x"}],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": good,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 1, "temperature": 0.0},
            },
            {
                "sample_id": "conformance_ok_b",
                "messages": [{"role": "user", "content": "x"}],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": good,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 1, "temperature": 0.0},
            },
        ]
    )
    status, pred = _http_json("POST", predict_url, body=body_two)
    if status != 200:
        _die(f"POST /predict(two-ok) expected 200, got {status}")
    _validate_predict_two(pred, {"conformance_ok_a", "conformance_ok_b"})

    # Partial failure batch.
    body_partial = build_envelope(
        [
            {
                "sample_id": "conformance_ok_partial",
                "messages": [{"role": "user", "content": "x"}],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": good,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 1, "temperature": 0.0},
            },
            {
                "sample_id": "conformance_bad_item",
                "messages": [],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": good,
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 1, "temperature": 0.0},
            },
        ]
    )
    status, pred_p = _http_json("POST", predict_url, body=body_partial)
    if status != 200:
        _die(f"POST /predict(partial) expected 200, got {status}")
    _validate_predict_partial(pred_p)

    # Oversized batch expects 413.
    n = max_batch + 1
    oversized = build_envelope(
        [
            {
                "sample_id": f"conformance_oversized_{i:05d}",
                "messages": [{"role": "user", "content": "x"}],
                "audio_inputs": [
                    {
                        "payload_type": "base64_wav",
                        "data": "AA==",
                        "sample_rate": 16000,
                    }
                ],
                "generation_config": {"max_tokens": 1, "temperature": 0.0},
            }
            for i in range(n)
        ]
    )
    status, _ = _http_json("POST", predict_url, body=oversized)
    if status != 413:
        _die(
            f"POST /predict(oversized) expected 413, got {status} "
            f"(max_batch_size={max_batch})"
        )

    print("OK: launcher conformance checks passed.")


if __name__ == "__main__":
    main(sys.argv)
