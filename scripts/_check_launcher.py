"""Stdlib-only helpers for ``check_launcher.sh`` (JSON build + validation).

This module intentionally avoids importing ``beans_next`` so the conformance
script can run on machines that only have Python 3 + curl, while still
matching the ``predictions_v1`` wire shapes in ``beans_next/api/http_schemas.py``.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import wave
from typing import Any


def _minimal_wav_b64(*, sample_rate: int = 16_000, n_frames: int = 16) -> str:
    """Build a base64-encoded mono int16 WAV payload.

    Parameters
    ----------
    sample_rate
        WAV sample rate in Hz.
    n_frames
        Number of audio frames (mono int16 samples).

    Returns
    -------
    str
        Base64-encoded WAV bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")


AUDIO_PLACEHOLDER = "<Audio><AudioHere></Audio>"


def _request_item(
    *,
    sample_id: str,
    wav_b64: str,
    sample_rate: int = 16_000,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a conformance probe for BEANS-Next.",
            },
            {
                "role": "user",
                "content": f"{AUDIO_PLACEHOLDER}\nReply with a short string.",
            },
        ],
        "audio_inputs": [
            {
                "payload_type": "base64_wav",
                "data": wav_b64,
                "sample_rate": sample_rate,
            }
        ],
        "generation_config": {"max_tokens": 32, "temperature": 0.0},
    }


def build_envelope(requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "predictions_v1", "requests": requests}


def cmd_build_body(kind: str) -> None:
    good = _minimal_wav_b64()
    if kind == "two-ok":
        env = build_envelope(
            [
                _request_item(sample_id="conformance_ok_a", wav_b64=good),
                _request_item(sample_id="conformance_ok_b", wav_b64=good),
            ]
        )
    elif kind == "partial":
        # Partial failure must be signaled via the per-item `error` field while
        # still returning HTTP 200 for the batch. To keep this launcher-agnostic
        # (no assumptions about WAV decoding), we trigger a *structural* per-item
        # failure: an empty `messages` list.
        ok = _request_item(sample_id="conformance_ok_partial", wav_b64=good)
        bad = _request_item(sample_id="conformance_bad_item", wav_b64=good)
        bad["messages"] = []
        env = build_envelope([ok, bad])
    elif kind.startswith("oversized-"):
        n = int(kind.split("-", 1)[1])
        if n < 2:
            print("oversized batch size must be >= 2", file=sys.stderr)
            sys.exit(1)
        items = [
            _request_item(sample_id=f"conformance_oversized_{i:05d}", wav_b64=good)
            for i in range(n)
        ]
        env = build_envelope(items)
    else:
        print(f"unknown build-body kind: {kind!r}", file=sys.stderr)
        sys.exit(2)
    sys.stdout.write(json.dumps(env, separators=(",", ":")))


def cmd_validate_info() -> None:
    raw = sys.stdin.read()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"FAIL: /info response is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(doc, dict):
        print("FAIL: /info top-level value must be a JSON object", file=sys.stderr)
        sys.exit(1)

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
        print(
            f"FAIL: /info missing required keys: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    for key in ("name", "model", "model_revision"):
        val = doc[key]
        if not isinstance(val, str) or not val.strip():
            print(
                f"FAIL: /info field {key!r} must be a non-empty string "
                f"(got {type(val).__name__})",
                file=sys.stderr,
            )
            sys.exit(1)

    apt = doc["audio_payload_types"]
    if not isinstance(apt, list) or not apt or not all(isinstance(x, str) for x in apt):
        print(
            "FAIL: /info audio_payload_types must be a non-empty list of strings",
            file=sys.stderr,
        )
        sys.exit(1)

    mbs = doc["max_batch_size"]
    if not isinstance(mbs, int) or isinstance(mbs, bool) or mbs < 1:
        print(
            f"FAIL: /info max_batch_size must be an integer >= 1 (got {mbs!r})",
            file=sys.stderr,
        )
        sys.exit(1)

    sb = doc["supports_batching"]
    if not isinstance(sb, bool):
        print(
            "FAIL: /info supports_batching must be a JSON boolean "
            f"(got {type(sb).__name__})",
            file=sys.stderr,
        )
        sys.exit(1)

    sv = doc["schema_versions"]
    if not isinstance(sv, list) or not all(isinstance(x, str) for x in sv):
        print(
            "FAIL: /info schema_versions must be a list of strings",
            file=sys.stderr,
        )
        sys.exit(1)
    if "predictions_v1" not in sv:
        print(
            "FAIL: /info schema_versions must include 'predictions_v1' "
            f"(got {sv!r})",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_extract_max_batch() -> None:
    try:
        doc = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"FAIL: could not parse /info JSON: {e}", file=sys.stderr)
        sys.exit(1)
    mbs = doc.get("max_batch_size")
    if not isinstance(mbs, int) or isinstance(mbs, bool) or mbs < 1:
        print(f"FAIL: max_batch_size invalid: {mbs!r}", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(str(mbs))


def _load_predict_response(raw: str) -> dict[str, Any]:
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"FAIL: /predict response is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(doc, dict):
        print("FAIL: /predict top-level value must be a JSON object", file=sys.stderr)
        sys.exit(1)
    return doc


def cmd_validate_predict_two(expected_csv: str) -> None:
    expected = {s.strip() for s in expected_csv.split(",") if s.strip()}
    raw = sys.stdin.read()
    doc = _load_predict_response(raw)
    if doc.get("schema_version") != "predictions_v1":
        print(
            f"FAIL: expected schema_version 'predictions_v1', "
            f"got {doc.get('schema_version')!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    responses = doc.get("responses")
    if not isinstance(responses, list):
        print("FAIL: responses must be a list", file=sys.stderr)
        sys.exit(1)
    if len(responses) != len(expected):
        print(
            f"FAIL: expected {len(expected)} responses, got {len(responses)}",
            file=sys.stderr,
        )
        sys.exit(1)
    seen: set[str] = set()
    for i, item in enumerate(responses):
        if not isinstance(item, dict):
            print(f"FAIL: responses[{i}] must be an object", file=sys.stderr)
            sys.exit(1)
        sid = item.get("sample_id")
        if not isinstance(sid, str) or not sid:
            print(
                f"FAIL: responses[{i}].sample_id must be a non-empty string",
                file=sys.stderr,
            )
            sys.exit(1)
        preds = item.get("predictions")
        if not isinstance(preds, list) or not all(isinstance(x, str) for x in preds):
            print(
                f"FAIL: responses[{i}].predictions must be a list of strings",
                file=sys.stderr,
            )
            sys.exit(1)
        err = item.get("error", None)
        if err is not None and not isinstance(err, str):
            print(
                f"FAIL: responses[{i}].error must be string or null, "
                f"got {type(err).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)
        if err is not None and err.strip():
            print(
                f"FAIL: responses[{sid!r}] has sample-level error set: {err!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not preds:
            print(
                f"FAIL: responses[{sid!r}].predictions must be non-empty for a "
                "successful two-item batch",
                file=sys.stderr,
            )
            sys.exit(1)
        seen.add(sid)
    if seen != expected:
        print(
            f"FAIL: response sample_id set mismatch.\n"
            f"  expected (unordered): {sorted(expected)}\n"
            f"  got (unordered):      {sorted(seen)}",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_validate_predict_partial() -> None:
    raw = sys.stdin.read()
    doc = _load_predict_response(raw)
    if doc.get("schema_version") != "predictions_v1":
        print(
            f"FAIL: expected schema_version 'predictions_v1', "
            f"got {doc.get('schema_version')!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    responses = doc.get("responses")
    if not isinstance(responses, list) or len(responses) != 2:
        print(
            "FAIL: partial-failure batch must return exactly 2 response objects",
            file=sys.stderr,
        )
        sys.exit(1)

    by_id: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(responses):
        if not isinstance(item, dict):
            print(f"FAIL: responses[{i}] must be an object", file=sys.stderr)
            sys.exit(1)
        sid = item.get("sample_id")
        if not isinstance(sid, str) or not sid:
            print(f"FAIL: responses[{i}].sample_id invalid", file=sys.stderr)
            sys.exit(1)
        by_id[sid] = item

    need = {"conformance_ok_partial", "conformance_bad_item"}
    if set(by_id) != need:
        print(
            f"FAIL: expected sample_ids {sorted(need)}, got {sorted(by_id)}",
            file=sys.stderr,
        )
        sys.exit(1)

    ok_item = by_id["conformance_ok_partial"]
    bad_item = by_id["conformance_bad_item"]

    ok_err = ok_item.get("error", None)
    if ok_err is not None and str(ok_err).strip():
        print(
            f"FAIL: ok sample must have empty error; got {ok_err!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    bad_err = bad_item.get("error", None)
    if not isinstance(bad_err, str) or not bad_err.strip():
        print(
            "FAIL: malformed/invalid-audio sample must return a non-empty "
            f"error string; got {bad_err!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    for label, item in ("ok", ok_item), ("bad", bad_item):
        preds = item.get("predictions")
        if not isinstance(preds, list) or not all(isinstance(x, str) for x in preds):
            print(
                f"FAIL: {label} item predictions must be a list of strings",
                file=sys.stderr,
            )
            sys.exit(1)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(
            "usage: _check_launcher.py "
            "{validate-info|extract-max-batch|build-body|validate-predict-two|"
            "validate-predict-partial} ...",
            file=sys.stderr,
        )
        sys.exit(2)
    cmd = argv[1]
    if cmd == "validate-info":
        cmd_validate_info()
    elif cmd == "extract-max-batch":
        cmd_extract_max_batch()
    elif cmd == "build-body":
        if len(argv) != 3:
            print("usage: _check_launcher.py build-body KIND", file=sys.stderr)
            sys.exit(2)
        cmd_build_body(argv[2])
    elif cmd == "validate-predict-two":
        if len(argv) != 3:
            print(
                "usage: _check_launcher.py validate-predict-two id1,id2 "
                "< response.json",
                file=sys.stderr,
            )
            sys.exit(2)
        cmd_validate_predict_two(argv[2])
    elif cmd == "validate-predict-partial":
        cmd_validate_predict_partial()
    else:
        print(f"unknown command: {cmd!r}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
