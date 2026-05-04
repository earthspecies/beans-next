"""Send one predictions_v1 request to a running launcher.

This is intentionally minimal: it does not touch the dataset stack. It just sends a
single, fixture-backed request to a `/predict` endpoint and prints the JSON response.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _load_first_request_item(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    if not line:
        raise ValueError(f"Empty requests JSONL: {path}")
    return json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict-url",
        required=True,
        help="Full predict URL (e.g. http://host:8000/predict).",
    )
    parser.add_argument(
        "--requests-jsonl",
        default=str(
            Path("tests/fixtures/beans_zero_slice_v1/inputs/requests.jsonl").as_posix()
        ),
        help="Path to a JSONL file containing PredictionsV1RequestItem objects.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    item = _load_first_request_item(Path(args.requests_jsonl))
    payload = {"schema_version": "predictions_v1", "requests": [item]}
    body = json.dumps(payload).encode("utf-8")

    req = Request(
        args.predict_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=float(args.timeout_sec)) as resp:
        resp_body = resp.read().decode("utf-8")
    print(resp_body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
