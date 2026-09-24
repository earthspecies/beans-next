"""Build compact metadata and lossless provenance locally, without uploading.

Usage: uv run python scripts/compact_beans_next_hf.py INPUT.parquet OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from beans_next.datasets.hub_schema import MAIN_COLUMNS, compact_hub_row


def main() -> None:
    """Write a compact bundle and check exact reconstruction after serialization.

    Raises
    ------
    ValueError
        If rows have duplicate identifiers or fail lossless reconstruction.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = pq.read_table(args.input)
    if not {"instruction", "output"}.issubset(original.column_names):
        raise ValueError("Input must be legacy metadata, not an already compact table")
    rows = original.to_pylist()
    compact, provenance = [], []
    for row in rows:
        main_row, source_row = compact_hub_row(row)
        compact.append(main_row)
        provenance.append(source_row)
    for key in ("id", "sample_id"):
        if len({r[key] for r in compact}) != len(rows):
            raise ValueError(f"Duplicate {key}")
    for directory in ("test", "provenance"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    main_path = args.output / "test/metadata.parquet"
    provenance_path = args.output / "provenance/metadata.parquet"
    schema = pa.schema([original.schema.field(k) for k in MAIN_COLUMNS])
    pq.write_table(
        pa.Table.from_pylist(compact, schema=schema), main_path, compression="zstd"
    )
    pq.write_table(
        pa.Table.from_pylist(provenance), provenance_path, compression="zstd"
    )
    reloaded = pq.read_table(main_path).to_pylist()
    sources = pq.read_table(provenance_path).to_pylist()
    for before, after, source in zip(rows, reloaded, sources, strict=True):
        restored = {**after, **json.loads(source["original_fields"])}
        if restored != before:
            raise ValueError(f"Lossless reconstruction failed for {before['id']}")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "columns": list(MAIN_COLUMNS),
                "lossless_reconstruction": True,
                "metadata_bytes": main_path.stat().st_size,
                "provenance_bytes": provenance_path.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
