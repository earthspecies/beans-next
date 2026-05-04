"""Deterministic JSON / JSONL artifact writers for benchmark runs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from beans_next.api.types import RunSummary, ScoredPrediction
from beans_next.runner.checkpoint import completed_sample_ids_from_checkpoint_json

__all__ = ["BenchmarkArtifactWriter"]


def dumps_canonical(obj: object) -> str:
    """Serialize ``obj`` to a canonical JSON string for artifacts.

    Uses sorted object keys and compact separators so repeated runs with the
    same logical content produce identical bytes.

    Parameters
    ----------
    obj
        JSON-serializable object (typically a ``dict`` from Pydantic ``model_dump``).

    Returns
    -------
    str
        UTF-8 JSON text without a trailing newline.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class BenchmarkArtifactWriter:
    """Write required benchmark artifacts under one output directory.

    Creates these files:

    * ``predictions.jsonl``
    * ``processed_predictions.jsonl``
    * ``scored_predictions.jsonl``
    * ``summary.json`` (written via :meth:`write_summary`)
    * ``model_identity.json`` (written via :meth:`write_model_identity`)
    * ``checkpoint.json`` (written via :meth:`write_checkpoint`)

    Resume behavior
    ---------------
    If ``checkpoint.json`` already exists in ``output_dir``, the writer enters a
    **resume** mode:

    - JSONL artifacts are opened in append mode and never truncated.
    - Appends are skipped for rows whose ``sample_id`` already appears in the
      checkpoint's ``completed_sample_ids``.
    - :meth:`write_checkpoint` merges previously completed ids into the new payload
      before atomically rewriting ``checkpoint.json``.

    Parameters
    ----------
    output_dir
        Directory that will be created if missing.

    Raises
    ------
    OSError
        If the directory cannot be created or files cannot be opened.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pred_path = self._output_dir / "predictions.jsonl"
        self._proc_path = self._output_dir / "processed_predictions.jsonl"
        self._scored_path = self._output_dir / "scored_predictions.jsonl"
        self._checkpoint_path = self._output_dir / "checkpoint.json"
        self._resume_completed_ids: set[str] = set()
        self._resume_enabled = False
        if self._checkpoint_path.is_file():
            try:
                self._resume_completed_ids = completed_sample_ids_from_checkpoint_json(
                    self._checkpoint_path
                )
                self._resume_enabled = True
            except Exception:
                # If checkpoint parsing fails, fall back to overwrite semantics.
                self._resume_completed_ids = set()
                self._resume_enabled = False

        if not self._resume_enabled:
            for p in (self._pred_path, self._proc_path, self._scored_path):
                p.write_text("", encoding="utf-8")
        else:
            for p in (self._pred_path, self._proc_path, self._scored_path):
                p.touch(exist_ok=True)

        self._pred_f = self._pred_path.open("a", encoding="utf-8")
        self._proc_f = self._proc_path.open("a", encoding="utf-8")
        self._scored_f = self._scored_path.open("a", encoding="utf-8")

    @property
    def output_dir(self) -> Path:
        """Output directory path."""
        return self._output_dir

    def append_prediction_record(self, record: BaseModel) -> None:
        """Append one JSON line for a raw ``ModelPrediction``-shaped record."""
        sid = getattr(record, "sample_id", None)
        if (
            self._resume_enabled
            and isinstance(sid, str)
            and sid in self._resume_completed_ids
        ):
            return
        line = dumps_canonical(record.model_dump(mode="json"))
        self._pred_f.write(line + "\n")
        self._pred_f.flush()

    def append_processed_prediction(self, row: ScoredPrediction) -> None:
        """Append one processed row (typically ``scores`` unset or null)."""
        if self._resume_enabled and row.sample_id in self._resume_completed_ids:
            return
        data = row.model_dump(mode="json")
        line = dumps_canonical(data)
        self._proc_f.write(line + "\n")
        self._proc_f.flush()

    def append_scored_prediction(self, row: ScoredPrediction) -> None:
        """Append one scored row including metric floats."""
        if self._resume_enabled and row.sample_id in self._resume_completed_ids:
            return
        line = dumps_canonical(row.model_dump(mode="json"))
        self._scored_f.write(line + "\n")
        self._scored_f.flush()

    def write_checkpoint(self, payload: Mapping[str, object]) -> None:
        """Atomically rewrite ``checkpoint.json`` with a canonical JSON object.

        Parameters
        ----------
        payload
            JSON-serializable mapping (typically includes ``run_id`` and progress
            fields).
        """
        path = self._checkpoint_path
        out = dict(payload)
        if self._resume_enabled:
            raw_completed = out.get("completed_sample_ids")
            new_completed: set[str] = set()
            if isinstance(raw_completed, list):
                for item in raw_completed:
                    if isinstance(item, str) and item.strip():
                        new_completed.add(item)
            merged = set(self._resume_completed_ids)
            merged.update(new_completed)
            out["completed_sample_ids"] = sorted(merged)
            out["n_predictions_written"] = len(merged)
            self._resume_completed_ids = merged
            raw_n_errors = out.get("n_errors")
            try:
                prev = json.loads(path.read_text(encoding="utf-8"))
                prev_n_errors = prev.get("n_errors") if isinstance(prev, dict) else None
            except Exception:
                prev_n_errors = None
            if isinstance(raw_n_errors, int) and isinstance(prev_n_errors, int):
                out["n_errors"] = max(prev_n_errors, raw_n_errors)

        body = dumps_canonical(out)
        tmp_dir = self._output_dir
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=tmp_dir,
            prefix="checkpoint.json.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            f.write(body + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)

    def write_summary(self, summary: RunSummary) -> None:
        """Write ``summary.json`` from a :class:`~beans_next.api.types.RunSummary`."""
        path = self._output_dir / "summary.json"
        path.write_text(
            dumps_canonical(summary.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )

    def write_model_identity(self, identity: Mapping[str, object]) -> None:
        """Write ``model_identity.json`` with a snapshot of launcher ``/info`` fields.

        Parameters
        ----------
        identity
            JSON-serializable mapping (typically the probed ``GET /info`` payload).
        """
        path = self._output_dir / "model_identity.json"
        path.write_text(dumps_canonical(dict(identity)) + "\n", encoding="utf-8")

    def write_judge_outputs(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Write ``judge_outputs.jsonl`` with one judge response row per line.

        Parameters
        ----------
        rows
            Serialized judge response items (each a plain mapping from
            ``JudgeScoresV1ResponseItem.model_dump(mode="json")``).
        """
        path = self._output_dir / "judge_outputs.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(dumps_canonical(dict(row)) + "\n")

    def close(self) -> None:
        """Flush and close open JSONL handles."""
        self._pred_f.close()
        self._proc_f.close()
        self._scored_f.close()

    def __enter__(self) -> BenchmarkArtifactWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
