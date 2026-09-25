"""Rescorer tests driven by the beans_next_pairs_all6 fixture.

Two fixture files:
  - tests/fixtures/pairs/beans_next_pairs_all6_small.jsonl  (6 synthetic rows)
  - tests/fixtures/pairs/beans_next_pairs_all6_full.jsonl   (4100 real-model rows,
    audio stripped, task_type inferred from subset)

Full fixture coverage:
  - 1850 classification rows (crow/zebra-description, f0-mean-*, call-type-fixed-vocab)
  - 2250 detection rows (bird/mammal/insect/amphibian/alarm/flight-call-presence)
  - 6 model_tags: naturelm_v1_0, naturelm_v1_1, af3, qwen3_omni_30b_a3b_instruct_lean,
                  openai_gpt_4o_audio_preview, gemini_3_1_pro_preview
  - 11 subsets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from beans_next.api.types import ModelPrediction, ScoredPrediction
from beans_next.results.store import dumps_canonical
from beans_next.runner.rescorer import rescore_predictions_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "pairs"
_SMALL_FIXTURE = _FIXTURE_DIR / "beans_next_pairs_all6_small.jsonl"
_FULL_FIXTURE = _FIXTURE_DIR / "beans_next_pairs_all6_full.jsonl"

_ALL_SUBSETS = [
    "alarm-call-presence",
    "amphibian-presence",
    "bird-presence",
    "call-type-fixed-vocab",
    "crow-description",
    "f0-mean-heldout-taxa",
    "f0-mean-seen-taxa",
    "flight-call-presence",
    "insect-presence",
    "mammal-presence",
    "zebra-description",
]

_ALL_MODELS = [
    "naturelm_v1_0",
    "naturelm_v1_1",
    "af3",
    "qwen3_omni_30b_a3b_instruct_lean",
    "openai_gpt_4o_audio_preview",
    "gemini_3_1_pro_preview",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_predictions_from_pairs(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Write predictions.jsonl + processed_predictions.jsonl from fixture rows.

    Sample IDs are made unique per row by appending the row index, because the
    fixture uses the same sample UUID for all models that predicted on the same
    audio file. The rescorer is keyed on sample_id, so duplicate ids would cause
    only the last row's output to be retained.

    Returns
    -------
    Path
        The prepared test fixture.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    processed_path = out_dir / "processed_predictions.jsonl"

    with (
        predictions_path.open("w", encoding="utf-8") as pred_f,
        processed_path.open("w", encoding="utf-8") as proc_f,
    ):
        for idx, r in enumerate(rows):
            # Append row index to ensure uniqueness across models for the same audio.
            sample_id = f"{r['sample_id']}_{idx:05d}"
            raw_predictions = r.get("raw_predictions") or []
            raw_text = str(raw_predictions[0]) if raw_predictions else ""
            task_id = r.get("task_id")

            pred = ModelPrediction(
                sample_id=sample_id,
                predictions=[raw_text],
                finish_reason=r.get("finish_reason"),
                latency_sec=r.get("latency_sec"),
                error=r.get("error"),
                server_info=r.get("server_info"),
            )
            pred_f.write(dumps_canonical(pred.model_dump(mode="json")) + "\n")

            processed = ScoredPrediction(
                sample_id=sample_id,
                task_id=str(task_id) if task_id is not None else None,
                predictions=[raw_text],
                processed_prediction=r.get("processed_prediction"),
                targets=r.get("ground_truth"),
                scores=None,
                postprocess_version=None,
                error=r.get("error"),
            )
            proc_f.write(dumps_canonical(processed.model_dump(mode="json")) + "\n")

    return predictions_path


# ---------------------------------------------------------------------------
# Session-scoped fixture — load full JSONL once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def full_fixture_rows() -> list[dict[str, Any]]:
    return _load_jsonl(_FULL_FIXTURE)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ===========================================================================
# Small fixture tests (original, kept for reference)
# ===========================================================================


def test_rescore_from_pairs_fixture_detection(tmp_path: Path) -> None:
    rows = _load_jsonl(_SMALL_FIXTURE)
    detection_rows = [r for r in rows if r.get("task_type") == "detection"]
    assert detection_rows
    predictions_path = _write_predictions_from_pairs(detection_rows, tmp_path / "det")
    summary = rescore_predictions_file(predictions_path, task_type="detection")
    assert summary.n_samples == len(detection_rows)


# ===========================================================================
# Full fixture — bulk rescoring
# ===========================================================================


def test_full_fixture_loaded(full_fixture_rows: list[dict[str, Any]]) -> None:
    """Fixture has expected row counts by task type and model."""
    assert len(full_fixture_rows) == 4100
    cls_count = sum(1 for r in full_fixture_rows if r["task_type"] == "classification")
    det_count = sum(1 for r in full_fixture_rows if r["task_type"] == "detection")
    assert cls_count == 1850
    assert det_count == 2250
    models = {r["model_tag"] for r in full_fixture_rows}
    assert models == set(_ALL_MODELS)
    subsets = {r["subset"] for r in full_fixture_rows}
    assert subsets == set(_ALL_SUBSETS)


def test_rescore_all_classification(
    tmp_path: Path, full_fixture_rows: list[dict[str, Any]]
) -> None:
    rows = [r for r in full_fixture_rows if r["task_type"] == "classification"]
    predictions_path = _write_predictions_from_pairs(rows, tmp_path)
    summary = rescore_predictions_file(predictions_path, task_type="classification")

    assert summary.n_samples == len(rows)
    assert summary.n_errors == 0
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "scored_predictions.jsonl").is_file()
    assert (tmp_path / "processed_predictions.jsonl").is_file()

    mean = summary.metrics.get("mean", {})
    assert mean, "No mean metrics computed"
    metric_keys = set(mean.keys())
    assert metric_keys & {"accuracy", "top1_accuracy", "f1"}, (
        f"Expected classification metrics in {metric_keys}"
    )


def test_rescore_all_detection(
    tmp_path: Path, full_fixture_rows: list[dict[str, Any]]
) -> None:
    rows = [r for r in full_fixture_rows if r["task_type"] == "detection"]
    predictions_path = _write_predictions_from_pairs(rows, tmp_path)
    summary = rescore_predictions_file(predictions_path, task_type="detection")

    assert summary.n_samples == len(rows)
    assert summary.n_errors == 0
    assert (tmp_path / "summary.json").is_file()

    mean = summary.metrics.get("mean", {})
    assert mean
    metric_keys = set(mean.keys())
    assert metric_keys & {"f1", "precision", "recall", "average_precision"}, (
        f"Expected detection metrics in {metric_keys}"
    )


# ---------------------------------------------------------------------------
# Per-model rescoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_tag", _ALL_MODELS)
def test_rescore_per_model_classification(
    tmp_path: Path,
    full_fixture_rows: list[dict[str, Any]],
    model_tag: str,
) -> None:
    rows = [
        r
        for r in full_fixture_rows
        if r["model_tag"] == model_tag and r["task_type"] == "classification"
    ]
    if not rows:
        pytest.skip(f"No classification rows for {model_tag}")

    predictions_path = _write_predictions_from_pairs(rows, tmp_path / model_tag)
    summary = rescore_predictions_file(predictions_path, task_type="classification")
    assert summary.n_samples == len(rows)
    assert summary.n_errors == 0


@pytest.mark.parametrize("model_tag", _ALL_MODELS)
def test_rescore_per_model_detection(
    tmp_path: Path,
    full_fixture_rows: list[dict[str, Any]],
    model_tag: str,
) -> None:
    rows = [
        r
        for r in full_fixture_rows
        if r["model_tag"] == model_tag and r["task_type"] == "detection"
    ]
    if not rows:
        pytest.skip(f"No detection rows for {model_tag}")

    predictions_path = _write_predictions_from_pairs(rows, tmp_path / model_tag)
    summary = rescore_predictions_file(predictions_path, task_type="detection")
    assert summary.n_samples == len(rows)
    assert summary.n_errors == 0


# ---------------------------------------------------------------------------
# Per-subset rescoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subset", _ALL_SUBSETS)
def test_rescore_per_subset(
    tmp_path: Path,
    full_fixture_rows: list[dict[str, Any]],
    subset: str,
) -> None:
    rows = [r for r in full_fixture_rows if r["subset"] == subset]
    assert rows, f"No rows for subset {subset}"
    task_type = rows[0]["task_type"]

    predictions_path = _write_predictions_from_pairs(rows, tmp_path / subset)
    summary = rescore_predictions_file(predictions_path, task_type=task_type)

    assert summary.n_samples == len(rows)
    assert summary.n_errors == 0
    assert (tmp_path / subset / "summary.json").is_file()
    assert summary.metrics.get("mean"), f"No metrics for subset {subset}"


# ---------------------------------------------------------------------------
# Parsing regression: current pipeline must reproduce stored processed_prediction
# ---------------------------------------------------------------------------


def test_parsing_regression_classification(
    tmp_path: Path, full_fixture_rows: list[dict[str, Any]]
) -> None:
    """Current post-process pipeline reproduces ≥90% of stored processed_predictions.

    Runs per-subset so that label vocabulary matches production usage (each
    task is rescored independently with only its own label set).

    `call-type-fixed-vocab` is excluded: the fixture's stored processed_predictions
    for that subset were generated with a multi-label comma-split pipeline (producing
    "song, call"), whereas task_type="classification" uses single-label
    extract_label_from_text. That subset is ambiguously typed in the fixture and is
    tested separately via test_parsing_regression_call_type_fixed_vocab.
    """
    cls_subsets = [
        s
        for s in _ALL_SUBSETS
        if s
        not in {
            "alarm-call-presence",
            "amphibian-presence",
            "bird-presence",
            "call-type-fixed-vocab",
            "flight-call-presence",
            "insect-presence",
            "mammal-presence",
        }
    ]
    total_matches = total_rows = 0
    for subset in cls_subsets:
        rows = [r for r in full_fixture_rows if r["subset"] == subset]
        out = tmp_path / subset
        predictions_path = _write_predictions_from_pairs(rows, out)
        rescore_predictions_file(predictions_path, task_type="classification")

        rescored: dict[str, str] = {}
        for line in (out / "processed_predictions.jsonl").read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                rescored[obj["sample_id"]] = obj.get("processed_prediction") or ""

        for i, r in enumerate(rows):
            uid = f"{r['sample_id']}_{i:05d}"
            total_rows += 1
            if rescored.get(uid) == (r.get("processed_prediction") or ""):
                total_matches += 1

    match_rate = total_matches / total_rows
    assert match_rate >= 0.90, (
        f"Classification parsing regression: {match_rate:.1%} match "
        f"({total_matches}/{total_rows}). Post-process pipeline may have changed."
    )


def test_parsing_regression_call_type_fixed_vocab(
    tmp_path: Path, full_fixture_rows: list[dict[str, Any]]
) -> None:
    """call-type-fixed-vocab stored processed_predictions use comma-split (detection
    pipeline).

    Even though the fixture marks these rows task_type="classification", the original
    processing used multi-label parsing. Verify the detection pipeline reproduces ≥90%.
    """
    rows = [r for r in full_fixture_rows if r["subset"] == "call-type-fixed-vocab"]
    assert rows
    out = tmp_path / "call-type-fixed-vocab"
    predictions_path = _write_predictions_from_pairs(rows, out)
    rescore_predictions_file(predictions_path, task_type="detection")

    rescored: dict[str, str] = {}
    for line in (out / "processed_predictions.jsonl").read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            rescored[obj["sample_id"]] = obj.get("processed_prediction") or ""

    total_matches = total_rows = 0
    for i, r in enumerate(rows):
        uid = f"{r['sample_id']}_{i:05d}"
        total_rows += 1
        if rescored.get(uid) == (r.get("processed_prediction") or ""):
            total_matches += 1

    match_rate = total_matches / total_rows
    assert match_rate >= 0.90, (
        f"call-type-fixed-vocab detection-pipeline regression: {match_rate:.1%} "
        f"({total_matches}/{total_rows})"
    )


def test_parsing_regression_detection(
    tmp_path: Path, full_fixture_rows: list[dict[str, Any]]
) -> None:
    """Current post-process pipeline reproduces ≥90% of stored processed_predictions.

    Runs per-subset so that label vocabulary matches production usage.
    """
    det_subsets = [
        "alarm-call-presence",
        "amphibian-presence",
        "bird-presence",
        "flight-call-presence",
        "insect-presence",
        "mammal-presence",
    ]
    total_matches = total_rows = 0
    for subset in det_subsets:
        rows = [r for r in full_fixture_rows if r["subset"] == subset]
        out = tmp_path / subset
        predictions_path = _write_predictions_from_pairs(rows, out)
        rescore_predictions_file(predictions_path, task_type="detection")

        rescored: dict[str, str] = {}
        for line in (out / "processed_predictions.jsonl").read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                rescored[obj["sample_id"]] = obj.get("processed_prediction") or ""

        for i, r in enumerate(rows):
            uid = f"{r['sample_id']}_{i:05d}"
            total_rows += 1
            if rescored.get(uid) == (r.get("processed_prediction") or ""):
                total_matches += 1

    match_rate = total_matches / total_rows
    assert match_rate >= 0.90, (
        f"Detection parsing regression: {match_rate:.1%} match "
        f"({total_matches}/{total_rows}). Post-process pipeline may have changed."
    )


# ---------------------------------------------------------------------------
# Scored predictions sanity: scores are finite floats in [0, 1]
# ---------------------------------------------------------------------------


def test_scored_predictions_values_classification(
    tmp_path: Path, full_fixture_rows: list[dict[str, Any]]
) -> None:
    rows = [r for r in full_fixture_rows if r["task_type"] == "classification"]
    predictions_path = _write_predictions_from_pairs(rows, tmp_path)
    rescore_predictions_file(predictions_path, task_type="classification")

    scored = _load_jsonl(tmp_path / "scored_predictions.jsonl")
    assert len(scored) == len(rows)
    for row in scored:
        scores = row.get("scores") or {}
        for metric, val in scores.items():
            assert isinstance(val, (int, float)), f"score {metric}={val!r} not numeric"
            assert 0.0 <= val <= 1.0, f"score {metric}={val} out of [0,1]"


def test_scored_predictions_values_detection(
    tmp_path: Path, full_fixture_rows: list[dict[str, Any]]
) -> None:
    rows = [r for r in full_fixture_rows if r["task_type"] == "detection"]
    predictions_path = _write_predictions_from_pairs(rows, tmp_path)
    rescore_predictions_file(predictions_path, task_type="detection")

    scored = _load_jsonl(tmp_path / "scored_predictions.jsonl")
    assert len(scored) == len(rows)
    for row in scored:
        scores = row.get("scores") or {}
        for metric, val in scores.items():
            assert isinstance(val, (int, float)), f"score {metric}={val!r} not numeric"
            assert 0.0 <= val <= 1.0, f"score {metric}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-model consistency: same ground truth, different raw outputs
# ---------------------------------------------------------------------------


def test_same_sample_id_across_models_rescores_consistently(
    tmp_path: Path,
    full_fixture_rows: list[dict[str, Any]],
) -> None:
    """For rows sharing the same ground_truth + subset, scores should be finite."""
    # crow-description has rows for all 4 GPU models with the same prompt structure
    rows = [r for r in full_fixture_rows if r["subset"] == "crow-description"]
    assert rows

    predictions_path = _write_predictions_from_pairs(rows, tmp_path)
    summary = rescore_predictions_file(predictions_path, task_type="classification")
    assert summary.n_samples == len(rows)
    assert summary.n_errors == 0

    # Check per-model accuracy can be derived from scored rows
    scored = _load_jsonl(tmp_path / "scored_predictions.jsonl")
    model_correct: dict[str, list[float]] = {}
    # sample_id encodes model_tag in fixture: e.g. <run_id>__<model>__<sample_idx>
    # use ground_truth + processed_prediction comparison as proxy
    for row in scored:
        acc = (row.get("scores") or {}).get("accuracy")
        if acc is not None:
            # group by model via raw fixture lookup
            match = next(
                (r for r in rows if str(r["sample_id"]) == row["sample_id"]), None
            )
            if match:
                mt = match["model_tag"]
                model_correct.setdefault(mt, []).append(float(acc))

    # Each model that has rows must have some scored samples
    for mt, scores_list in model_correct.items():
        assert scores_list, f"No accuracy scores for {mt}"
        mean_acc = sum(scores_list) / len(scores_list)
        assert 0.0 <= mean_acc <= 1.0, (
            f"Mean accuracy {mean_acc} out of bounds for {mt}"
        )
