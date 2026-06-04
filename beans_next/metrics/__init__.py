"""Deterministic evaluation metrics (classification, detection, captioning)."""

from __future__ import annotations

import importlib.resources
import json
import re
from collections.abc import Mapping

from beans_next.api.types import DatasetExample
from beans_next.metrics.base import (
    MetricsError,
    get_scorer,
    list_scorers,
    register_scorer,
    validate_equal_length,
)
from beans_next.metrics.captioning import cider, cider_corpus_mean_normalized, spider
from beans_next.metrics.classification import (
    accuracy,
    f1,
    precision,
    recall,
    top1_accuracy,
)
from beans_next.metrics.dataset import compute_dataset_map, compute_macro_f1
from beans_next.metrics.detection import average_precision
from beans_next.metrics.regression import (
    extract_frequency_range,
    extract_numeric_value,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from beans_next.post_process.pipeline import PostProcessResult

__all__ = [
    "MetricsError",
    "accuracy",
    "average_precision",
    "cider",
    "cider_corpus_mean_normalized",
    "compute_dataset_map",
    "compute_macro_f1",
    "extract_frequency_range",
    "extract_numeric_value",
    "f1",
    "get_scorer",
    "list_scorers",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "precision",
    "recall",
    "register_scorer",
    "root_mean_squared_error",
    "score_sample",
    "spider",
    "top1_accuracy",
    "validate_equal_length",
]


def _normalize_label_token(s: str) -> str:
    return " ".join(s.strip().split())


def _normalize_mcq_choice_token(s: str) -> str | None:
    """Return a canonical single-letter MCQ token when ``s`` is one.

    Parameters
    ----------
    s : str
        Candidate prediction or target text.

    Returns
    -------
    str or None
        Lowercase MCQ letter when ``s`` is a bare option token, otherwise
        ``None``.
    """
    match = re.fullmatch(r"\s*[\(\[]?\s*([A-Za-z])\s*[\)\]\.:]?\s*", s)
    if match is None:
        return None
    return match.group(1).lower()


def _mcq_content_match(y_pred: str, y_true: str, pred_mcq: str | None) -> bool:
    """Return True when ``y_pred`` matches the content of a full-label MCQ ground truth.

    Handles models that output the option letter, the option text, or the option
    text embedded in a sentence, without penalising case or trailing punctuation.
    Only activates when ``y_true`` contains a recognisable ``(A) …`` prefix so
    bare-letter and plain-text ground-truth tasks are unaffected.

    Parameters
    ----------
    y_pred
        Normalised prediction string (lowercased, whitespace-collapsed).
    y_true
        Normalised ground-truth string (lowercased, whitespace-collapsed).
    pred_mcq
        Lowercase single letter when ``y_pred`` is a bare MCQ token, else ``None``.

    Returns
    -------
    bool
        ``True`` if the prediction matches the MCQ option content.
    """
    true_content = _MCQ_PREFIX_RE.sub("", y_true, count=1)
    if not true_content or true_content == y_true:
        # Ground truth has no MCQ prefix — nothing to do here.
        return False
    # Fallback 1: pred is a bare letter matching the GT prefix letter.
    if pred_mcq is not None:
        letter_pat = re.compile(
            r"^\s*\(?" + re.escape(pred_mcq) + r"\)?[).\]:\s]", re.IGNORECASE
        )
        if letter_pat.match(y_true):
            return True
    # Fallback 2: strip any MCQ prefix from the prediction then check whether
    # the true content appears as a substring.  Single-token content (e.g. a
    # bare count like "3") uses word-boundary matching to avoid "3" matching
    # inside "32".
    pred_content = _MCQ_PREFIX_RE.sub("", y_pred.lower(), count=1)
    true_content_lc = true_content.lower()
    if len(true_content_lc.split()) == 1:
        return bool(
            re.search(r"\b" + re.escape(true_content_lc) + r"\b", pred_content)
        )
    return true_content_lc in pred_content


def _parse_label_list(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


_SPECIES_COUNT_RE = re.compile(r"([^,:]+?)\s*:\s*(\d+(?:\.\d+)?)")
# Matches leading MCQ letter prefix: "(A) ", "A. ", "A: ", "a) ", etc.
_MCQ_PREFIX_RE = re.compile(r"^\s*\(?[A-Za-z]\)?[).\]:\s]+")
# Per-species frequency range: "Chloris chloris: 2440-5130 Hz"
_SPECIES_FREQ_RE = re.compile(
    r"([^,;:]+?)\s*:\s*(\d[\d,.]*)\s*[-–—]\s*(\d[\d,.]*)\s*(kHz|Hz)\b",
    re.IGNORECASE,
)
# Per-species summary: "Pipilo erythrophthalmus: 2 calls, 2330-5150 Hz"
_SPECIES_SUMMARY_RE = re.compile(
    r"([^;:]+?)\s*:\s*(\d+)\s*calls?\s*,\s*(\d[\d,.]*)\s*[-–—]\s*(\d[\d,.]*)\s*(kHz|Hz)\b",
    re.IGNORECASE,
)


def _load_t3_species_names() -> dict[str, list[str]]:
    """Load ``registry/t3_species_names.json`` once and cache it.

    Returns
    -------
    dict[str, list[str]]
        Scientific name → list of accepted English common names, all
        lowercased for comparison.
    """
    path = (
        importlib.resources.files("beans_next")
        .joinpath("registry")
        .joinpath("t3_species_names.json")
    )
    try:
        raw: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        k.lower(): [v.lower() for v in vs]
        for k, vs in raw.items()
        if isinstance(k, str) and isinstance(vs, list) and not k.startswith("_")
    }


_T3_SPECIES_NAMES: dict[str, list[str]] | None = None


def _t3_species_names() -> dict[str, list[str]]:
    global _T3_SPECIES_NAMES
    if _T3_SPECIES_NAMES is None:
        _T3_SPECIES_NAMES = _load_t3_species_names()
    return _T3_SPECIES_NAMES


def _species_name_match(pred: str, true: str) -> bool:
    """Return True if ``pred`` and ``true`` refer to the same T3 species.

    Accepts both directions: scientific→common and common→scientific.
    Falls back gracefully when the lookup is unavailable.

    Parameters
    ----------
    pred
        Lowercased, whitespace-normalised prediction.
    true
        Lowercased, whitespace-normalised ground truth.

    Returns
    -------
    bool
    """
    if pred == true:
        return True
    lookup = _t3_species_names()
    # GT is a scientific name → check if pred is a known common name for it
    common_names = lookup.get(true, [])
    if pred in common_names:
        return True
    # GT is a common name → find the scientific name and check if pred is it
    # (or another common name variant)
    for sci, commons in lookup.items():
        if true == sci or true in commons:
            if pred == sci or pred in commons:
                return True
    return False


def _parse_species_count_dict(text: str) -> dict[str, float]:
    """Parse ``'Species A: 3, Species B: 2'`` into a normalised ``{name: count}`` dict.

    Returns
    -------
    dict[str, float]
        Lowercase-normalised species names mapped to their counts.
    """
    result: dict[str, float] = {}
    for match in _SPECIES_COUNT_RE.finditer(text):
        name = _normalize_label_token(match.group(1)).lower()
        if name:
            result[name] = float(match.group(2))
    return result


def _parse_species_freq_range_dict(text: str) -> dict[str, tuple[float, float]]:
    """Parse ``'Species A: 200-8000 Hz, Species B: 1000-4000 Hz'`` into
    ``{name: (low_hz, high_hz)}``.  ``Unknown`` entries are excluded.

    Returns
    -------
    dict[str, tuple[float, float]]
        Lowercase-normalised species names mapped to ``(low_hz, high_hz)``.
    """
    result: dict[str, tuple[float, float]] = {}
    for m in _SPECIES_FREQ_RE.finditer(text):
        name = _normalize_label_token(m.group(1)).lower()
        if not name or name == "unknown":
            continue
        a = float(m.group(2).replace(",", ""))
        b = float(m.group(3).replace(",", ""))
        unit = m.group(4).lower()
        if unit == "khz":
            a, b = a * 1000.0, b * 1000.0
        result[name] = (min(a, b), max(a, b))
    return result


def _parse_species_summary_dict(
    text: str,
) -> dict[str, tuple[float, float, float]]:
    """Parse ``'Species A: 2 calls, 2330-5150 Hz; Species B: 1 call, …'``
    into ``{name: (count, low_hz, high_hz)}``.  ``Unknown`` entries excluded.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Lowercase-normalised names mapped to ``(count, low_hz, high_hz)``.
    """
    result: dict[str, tuple[float, float, float]] = {}
    for m in _SPECIES_SUMMARY_RE.finditer(text):
        name = _normalize_label_token(m.group(1)).lower()
        if not name or name == "unknown":
            continue
        count = float(m.group(2))
        a = float(m.group(3).replace(",", ""))
        b = float(m.group(4).replace(",", ""))
        unit = m.group(5).lower()
        if unit == "khz":
            a, b = a * 1000.0, b * 1000.0
        result[name] = (count, min(a, b), max(a, b))
    return result


def _freq_range_metrics(
    true_dict: dict[str, tuple[float, float]],
    pred_dict: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Compute species F1 + per-matched-species frequency MAE and IoU.

    Frequency metrics are computed only over species present in both dicts
    (matched-only).  Species F1 captures identification completeness
    independently.

    Parameters
    ----------
    true_dict
        Ground-truth ``{species: (low_hz, high_hz)}``.
    pred_dict
        Predicted ``{species: (low_hz, high_hz)}``.

    Returns
    -------
    dict[str, float]
        ``species_precision``, ``species_recall``, ``species_f1``,
        and (when ≥1 species matched) ``freq_mae_low``, ``freq_mae_high``,
        ``freq_mean_iou``.
    """
    true_set = set(true_dict)
    pred_set = set(pred_dict)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1v = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out: dict[str, float] = {
        "species_precision": float(prec),
        "species_recall": float(rec),
        "species_f1": float(f1v),
    }
    matched = true_set & pred_set
    if matched:
        ae_lows, ae_highs, ious = [], [], []
        for sp in matched:
            tl, th = true_dict[sp]
            pl, ph = pred_dict[sp]
            ae_lows.append(abs(pl - tl))
            ae_highs.append(abs(ph - th))
            intersection = max(0.0, min(ph, th) - max(pl, tl))
            pw, tw = ph - pl, th - tl
            union = pw + tw - intersection
            iou = intersection / union if union > 0.0 else (1.0 if pl == tl else 0.0)
            ious.append(iou)
        n = float(len(matched))
        out["freq_mae_low"] = sum(ae_lows) / n
        out["freq_mae_high"] = sum(ae_highs) / n
        out["freq_mean_iou"] = sum(ious) / n
    return out


def score_sample(
    example: DatasetExample,
    *,
    post: PostProcessResult,
    raw_predictions: list[str],
    task_type: str | None = None,
) -> Mapping[str, float]:
    """Compute a minimal deterministic metric payload for one example.

    This is a lightweight bridge used by
    :class:`beans_next.runner.runner.BenchmarkRunner` in iteration 1. It infers
    which metric family to apply using `DatasetExample`
    content, preferring explicit metadata when available.

    Parameters
    ----------
    example
        One normalized dataset row.
    post
        A :class:`~beans_next.post_process.pipeline.PostProcessResult`.
    raw_predictions
        Raw decoded prediction strings from the launcher (n-best); the first entry
        is treated as the primary output.
    task_type : str or None, optional
        Explicit task type string (e.g. ``"classification"``, ``"detection"``).
        When provided, takes precedence over ``example.metadata["task"]``.

    Returns
    -------
    Mapping[str, float]
        Per-sample metric values. Empty when inputs do not match any supported
        pattern. For ``task_type`` captioning, returns an empty mapping because
        CIDEr is computed once over the full corpus (see ``mean.cider`` in
        ``summary.json`` from :class:`~beans_next.runner.runner.BenchmarkRunner`).
    """
    pred_text = raw_predictions[0] if raw_predictions else ""
    processed = getattr(post, "text", pred_text) or pred_text
    labels = getattr(example, "labels", None)
    meta = getattr(example, "metadata", {}) or {}
    # Explicit task_type kwarg wins over metadata.
    if task_type is not None:
        task_s = task_type.lower()
    else:
        task = meta.get("task") if isinstance(meta, dict) else None
        task_s = task.lower() if isinstance(task, str) else ""

    if isinstance(labels, str):
        if "caption" in task_s:
            return {}
        if "species_freq_range" in task_s:
            true_fr = _parse_species_freq_range_dict(labels)
            pred_fr = _parse_species_freq_range_dict(processed)
            if not true_fr and not pred_fr:
                return {"parse_success": 1.0, "species_f1": 1.0}
            if not true_fr:
                return {"parse_success": 0.0}
            return {"parse_success": 1.0, **_freq_range_metrics(true_fr, pred_fr)}
        if "species_summary" in task_s:
            true_sm = _parse_species_summary_dict(labels)
            pred_sm = _parse_species_summary_dict(processed)
            if not true_sm and not pred_sm:
                return {"parse_success": 1.0, "species_f1": 1.0}
            if not true_sm:
                return {"parse_success": 0.0}
            # Freq-range sub-dicts for shared helper
            true_fr = {sp: (v[1], v[2]) for sp, v in true_sm.items()}
            pred_fr = {sp: (v[1], v[2]) for sp, v in pred_sm.items()}
            base = _freq_range_metrics(true_fr, pred_fr)
            # Count MAE over matched species only
            matched = set(true_sm) & set(pred_sm)
            if matched:
                count_errors = [
                    abs(pred_sm[sp][0] - true_sm[sp][0]) for sp in matched
                ]
                base["count_mae"] = sum(count_errors) / float(len(matched))
            return {"parse_success": 1.0, **base}
        if "frequency_range" in task_s:
            try:
                true_low, true_high = extract_frequency_range(labels)
                pred_low, pred_high = extract_frequency_range(processed)
            except MetricsError:
                return {"parse_success": 0.0}
            ae_low = abs(pred_low - true_low)
            ae_high = abs(pred_high - true_high)
            intersection = max(0.0, min(pred_high, true_high) - max(pred_low, true_low))
            pred_width = pred_high - pred_low
            true_width = true_high - true_low
            union = pred_width + true_width - intersection
            if union > 0.0:
                iou = intersection / union
            else:
                iou = 1.0 if pred_low == true_low else 0.0
            return {
                "parse_success": 1.0,
                "absolute_error_low": float(ae_low),
                "absolute_error_high": float(ae_high),
                "iou": float(iou),
            }
        if "species_count_dict" in task_s:
            true_dict = _parse_species_count_dict(labels)
            pred_dict = _parse_species_count_dict(processed)
            if not true_dict:
                return {"parse_success": 0.0}
            true_set = set(true_dict)
            pred_set = set(pred_dict)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1v = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            all_species = true_set | pred_set
            count_errors = [
                abs(pred_dict.get(s, 0.0) - true_dict.get(s, 0.0))
                for s in all_species
            ]
            count_mae = sum(count_errors) / len(count_errors)
            return {
                "parse_success": 1.0,
                "species_precision": float(prec),
                "species_recall": float(rec),
                "species_f1": float(f1v),
                "count_mae": float(count_mae),
            }
        if "species_set" in task_s:
            true_set = {
                _normalize_label_token(p).lower() for p in _parse_label_list(labels)
            }
            pred_set = {
                _normalize_label_token(p).lower() for p in _parse_label_list(processed)
            }
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1v = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            return {"precision": float(prec), "recall": float(rec), "f1": float(f1v)}
        _ci_exact = ("species_order", "species_name", "presence_binary")
        if any(k in task_s for k in _ci_exact):
            y_pred = _normalize_label_token(processed).lower()
            y_true = _normalize_label_token(labels).lower()
            if "species_name" in task_s:
                acc = 1.0 if (y_true and _species_name_match(y_pred, y_true)) else 0.0
            else:
                acc = 1.0 if (y_pred == y_true and y_true) else 0.0
            return {"accuracy": acc, "top1_accuracy": acc}
        if "regression" in task_s:
            try:
                y_true_num = extract_numeric_value(labels)
                target_unit = (
                    "hz"
                    if "hz" in labels.lower()
                    else "db"
                    if "db" in labels.lower()
                    else None
                )
                y_pred_num = extract_numeric_value(
                    processed,
                    target_value=y_true_num,
                    unit=target_unit,
                )
            except MetricsError:
                return {"numeric_parse_success": 0.0}
            err = y_pred_num - y_true_num
            return {
                "numeric_parse_success": 1.0,
                "signed_error": float(err),
                "absolute_error": float(abs(err)),
                "squared_error": float(err * err),
            }
        y_pred = _normalize_label_token(processed)
        y_true = _normalize_label_token(labels)
        pred_mcq = _normalize_mcq_choice_token(y_pred)
        true_mcq = _normalize_mcq_choice_token(y_true)
        if pred_mcq is not None and true_mcq is not None:
            y_pred = pred_mcq
            y_true = true_mcq
        acc = 1.0 if (y_pred == y_true and y_true) else 0.0
        if not acc:
            acc = float(_mcq_content_match(y_pred, y_true, pred_mcq))
        return {
            "accuracy": acc,
            # Match BEANS-Zero evaluator semantics: for single-label classification,
            # top-1 accuracy is identical to exact-match accuracy after postprocess.
            "top1_accuracy": acc,
            "precision": acc,
            "recall": acc,
            "f1": acc,
        }

    if isinstance(labels, list):
        tgt = [_normalize_label_token(x) for x in labels if isinstance(x, str) and x]
        pred_labels = [_normalize_label_token(x) for x in _parse_label_list(processed)]

        if (
            "classification" in task_s or "open_ended" in task_s
        ) and "detection" not in task_s:
            # Multi-reference classification: correct if prediction matches any label.
            # ``open_ended`` is included so BirdSet (and similar open-set tasks that
            # ship multi-species gold) gets top-1 any-of, not multi-label AP.
            norm_pred = _normalize_label_token(processed)
            top1 = 1.0 if (processed.strip() in tgt or norm_pred in tgt) else 0.0
            if tgt:
                try:
                    top1 = float(top1_accuracy([processed], [",".join(tgt)]))
                except Exception:
                    pass
            return {"accuracy": top1, "top1_accuracy": top1}

        # Detection task (or unknown task type with list labels): multi-label AP.
        vocab = sorted(set(tgt) | set(pred_labels))
        if not vocab:
            return {}
        y_true = [[1 if v in tgt else 0 for v in vocab]]
        y_score = [[1.0 if v in pred_labels else 0.0 for v in vocab]]
        ap = float(average_precision(y_score, y_true, average="macro"))
        tp = sum(1 for v in vocab if v in tgt and v in pred_labels)
        fp = sum(1 for v in vocab if v not in tgt and v in pred_labels)
        fn = sum(1 for v in vocab if v in tgt and v not in pred_labels)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1v = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return {
            "average_precision": ap,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
        }

    return {}
