"""Deterministic evaluation metrics (classification, detection, captioning)."""

from __future__ import annotations

import importlib.resources
import json
import re
from collections.abc import Mapping
from functools import lru_cache

from beans_next.api.types import DatasetExample
from beans_next.metrics.base import (
    MetricsError,
    get_scorer,
    list_scorers,
    register_scorer,
    validate_equal_length,
)
from beans_next.metrics.captioning import cider, cider_corpus_mean_normalized
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
from beans_next.metrics.species_counts import parse_species_counts
from beans_next.post_process.answers import (
    FREE_TEXT_TASKS,
    INVALID_ANSWER,
    clean_answer,
    explicit_empty,
    parse_call_types,
    parse_choice_set,
    question_options,
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
    """Compare literal full-label answers without target-substring searching.

    Returns
    -------
    bool
        Whether the full label matches exactly.
    """
    # Legacy full-label artifacts without the original question support only
    # literal letter/content equivalence. Never search the answer for GT text.
    match = re.match(r"^\s*\(?([a-h])\)?[).:\s]+(.+)$", y_true, re.I)
    if not match:
        return False
    from beans_next.post_process.answers import _plain, _unwrap

    candidate = _unwrap(y_pred)
    prefix = re.match(r"^\s*\(?([a-h])\)?[).:\s]+(.+)$", candidate, re.I)
    if prefix:
        return prefix.group(1).lower() == match.group(1).lower() and _plain(
            prefix.group(2)
        ) == _plain(match.group(2))
    if pred_mcq is not None:
        return pred_mcq == match.group(1).lower()
    return _plain(candidate) == _plain(match.group(2))


def _parse_label_list(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


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


@lru_cache(maxsize=1)
def _species_alias_pattern() -> tuple[re.Pattern[str], dict[str, str]]:
    aliases: dict[str, set[str]] = {}
    for scientific, commons in _t3_species_names().items():
        for alias in [scientific, *commons]:
            aliases.setdefault(alias, set()).add(scientific)
    unique = {a: next(iter(names)) for a, names in aliases.items() if len(names) == 1}
    pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(a) for a in sorted(unique, key=len, reverse=True))
        + r")(?!\w)"
    )
    return re.compile(pattern, re.I), unique


def parse_summary_species(text: str) -> set[str] | None:
    """Extract named species independently of count/band format and target.

    None means unparseable/unknown-only. An empty set requires explicit absence.
    Unknown annotations are excluded from named-species evaluation, not credited
    as an empty correct answer. Unrecognized scientific binomials remain false
    positives rather than disappearing from the prediction.

    Returns
    -------
    set[str] | None
        Named species, an explicit empty set, or None for an invalid answer.
    """
    text = clean_answer(text)
    if explicit_empty(text):
        return set()
    if re.search(r"\b(?:not|no|neither)\b", text, re.I):
        return None
    pattern, aliases = _species_alias_pattern()
    found = {aliases[m.group().lower()] for m in pattern.finditer(text)}
    # Scientific binomials not in the benchmark lexicon still count as claims.
    excluded = {
        "The",
        "This",
        "There",
        "Based",
        "Here",
        "Unknown",
        "Frequency",
        "Call",
        "Species",
        "No",
        "It",
        "In",
        "Each",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Bird",
        "Birds",
        "Audio",
        "From",
        "We",
        "I",
        "Number",
        "First",
        "Second",
        "Third",
        "A",
        "An",
        "For",
        "As",
        "At",
        "These",
        "They",
        "You",
        "Only",
        "Overall",
        "Several",
        "Multiple",
        "Some",
        "All",
    }
    for m in re.finditer(r"\b([A-Z][a-z]+) ([a-z][a-z-]+)\b", text):
        if m.group(1) not in excluded:
            name = m.group().lower()
            # A common-name alias already recognized in this span is not a
            # second invented scientific species.
            if name not in aliases:
                found.add(name)
    return found or None


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

    # Scoring must never inherit target-vocabulary snapping from old artifacts.
    if task_s in FREE_TEXT_TASKS and raw_predictions:
        processed = clean_answer(pred_text)

    if task_s in {
        "multilabel_accuracy",
        "multilabel_classification",
        "multilabel_detection",
    } and isinstance(labels, str):
        options = question_options(meta)
        if task_s in {"multilabel_accuracy", "multilabel_detection"}:
            true_set = parse_choice_set(labels, options, multiple=True)
            pred_set = parse_choice_set(pred_text, options, multiple=True)
        else:
            true_set = parse_call_types(labels)
            pred_set = parse_call_types(pred_text)
        if true_set is None:
            return {"target_parse_success": 0.0}
        valid = pred_set is not None and INVALID_ANSWER not in pred_set
        tp = len(true_set & (pred_set or set()))
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        f1v = (
            2 * tp / (len(true_set) + len(pred_set or set()))
            if true_set or pred_set
            else float(valid)
        )
        return {
            "target_parse_success": 1.0,
            "parse_success": float(valid),
            "accuracy": float(valid and pred_set == true_set),
            "top1_accuracy": float(valid and pred_set == true_set),
            "precision": precision,
            "recall": recall,
            "f1": f1v,
        }

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
            true_species = parse_summary_species(labels)
            pred_species = parse_summary_species(processed)
            if true_species is None:
                return {"target_parse_success": 0.0}
            valid = pred_species is not None
            tp = len(true_species & (pred_species or set()))
            denominator = len(true_species) + len(pred_species or set())
            species_scores = {
                "target_parse_success": 1.0,
                "parse_success": float(valid),
                "species_precision": tp / len(pred_species) if pred_species else 0.0,
                "species_recall": tp / len(true_species) if true_species else 0.0,
                "species_f1": 2 * tp / denominator if denominator else float(valid),
            }
            true_sm = _parse_species_summary_dict(labels)
            pred_sm = _parse_species_summary_dict(processed)
            if not true_sm:
                return species_scores
            # Freq-range sub-dicts for shared helper
            true_fr = {sp: (v[1], v[2]) for sp, v in true_sm.items()}
            pred_fr = {sp: (v[1], v[2]) for sp, v in pred_sm.items()}
            base = _freq_range_metrics(true_fr, pred_fr)
            # Count MAE over matched species only
            matched = set(true_sm) & set(pred_sm)
            if matched:
                count_errors = [abs(pred_sm[sp][0] - true_sm[sp][0]) for sp in matched]
                base["count_mae"] = sum(count_errors) / float(len(matched))
            return {**base, **species_scores}
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
            true_dict = parse_species_counts(labels)
            pred_dict = parse_species_counts(processed)
            if true_dict is None:
                return {"target_parse_success": 0.0}
            if pred_dict is None:
                return {
                    "target_parse_success": 1.0,
                    "parse_success": 0.0,
                    "species_precision": 0.0,
                    "species_recall": 0.0,
                    "species_f1": 0.0,
                }
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
                abs(pred_dict.get(s, 0.0) - true_dict.get(s, 0.0)) for s in all_species
            ]
            count_mae = sum(count_errors) / len(count_errors) if count_errors else 0.0
            return {
                "parse_success": 1.0,
                "target_parse_success": 1.0,
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
        options = question_options(meta)
        # Letter answers are interpreted from raw text with this row's options.
        # A target letter selects comparison semantics, never the interpretation.
        if options or _normalize_mcq_choice_token(labels) is not None:
            chosen = parse_choice_set(pred_text, options)
            true_choice = parse_choice_set(labels, options)
            acc = float(
                chosen is not None and true_choice is not None and chosen == true_choice
            )
            return {
                "accuracy": acc,
                "top1_accuracy": acc,
                "precision": acc,
                "recall": acc,
                "f1": acc,
                "parse_success": float(chosen is not None),
            }
        y_pred = _normalize_label_token(processed)
        if _MCQ_PREFIX_RE.match(labels):
            y_pred = _normalize_label_token(clean_answer(pred_text))
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
