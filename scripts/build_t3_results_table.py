#!/usr/bin/env python3
"""Rebuild the BEANS-Next Tier 3 (scene understanding) LaTeX results table.

The table layout mirrors the hand-written ``tab:beansnext-t3-results`` (a
9-column "Counting / Temporal / Scene-open" panel plus a 10-column "Species ID
by characteristic" panel split into open-ended and MCQ halves). Unlike the
original, every cell here is recomputed from the per-sample prediction CSVs in
``beans-next-predictions-export/`` using each task's *authoritative* primary
metric taken from the ``beans_next`` eval-task registry.

Because the export ships different score columns than the original table assumed,
the reported metrics differ from the original caption:

- MCQ / open-ended classification and temporal tasks -> ``top1_accuracy`` (%).
- Counting tasks -> ``mean_absolute_error`` (lower is better, marked ``down``).
- Frequency-range description -> coverage-aware species-band IoU (%), computed on
  the fly from ``processed_prediction`` / ``target``. For each sample the IoU is
  averaged over the *ground-truth* species, with missed species scoring 0 (so a
  model cannot look good by identifying only a couple of easy species). This
  replaces the export's ``freq_mae_low``, which averages error only over matched
  species and therefore rewards abstaining.
- Ordered species summary -> ``species_f1`` (%).
- Structural captioning -> corpus CIDEr, computed on the fly from
  ``processed_prediction`` / ``target`` (there is no precomputed score column).

Run with ``uv run python scripts/build_t3_results_table.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from beans_next.metrics import _parse_species_freq_range_dict
from beans_next.metrics.captioning import cider_corpus_mean_normalized

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EXPORT = _REPO_ROOT / "beans-next-predictions-export"

# Export label -> display name, in table row order. ``nlm_v1_1_t3cap`` is also
# available in the export (MCQ/binary/captioning only) but is omitted here to
# match the original five-row table; add it to include it.
MODELS: list[tuple[str, str]] = [
    ("af3", "Audio Flamingo 3"),
    ("qwen3_omni", "Qwen3-Omni"),
    ("nlm_v1_0", "NatureLM-audio v1"),
    ("nlm_v1_1", "NatureLM-audio v1.1 base"),
    ("nlm_v1_1_t3tier3_70k", "NatureLM-audio v1.1 T3"),
]

# Metric name -> CSV column carrying that per-sample metric.
_METRIC_COLUMN = {
    "top1_accuracy": "top1_accuracy",
    "species_f1": "species_f1",
    "freq_mae_low": "freq_mae_low",
    "f1": "f1",
}
# Counting tasks declare ``mean_absolute_error`` in the registry but store it
# under task-specific column names.
_MAE_COLUMN = {
    "species_count_oe": "absolute_error",
    "vocalization_count_total_oe": "absolute_error",
    "vocalization_count_per_species_oe": "count_mae",
}


@dataclass(frozen=True)
class Column:
    """One data column of the table.

    Parameters
    ----------
    task
        Export task stem (the part after ``{model}__`` and before ``.csv``).
    header
        Literal LaTeX for the column header.
    metric
        Registry primary-metric name, or ``"cider"`` for computed captioning.
    higher_is_better
        Direction used when bolding the best value.
    scale
        Multiplier applied before formatting (100 turns a [0, 1] score into %).
    decimals
        Decimal places in the formatted cell.
    """

    task: str
    header: str
    metric: str
    higher_is_better: bool = True
    scale: float = 100.0
    decimals: int = 1

    @property
    def down_marker(self) -> str:
        return "" if self.higher_is_better else r"$\downarrow$"


@dataclass(frozen=True)
class Group:
    """A header group spanning one or more :class:`Column` objects."""

    title: str
    columns: list[Column] = field(default_factory=list)


# --- Panel 1: Counting / Temporal / Scene-open -----------------------------
PANEL1: list[Group] = [
    Group(
        "Counting",
        [
            Column(
                "species_count_oe",
                r"\texttt{\#S}",
                "mean_absolute_error",
                higher_is_better=False,
                scale=1.0,
                decimals=2,
            ),
            Column(
                "vocalization_count_total_oe",
                r"\texttt{\#V}",
                "mean_absolute_error",
                higher_is_better=False,
                scale=1.0,
                decimals=2,
            ),
            Column(
                "vocalization_count_per_species_oe",
                r"\texttt{\#V}/\texttt{S}",
                "mean_absolute_error",
                higher_is_better=False,
                scale=1.0,
                decimals=2,
            ),
        ],
    ),
    Group(
        "Temporal",
        [
            Column("vocalization_referring_mcq", "Referring", "top1_accuracy"),
            Column("vocalization_cooccurrence_binary", "Overlap", "top1_accuracy"),
        ],
    ),
    Group(
        "Scene / open",
        [
            Column(
                "frequency_range_description",
                "Freq.",
                "freq_gt_iou",
            ),
            Column("ordered_species_summary", "Sum.", "species_f1"),
            Column("structural_captioning", "Caption", "cider"),
        ],
    ),
]

# --- Panel 2: Species ID by characteristic ---------------------------------
_CHARACTERISTICS = [
    ("species_by_vocalization_order", "Order"),
    ("species_by_highest_pitch", "High"),
    ("species_by_lowest_pitch", "Low"),
    ("species_by_longest_vocalization", "Long"),
    ("species_by_vocalization_frequency", "Freq."),
]
PANEL2: list[Group] = [
    Group(
        "Open-ended",
        [
            Column(f"{stem}_oe", head, "top1_accuracy")
            for stem, head in _CHARACTERISTICS
        ],
    ),
    Group(
        "MCQ",
        [
            Column(f"{stem}_mcq", head, "top1_accuracy")
            for stem, head in _CHARACTERISTICS
        ],
    ),
]


def _band_iou(pred: tuple[float, float], true: tuple[float, float]) -> float:
    """Intersection-over-union of two ``(low_hz, high_hz)`` frequency bands.

    Parameters
    ----------
    pred
        Predicted ``(low_hz, high_hz)`` band.
    true
        Ground-truth ``(low_hz, high_hz)`` band.

    Returns
    -------
    float
        IoU in ``[0.0, 1.0]``; ``1.0`` for two identical zero-width bands.
    """
    pl, ph = pred
    tl, th = true
    intersection = max(0.0, min(ph, th) - max(pl, tl))
    union = (ph - pl) + (th - tl) - intersection
    if union > 0.0:
        return intersection / union
    return 1.0 if pl == tl else 0.0


def _freq_gt_iou_sample(pred_text: str, true_text: str) -> float | None:
    """Coverage-aware frequency IoU for one sample, over ground-truth species.

    Each ground-truth species contributes its predicted-vs-true band IoU, or 0
    when the species is absent from the prediction. Spurious predicted species
    are ignored. The sample score is the mean over ground-truth species.

    Parameters
    ----------
    pred_text
        Model output, e.g. ``"Parus major: 2500-4970 Hz"``.
    true_text
        Ground-truth string in the same format.

    Returns
    -------
    float or None
        Mean IoU over ground-truth species, or ``None`` when the target parses
        to no species (the sample cannot be scored).
    """
    true_dict = _parse_species_freq_range_dict(true_text)
    if not true_dict:
        return None
    pred_dict = _parse_species_freq_range_dict(pred_text)
    ious = [
        _band_iou(pred_dict[sp], band) if sp in pred_dict else 0.0
        for sp, band in true_dict.items()
    ]
    return sum(ious) / len(ious)


def _compute_cell(export_dir: Path, model: str, col: Column) -> float | None:
    """Aggregate ``col``'s metric for one ``(model, task)`` CSV.

    Returns
    -------
    float or None
        The scaled metric value, or ``None`` when the CSV is missing or holds
        no usable score for the metric.
    """
    path = export_dir / f"{model}__{col.task}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    if col.metric == "cider":
        pairs = df[["processed_prediction", "target"]].dropna()
        if len(pairs) < 2:
            return None
        value = cider_corpus_mean_normalized(
            [str(x) for x in pairs["processed_prediction"]],
            [str(x) for x in pairs["target"]],
        )
        return value * col.scale

    if col.metric == "freq_gt_iou":
        # Prefer the post-processed answer, but fall back to the raw output when
        # the export stored an empty processed column (e.g. the T3 freq run).
        pred = df["processed_prediction"].fillna(df.get("raw_prediction"))
        pairs = pd.DataFrame({"pred": pred, "target": df["target"]}).dropna()
        scores = [
            s
            for _, row in pairs.iterrows()
            if (s := _freq_gt_iou_sample(str(row["pred"]), str(row["target"])))
            is not None
        ]
        if not scores:
            return None
        return sum(scores) / len(scores) * col.scale

    if col.metric == "mean_absolute_error":
        column = _MAE_COLUMN[col.task]
    else:
        column = _METRIC_COLUMN[col.metric]
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean()) * col.scale


def _format(value: float | None, col: Column) -> str:
    if value is None:
        return "--"
    return f"{value:.{col.decimals}f}"


def _column_best(values: list[float | None], col: Column) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return max(present) if col.higher_is_better else min(present)


def _cell_latex(value: float | None, best: float | None, col: Column) -> str:
    text = _format(value, col)
    if value is None or best is None:
        return text
    if round(value, col.decimals) == round(best, col.decimals):
        return rf"\textbf{{{text}}}"
    return text


def _render_panel1(rows: dict[str, dict[str, float | None]], tabcolsep: int) -> str:
    columns = [c for g in PANEL1 for c in g.columns]
    best = {
        c.task: _column_best([rows[m][c.task] for m, _ in MODELS], c) for c in columns
    }
    lines: list[str] = [
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}",
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
    ]
    # Group header row.
    span_cells = [
        rf"\multicolumn{{{len(g.columns)}}}{{c}}{{{g.title}}}" for g in PANEL1
    ]
    lines.append("& " + "\n& ".join(span_cells) + r" \\")
    # cmidrules under each group.
    start = 2
    cmid: list[str] = []
    for g in PANEL1:
        end = start + len(g.columns) - 1
        cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmid))
    # Column header row.
    heads = [f"{c.header}{c.down_marker}" for c in columns]
    lines.append("Model\n& " + "\n& ".join(heads) + r" \\")
    lines.append(r"\midrule")
    for label, display in MODELS:
        cells = [
            _cell_latex(rows[label][c.task], best[c.task], c) for c in columns
        ]
        lines.append(f"{display}\n& " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _render_panel2(rows: dict[str, dict[str, float | None]], tabcolsep: int) -> str:
    columns = [c for g in PANEL2 for c in g.columns]
    best = {
        c.task: _column_best([rows[m][c.task] for m, _ in MODELS], c) for c in columns
    }
    n = len(columns)
    lines: list[str] = [
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}",
        r"\begin{tabular}{l" + "r" * n + "}",
        r"\toprule",
        rf"& \multicolumn{{{n}}}{{c}}{{Species ID by characteristic}} \\",
        rf"\cmidrule(lr){{2-{n + 1}}}",
    ]
    span_cells = [
        rf"\multicolumn{{{len(g.columns)}}}{{c}}{{{g.title}}}" for g in PANEL2
    ]
    lines.append("& " + "\n& ".join(span_cells) + r" \\")
    start = 2
    cmid: list[str] = []
    for g in PANEL2:
        end = start + len(g.columns) - 1
        cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("\n".join(cmid))
    heads = [f"{c.header}{c.down_marker}" for c in columns]
    lines.append("Model\n& " + "\n& ".join(heads) + r" \\")
    lines.append(r"\midrule")
    for label, display in MODELS:
        cells = [
            _cell_latex(rows[label][c.task], best[c.task], c) for c in columns
        ]
        lines.append(f"{display}\n& " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_table(export_dir: Path) -> str:
    """Build the full LaTeX ``table`` environment for the T3 results.

    Parameters
    ----------
    export_dir
        Directory of ``{model}__{task}.csv`` prediction exports.

    Returns
    -------
    str
        The complete LaTeX ``table`` environment.
    """
    all_columns = [c for g in (*PANEL1, *PANEL2) for c in g.columns]
    rows: dict[str, dict[str, float | None]] = {}
    for label, _ in MODELS:
        rows[label] = {
            c.task: _compute_cell(export_dir, label, c) for c in all_columns
        }

    caption = (
        "Beans-Next Tier 3 (Scene understanding) results, recomputed from the "
        "prediction export using each task's registry primary metric. "
        r"Classification and temporal tasks use top-1 accuracy (\%); counting "
        r"tasks use mean absolute error ($\downarrow$, lower is better); the "
        r"frequency-range task uses coverage-aware species-band IoU (\%, mean "
        "over ground-truth species, missed species scored 0); summary uses "
        "species F1; captioning uses corpus CIDEr. "
        r"\texttt{S}: species, \texttt{V}: vocalization, \texttt{\#}: number of."
    )

    parts: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:beansnext-t3-results}",
        "",
        r"\small",
        "",
        r"\vspace{0.4em}",
        _render_panel1(rows, tabcolsep=3),
        "",
        r"\vspace{1.0em}",
        "",
        r"\vspace{0.4em}",
        _render_panel2(rows, tabcolsep=4),
        "",
        r"\end{table}",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=_DEFAULT_EXPORT,
        help="Directory of {model}__{task}.csv prediction exports.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table (defaults to stdout).",
    )
    args = parser.parse_args()
    if not args.export_dir.is_dir():
        parser.error(f"Not a directory: {args.export_dir}")

    latex = build_table(args.export_dir)
    if args.out is not None:
        args.out.write_text(latex + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
