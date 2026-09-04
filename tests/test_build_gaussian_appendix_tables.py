"""Tests for the Gaussian appendix-table generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def table_module() -> ModuleType:
    """Load the table script as a module.

    Returns
    -------
    ModuleType
        Loaded table-generator module.
    """

    path = Path(__file__).parents[1] / "scripts" / "build_gaussian_appendix_tables.py"
    spec = importlib.util.spec_from_file_location(
        "build_gaussian_appendix_tables", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(metric: str = "accuracy") -> dict[str, object]:
    diagnostic = {
        "non_error_denominator": 10,
        "numeric_parse_denominator": 0,
        "numeric_parse_failure_count": 0,
        "numeric_parse_failure_rate": None,
        "refusal_heuristic_count": 1,
    }
    return {
        "n_samples": 10,
        "metrics": {
            metric: {"real": 0.75, "noise": 0.25, "audio_minus_noise": 0.5}
        },
        "real_diagnostics": diagnostic,
        "noise_diagnostics": diagnostic,
    }


def test_task_tier_uses_explicit_tier4_and_prefixes(table_module: ModuleType) -> None:
    """Tier classification keeps few-shot tasks out of Tier 2."""

    assert table_module.task_tier("beans_next_t1_caption") == 1
    assert table_module.task_tier("beans_next_bird_presence") == 2
    assert table_module.task_tier("beans_next_t3_species_count_oe") == 3
    assert table_module.task_tier("beans_next_gibbon_fewshot_detection_balanced") == 4


def test_build_tables_contains_tier_task_and_parse_tables(
    table_module: ModuleType,
) -> None:
    """The generator emits all three appendix table types."""

    numeric = _row("absolute_error")
    for arm in ("real_diagnostics", "noise_diagnostics"):
        numeric[arm] = {
            **numeric[arm],
            "numeric_parse_denominator": 10,
            "numeric_parse_failure_count": 2,
            "numeric_parse_failure_rate": 0.2,
        }
    tasks = {
        "beans_next_t1_description_mcq": _row(),
        "beans_next_bird_presence": _row(),
        "beans_next_t3_species_by_highest_pitch_mcq": _row(),
        "beans_next_gibbon_fewshot_detection_balanced": _row(),
        "beans_next_t1_snr_regression": numeric,
    }
    latex = table_module.build_tables({"naturelm": {"tasks": tasks}})
    assert "tab:gaussian-tier-summary" in latex
    assert "tab:gaussian-naturelm" in latex
    assert "tab:gaussian-parse" in latex
    assert "t1 snr regression & MAE (dB)" in latex
    assert "& 8 & 8 & 20.0 & 20.0" in latex
