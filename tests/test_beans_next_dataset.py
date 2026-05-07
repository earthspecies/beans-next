"""Focused tests for the BeansPro loader + registry (I15-C)."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import pytest
import yaml

from beans_next.api.types import DatasetExample
from beans_next.config.run_config import RegistryResolutionError, _load_suite_eval_tasks
from beans_next.prompts.renderer import AudioSlotSpec, PromptRenderer, PromptSpec


@pytest.fixture
def beans_next_instruction_snippet() -> str:
    # Copied from docs/beans_next_dataset.md ("bird-presence" example).
    return (
        "<Audio><AudioHere></Audio> Is there a bird vocalizing in this recording? "
        "Answer Yes or No."
    )


def _registry_root() -> Path:
    return Path(importlib.resources.files("beans_next")).joinpath("registry")


def test_registry_dataset_beans_next_esp_yaml_parses_and_declares_id() -> None:
    path = _registry_root() / "dataset" / "beans_next_esp.yaml"
    if not path.exists():
        pytest.skip("beans_next_esp dataset registry not present in this checkout")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "beans_next_esp" in data
    assert isinstance(data["beans_next_esp"], dict)


def test_suite_beans_next_core_expands_to_exactly_three_eval_tasks() -> None:
    try:
        eval_tasks = _load_suite_eval_tasks("beans_next_core")
    except RegistryResolutionError:
        pytest.skip("beans_next_core suite registry not present in this checkout")
    assert len(eval_tasks) == 43


@pytest.mark.parametrize(
    ("suite_id", "expected_len"),
    (
        ("beans_next_tier_1_hf", 8),
        ("beans_next_tier_2_hf", 8),
        ("beans_next_tier_3_hf", 21),
        ("beans_next_tier_4_hf", 6),
    ),
)
def test_suite_beans_next_per_tier_hf_expands_to_expected_task_count(
    suite_id: str,
    expected_len: int,
) -> None:
    try:
        eval_tasks = _load_suite_eval_tasks(suite_id)
    except RegistryResolutionError:
        pytest.skip(f"{suite_id} suite registry not present in this checkout")
    assert len(eval_tasks) == expected_len


def test_prompt_renderer_passes_instruction_through_unchanged_official_mode(
    beans_next_instruction_snippet: str,
) -> None:
    spec = PromptSpec(
        prompt_id="test.prompt.beans_next.instruction_passthrough.v1",
        message_templates=(("user", "{{ metadata.instruction }}"),),
        audio_slots=(
            AudioSlotSpec(
                metadata_key="audio_path",
                payload_type="file_path",
            ),
        ),
    )
    renderer = PromptRenderer(spec)
    ex = DatasetExample(
        sample_id="s0",
        task_id="beans_next_crow_description",
        split="test",
        labels="No",
        metadata={
            "instruction": beans_next_instruction_snippet,
            "audio_path": "/tmp/x.wav",
        },
    )
    req = renderer.render(ex)
    assert len(req.messages) == 1
    assert req.messages[0].content == beans_next_instruction_snippet


@pytest.fixture
def _requires_esp_data() -> None:
    pytest.importorskip("esp_data")


def test_build_dataset_example_from_beans_next_row_fixture(
    beans_next_example: DatasetExample,
) -> None:
    ex = beans_next_example
    assert isinstance(ex, DatasetExample)
    assert ex.sample_id == "beanspro:test:0"
    assert ex.task_id == "beans_next_crow_description"
    assert ex.split == "test"
    assert ex.labels == "A"
    assert "audio_path" in ex.metadata
    assert "instruction" in ex.metadata


def test_synthesize_esp_data_sample_id_is_stable_for_beans_next_dataset() -> None:
    from beans_next.datasets.esp_data import synthesize_esp_data_sample_id

    a = synthesize_esp_data_sample_id(
        dataset="beans_next",
        subset="crow-description",
        split="test",
        ordinal=0,
    )
    b = synthesize_esp_data_sample_id(
        dataset="beans_next",
        subset="crow-description",
        split="test",
        ordinal=0,
    )
    c = synthesize_esp_data_sample_id(
        dataset="beans_next",
        subset="crow-description",
        split="test",
        ordinal=1,
    )
    assert a == b
    assert a != c
    assert a.startswith("beanspro:esp_data:")


def test_normalize_birdset_row_prefers_scientific_name_fields() -> None:
    from beans_next.datasets.esp_data import _normalize_birdset_row

    row_species = {
        "species": "Erithacus rubecula",
        "species_common": "European Robin",
        "canonical_name_multispecies": '["Erithacus rubecula"]',
    }
    out_species = _normalize_birdset_row(row_species, data_root="gs://x/")
    assert out_species["output"] == ["Erithacus rubecula"]

    row_canonical = {
        "species": None,
        "species_common": None,
        "canonical_name_multispecies": '["Jynx torquilla"]',
    }
    out_canonical = _normalize_birdset_row(row_canonical, data_root="gs://x/")
    assert out_canonical["output"] == ["Jynx torquilla"]

    row_common_fallback = {
        "species": "",
        "canonical_name_multispecies": "",
        "species_common": "European Robin",
    }
    out_common = _normalize_birdset_row(row_common_fallback, data_root="gs://x/")
    assert out_common["output"] == ["European Robin"]


def test_normalize_birdset_row_multi_species_uses_canonical_authoritatively() -> None:
    from beans_next.datasets.esp_data import _normalize_birdset_row

    # JSON-list canonical_name_multispecies wins outright — even when other
    # taxonomy fields name a different species, those values are NOT mixed in
    # because canonical_name_multispecies is the authoritative ground-truth set.
    row_multi = {
        "canonical_name_multispecies": '["Turdus migratorius", "Junco hyemalis"]',
        "species": "Setophaga coronata",
        "scientific_name_unified_original": "Vireo gilvus",
        "species_common": "American Robin",
    }
    out_multi = _normalize_birdset_row(row_multi, data_root="gs://x/")
    assert out_multi["output"] == ["Turdus migratorius", "Junco hyemalis"]

    # Real list (not JSON string) is handled identically.
    row_list = {
        "canonical_name_multispecies": ["Turdus migratorius", "Junco hyemalis"],
    }
    out_list = _normalize_birdset_row(row_list, data_root="gs://x/")
    assert out_list["output"] == ["Turdus migratorius", "Junco hyemalis"]

    # Duplicates within canonical_name_multispecies are collapsed, order preserved.
    row_dupes = {
        "canonical_name_multispecies": [
            "Turdus migratorius",
            "Junco hyemalis",
            "Turdus migratorius",
        ],
    }
    out_dupes = _normalize_birdset_row(row_dupes, data_root="gs://x/")
    assert out_dupes["output"] == ["Turdus migratorius", "Junco hyemalis"]

    # When canonical_name_multispecies is empty, the next-priority field is used.
    row_fallback = {
        "canonical_name_multispecies": "",
        "species": "Setophaga coronata",
    }
    out_fallback = _normalize_birdset_row(row_fallback, data_root="gs://x/")
    assert out_fallback["output"] == ["Setophaga coronata"]
