"""Dataset loading utilities (HuggingFace map + streaming + optional Polars)."""

from beans_next.datasets.base import (
    dataset_name_equals,
    hf_row_metadata,
    hf_row_to_dataset_example,
    require_datasets,
    resolve_hf_sample_id,
    synthesize_hf_sample_id,
)
from beans_next.datasets.esp_data import (
    iter_esp_data_beans_next_multiaudio_examples,
    iter_esp_data_beans_zero_examples,
    iter_esp_data_birdset_examples,
    require_esp_data,
)
from beans_next.datasets.hf import iter_hf_dataset_examples
from beans_next.datasets.hf_multiaudio import (
    beans_next_multiaudio_row_filter,
    iter_hf_streaming_multiaudio_examples,
)
from beans_next.datasets.hf_streaming import iter_hf_streaming_examples
from beans_next.datasets.polars import (
    iter_polars_parquet_examples,
    require_polars,
    resolve_polars_sample_id,
    synthesize_polars_sample_id,
)

__all__ = [
    "dataset_name_equals",
    "iter_esp_data_beans_next_multiaudio_examples",
    "iter_esp_data_beans_zero_examples",
    "iter_esp_data_birdset_examples",
    "hf_row_metadata",
    "hf_row_to_dataset_example",
    "iter_hf_dataset_examples",
    "iter_hf_streaming_examples",
    "beans_next_multiaudio_row_filter",
    "iter_hf_streaming_multiaudio_examples",
    "iter_polars_parquet_examples",
    "require_datasets",
    "require_esp_data",
    "require_polars",
    "resolve_hf_sample_id",
    "resolve_polars_sample_id",
    "synthesize_hf_sample_id",
    "synthesize_polars_sample_id",
]
