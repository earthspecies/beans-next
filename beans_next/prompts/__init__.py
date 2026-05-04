"""Prompt rendering for bioacoustic benchmark tasks (Jinja2 + audio placeholders)."""

from beans_next.prompts.renderer import (
    AudioPlaceholderAlignmentError,
    AudioSlotSpec,
    PromptRenderer,
    PromptSpec,
    builtin_prompt_registry_path,
    load_builtin_prompt_yaml,
    load_prompt_spec_from_path,
)

__all__ = [
    "AudioPlaceholderAlignmentError",
    "AudioSlotSpec",
    "PromptRenderer",
    "PromptSpec",
    "builtin_prompt_registry_path",
    "load_builtin_prompt_yaml",
    "load_prompt_spec_from_path",
]
