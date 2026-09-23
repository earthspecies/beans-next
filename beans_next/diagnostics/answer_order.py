"""Deterministic answer-order permutation for explicit multiple-choice prompts."""

from __future__ import annotations

import hashlib
import re

from beans_next.api.types import DatasetExample

_INLINE_MARKER = re.compile(r"(?:(?<=Options: )|(?<=, ))([a-z])\) ")
_LINE_MARKER = re.compile(r"(?m)^([A-Z]): ")
_OPTION_BLOCK_SUFFIX = re.compile(r"\n\n")


def _permutation(size: int, *, sample_id: str, seed: int) -> list[int]:
    order = sorted(
        range(size),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{sample_id}\0{index}".encode()
        ).digest(),
    )
    if order == list(range(size)) and size > 1:
        order = order[1:] + order[:1]
    return order


def _split_inline(instruction: str) -> tuple[str, list[str], str, list[str]] | None:
    markers = list(_INLINE_MARKER.finditer(instruction))
    if len(markers) < 2:
        return None
    labels = [match.group(1) for match in markers]
    if labels != [chr(ord("a") + index) for index in range(len(labels))]:
        return None
    prefix = instruction[: markers[0].start(1)]
    choices: list[str] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(instruction)
        )
        text = instruction[start:end]
        if index + 1 < len(markers):
            text = text.removesuffix(", ")
        choices.append(text)
    return prefix, choices, "", labels


def _split_lines(instruction: str) -> tuple[str, list[str], str, list[str]] | None:
    markers = list(_LINE_MARKER.finditer(instruction))
    if len(markers) < 2:
        return None
    labels = [match.group(1) for match in markers]
    if labels != [chr(ord("A") + index) for index in range(len(labels))]:
        return None
    suffix_match = _OPTION_BLOCK_SUFFIX.search(instruction, markers[-1].end())
    option_end = suffix_match.start() if suffix_match is not None else len(instruction)
    prefix = instruction[: markers[0].start(1)]
    choices: list[str] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else option_end
        choices.append(instruction[start:end].rstrip("\n"))
    suffix = instruction[option_end:]
    return prefix, choices, suffix, labels


def permute_answer_order(
    example: DatasetExample, *, seed: int
) -> DatasetExample:
    """Permute explicit answer choices and remap the reference label.

    Prompts without textual choices, including audio-only few-shot A/B/C/D tasks,
    are returned unchanged.

    Returns
    -------
    DatasetExample
        A copied row with permuted choices and remapped label, or the input row
        when it has no supported explicit choices.
    """
    instruction = example.metadata.get("instruction")
    label = example.labels
    if not isinstance(instruction, str) or not isinstance(label, str):
        return example

    split = _split_inline(instruction) or _split_lines(instruction)
    if split is None:
        return example
    prefix, choices, suffix, labels = split
    textual_choices = [
        re.sub(r"</?Audio>|<AudioHere>|\[audio\]", "", choice).strip()
        for choice in choices
    ]
    if not any(textual_choices):
        return example
    normalized_label = label.strip()
    if normalized_label not in labels:
        return example

    order = _permutation(len(choices), sample_id=example.sample_id, seed=seed)
    old_correct = labels.index(normalized_label)
    new_correct = order.index(old_correct)
    if labels[0].islower():
        body = ", ".join(
            f"{labels[new_index]}) {choices[old_index]}"
            for new_index, old_index in enumerate(order)
        )
    else:
        body = "\n".join(
            f"{labels[new_index]}: {choices[old_index]}"
            for new_index, old_index in enumerate(order)
        )
    metadata = dict(example.metadata)
    metadata["instruction"] = prefix + body + suffix
    metadata["answer_order_permutation"] = order
    return example.model_copy(
        update={"labels": labels[new_correct], "metadata": metadata}
    )
