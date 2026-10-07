"""Generate example identifiers from source keys and evaluation content."""

import hashlib
import json
from collections.abc import Mapping, Sequence


def content_example_id(
    *,
    source_key: str,
    task: str,
    audio_paths: Sequence[str],
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Return a deterministic ID that changes with the evaluation input or target.

    Parameters
    ----------
    source_key
        Stable source-row key, distinguishing repeated examples.
    task
        Evaluation task name.
    audio_paths
        Ordered paths to content-addressed audio files.
    messages
        Ordered prompt and target messages.

    Returns
    -------
    str
        A `beans_next_` prefix followed by 32 hexadecimal characters.

    Raises
    ------
    ValueError
        If a required key, audio path, or message is empty or invalid.
    """
    if not source_key or not task or not audio_paths or not messages:
        raise ValueError("Source key, task, audio paths, and messages are required")
    if any(not isinstance(path, str) or not path for path in audio_paths):
        raise ValueError("Audio paths must be nonempty strings")
    if any(
        not isinstance(message.get(key), str) or not message[key]
        for message in messages
        for key in ("role", "content")
    ):
        raise ValueError("Messages require nonempty role and content strings")
    payload = json.dumps(
        {
            "source_key": source_key,
            "task": task,
            "audio_paths": list(audio_paths),
            "messages": [dict(message) for message in messages],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(("beans-next-example-v2\0" + payload).encode()).hexdigest()
    return "beans_next_" + digest[:32]
