"""Audio placeholder markers aligned with the `predictions_v1` prompt convention.

The i-th entry in `audio_inputs` on the wire aligns with the i-th `AUDIO_HERE`
placeholder in the rendered user/system message text (see DESIGN §4.1).
"""

# Outer wrapper (NatureLM-style multimodal marker)
AUDIO_OPEN_TAG: str = "<Audio>"
AUDIO_CLOSE_TAG: str = "</Audio>"

# Placeholder token replaced by the launcher when binding waveform i to the prompt
AUDIO_HERE_TAG: str = "<AudioHere>"

# Full placeholder span as used in example prompts (single slot)
AUDIO_PLACEHOLDER: str = f"{AUDIO_OPEN_TAG}{AUDIO_HERE_TAG}{AUDIO_CLOSE_TAG}"

__all__ = [
    "AUDIO_CLOSE_TAG",
    "AUDIO_HERE_TAG",
    "AUDIO_OPEN_TAG",
    "AUDIO_PLACEHOLDER",
]
