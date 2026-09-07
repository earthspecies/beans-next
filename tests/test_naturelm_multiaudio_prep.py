"""Tests for NatureLM v1.1 multi-audio waveform preparation.

NatureLM splices one audio embedding per `<AudioHere>` placeholder, flattening
across the audio batch, so N clips for one conversation are passed as the audio
batch. These tests pin the per-clip preprocessing that builds that batch.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_SERVE = Path(__file__).resolve().parents[1] / "examples/servers/naturelm-v1.1/serve.py"


def _prepare() -> Callable[..., Any]:
    """Exec `_prepare_waveform` from the launcher without its heavy imports.

    Returns
    -------
    collections.abc.Callable
        The launcher's `_prepare_waveform` function.
    """
    src = _SERVE.read_text()
    start = src.index("def _prepare_waveform(")
    end = src.index("\ndef ", start)
    ns: dict[str, Any] = {"np": np, "Any": Any}
    exec(  # noqa: S102
        compile("from __future__ import annotations\n" + src[start:end],
                "serve_extract", "exec"), ns)
    return ns["_prepare_waveform"]


def test_crops_and_pads_to_target_length() -> None:
    prep = _prepare()
    sr = 16000
    short = np.zeros(sr, dtype=np.float32)          # 1 s
    wav, mask, err = prep(short, sr, sample_rate=sr, max_length_seconds=10)
    assert err is None
    assert wav.shape[0] == sr * 10
    assert mask[:sr].sum() == 0          # real signal not masked
    assert bool(mask[sr:].all())         # padding masked

    long = np.zeros(sr * 30, dtype=np.float32)      # 30 s
    wav, mask, err = prep(long, sr, sample_rate=sr, max_length_seconds=10)
    assert err is None and wav.shape[0] == sr * 10
    assert mask.sum() == 0               # nothing padded when cropping


def test_stereo_is_downmixed() -> None:
    prep = _prepare()
    sr = 16000
    stereo = np.stack([np.ones(sr), -np.ones(sr)], axis=1).astype(np.float32)
    wav, _, err = prep(stereo, sr, sample_rate=sr, max_length_seconds=1)
    assert err is None and wav.ndim == 1
    assert np.allclose(wav[:sr], 0.0)    # +1 and -1 average to 0


def test_values_are_clamped() -> None:
    prep = _prepare()
    sr = 16000
    loud = (np.ones(sr) * 5.0).astype(np.float32)
    wav, _, err = prep(loud, sr, sample_rate=sr, max_length_seconds=1)
    assert err is None
    assert wav.max() <= 1.0 and wav.min() >= -1.0


def test_batch_of_clips_stacks_to_uniform_shape() -> None:
    """Clips of differing length must stack into one (N, T) audio batch."""
    prep = _prepare()
    sr = 16000
    clips = [np.zeros(int(sr * s), dtype=np.float32) for s in (1, 3, 7)]
    out = [prep(c, sr, sample_rate=sr, max_length_seconds=10) for c in clips]
    assert all(e is None for _, _, e in out)
    batch = np.stack([w for w, _, _ in out], axis=0)
    masks = np.stack([m for _, m, _ in out], axis=0)
    assert batch.shape == (3, sr * 10)
    assert masks.shape == (3, sr * 10)


def test_launcher_counts_placeholders_against_clip_count() -> None:
    """The launcher must reject a placeholder/clip mismatch, not guess."""
    src = _SERVE.read_text()
    assert "_NATURELM_AUDIO_PLACEHOLDER" in src
    assert re.search(r"placeholder\(s\) but\s*\"?\s*\n?\s*f?\"?\{n_audio\}", src) or \
        "they must match" in src
