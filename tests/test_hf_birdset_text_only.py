"""BirdSet no-audio evaluation must not decode or materialize waveforms."""

import io
import tarfile
from pathlib import Path
from typing import Never

import pytest

from beans_next.datasets import hf_birdset


def test_metadata_only_stream_preserves_labels_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hf_birdset,
        "_birdset_metadata",
        lambda *a: [
            {"filepath": "clip.ogg", "ebird_code_multilabel": ["gretit1"]},
            {"filepath": "other.ogg", "ebird_code_multilabel": ["unknown"]},
        ],
    )
    monkeypatch.setattr(
        hf_birdset, "_ebird_taxonomy", lambda: {"gretit1": "Parus major"}
    )

    def forbidden(*args: object, **kwargs: object) -> Never:
        raise AssertionError("audio decoder called in text-only evaluation")

    monkeypatch.setattr(hf_birdset, "_birdset_audio", forbidden)
    examples = list(
        hf_birdset.iter_hf_birdset_examples(
            subset="HSN-test_5s", limit=1, load_audio=False, revision="pinned"
        )
    )
    assert len(examples) == 1
    assert examples[0].labels == ["Parus major"]
    assert "audio_path" not in examples[0].metadata
    monkeypatch.setattr(
        hf_birdset, "_birdset_audio", lambda *a: {"clip.ogg": "/tmp/clip.ogg"}
    )
    audio = list(
        hf_birdset.iter_hf_birdset_examples(
            subset="HSN-test_5s", limit=1, revision="pinned"
        )
    )
    assert audio[0].sample_id == examples[0].sample_id
    assert audio[0].labels == examples[0].labels
    assert audio[0].metadata["audio_path"] == "/tmp/clip.ogg"


def test_unknown_species_codes_are_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="unknown"):
        hf_birdset._birdset_hf_labels(
            {"ebird_code_multilabel": ["known", "unknown"]},
            single_feat=None,
            multi_feat=None,
            taxonomy={"known": "Parus major"},
        )


def test_archive_members_stay_inside_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    archive = tmp_path / "audio.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../clip.ogg")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"data"))
    monkeypatch.setenv("BEANS_NEXT_HF_AUDIO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **kw: str(archive)
    )
    result = hf_birdset._birdset_audio("HSN", "revision", {"clip.ogg"})
    path = Path(result["clip.ogg"])
    assert path.is_relative_to(tmp_path / "cache")
    assert path.read_bytes() == b"data"
    assert not (tmp_path / "clip.ogg").exists()
    assert archive.exists()
