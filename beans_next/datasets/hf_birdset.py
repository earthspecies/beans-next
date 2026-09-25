"""HuggingFace-backed loader for the BirdSet evaluation benchmark.

Loads BirdSet test_5s metadata and audio archives directly from a pinned payload
revision of ``DBD-research-group/BirdSet`` on Hugging Face Hub. Text-only runs
read only metadata; audio runs reuse the same rows and sample identifiers.

The eBird taxonomy CSV is resolved via:
1. ``BEANS_NEXT_EBIRD_TAXONOMY_CSV`` environment variable (explicit local path).
2. The bundled public eBird 2024 mapping, with source URL and download hash.
3. ``huggingface_hub.hf_hub_download`` from the BirdSet dataset repo
   (``resources/ebird_codes/eBird_taxonomy_v2024.csv``).

If neither source is available a ``RuntimeError`` with an actionable message
is raised.

Raises
------
RuntimeError
    If the eBird taxonomy CSV cannot be resolved.
ValueError
    If a subset name is not of the form ``"CONFIG-SPLIT"`` (e.g. ``"HSN-test_5s"``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from beans_next.api.types import DatasetExample
from beans_next.datasets.base import (
    synthesize_hf_sample_id,
)

_LOG = logging.getLogger(__name__)

_BIRDSET_HF_REPO = "DBD-research-group/BirdSet"
_EBIRD_TAXONOMY_CSV_ENV = "BEANS_NEXT_EBIRD_TAXONOMY_CSV"
_EBIRD_TAXONOMY_FILENAME = "resources/ebird_codes/eBird_taxonomy_v2024.csv"

# eBird codes used in the BirdSet HF dataset that were deprecated in later taxonomy
# releases. Maps the old code to the current scientific name directly.
_DEPRECATED_CODE_FALLBACK: dict[str, str] = {
    "runwre1": "Campylorhynchus rufinucha",
}

# Maps eBird taxonomy v2024 scientific names to the canonical evaluation vocabulary
# used in the benchmark BirdSet labels. Only non-identity entries are listed.
# Sources: genus renames (eBird 2021-2024), gender corrections, and eBird
# group-notation simplifications for newly added HF-only species.
_HF_TO_CANONICAL: dict[str, str] = {
    # Genus renames — Kinglets
    "Corthylio calendula": "Regulus calendula",
    # Genus renames — Woodpeckers
    "Dryobates villosus": "Leuconotopicus villosus",
    "Dryobates albolarvatus": "Leuconotopicus albolarvatus",
    "Dryobates passerinus": "Veniliornis passerinus",
    # Genus renames — Raptors
    "Astur cooperii": "Accipiter cooperii",
    "Daptrius chimachima": "Milvago chimachima",
    "Buteo plagiatus": "Buteo nitidus",
    # Genus renames — Hawaiian birds
    "Drepanis coccinea": "Vestiaria coccinea",
    "Hydrobates castro": "Oceanodroma castro",
    # Genus renames — Parakeets / Parrots
    "Eupsittula canicularis": "Aratinga canicularis",
    "Psittacara finschi": "Aratinga finschi",
    # Genus renames — Antbirds
    "Akletos goeldii": "Myrmeciza goeldii",
    "Myrmophylax atrothorax": "Myrmeciza atrothorax",
    "Myrmelastes hyperythrus": "Myrmeciza hyperythra",
    # Genus renames — Other Neotropical
    "Pachysylvia hypoxantha": "Hylophilus hypoxanthus",
    "Dendroplex picus": "Xiphorhynchus picus",
    "Dendroma erythroptera": "Philydor erythropterum",
    "Cyanocorax morio": "Psilorhinus morio",
    # Gender / epithet corrections
    "Aramides cajaneus": "Aramides cajanea",
    "Orthopsittaca manilatus": "Orthopsittaca manilata",
    # Evening Grosbeak genus change
    "Coccothraustes vespertinus": "Hesperiphona vespertina",
    # eBird group-notation → simplified name (new HF-only species)
    "Celeus undatus [grammicus Group]": "Celeus grammicus",
    "Empidonax difficilis [difficilis Group]": "Empidonax difficilis",
}


def _parse_birdset_subset(subset: str) -> tuple[str, str]:
    """Parse a BirdSet subset name into an HF config name and HF split name.

    Parameters
    ----------
    subset
        Subset name of the form ``"CONFIG-SPLIT"`` (e.g. ``"HSN-test_5s"``).

    Returns
    -------
    tuple[str, str]
        ``(hf_config, hf_split)`` — e.g. ``("HSN", "test_5s")``.

    Raises
    ------
    ValueError
        If the subset name cannot be split into two non-empty parts on ``"-"``.
    """
    parts = subset.strip().split("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"BirdSet subset must be of the form 'CONFIG-SPLIT', got {subset!r}. "
            "Example: 'HSN-test_5s'."
        )
    return parts[0], parts[1]


def _parse_ebird_taxonomy_csv(csv_path: str) -> dict[str, str]:
    """Parse an eBird taxonomy CSV into a species-code → scientific-name mapping.

    Parameters
    ----------
    csv_path
        Path to the eBird taxonomy CSV file (``eBird_taxonomy_v2024.csv``).

    Returns
    -------
    dict[str, str]
        Mapping from eBird species code to scientific name.

    """
    import csv

    mapping: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = (row.get("SPECIES_CODE") or "").strip()
            sci = (row.get("SCI_NAME") or row.get("SCIENTIFIC_NAME") or "").strip()
            if code and sci:
                mapping[code] = sci
    return mapping


@lru_cache(maxsize=1)
def _ebird_taxonomy() -> dict[str, str]:
    """Return the cached eBird species-code → scientific-name mapping.

    Resolution order:
    1. ``BEANS_NEXT_EBIRD_TAXONOMY_CSV`` env var (must point to an existing file).
    2. Bundled eBird 2024 mapping with provenance.
    3. ``huggingface_hub.hf_hub_download`` from the BirdSet dataset repo.

    Returns
    -------
    dict[str, str]
        Mapping from eBird species code to scientific name.

    Raises
    ------
    RuntimeError
        If the taxonomy CSV cannot be resolved from any source.
    """
    csv_path = os.environ.get(_EBIRD_TAXONOMY_CSV_ENV, "").strip()
    if csv_path and os.path.isfile(csv_path):
        _LOG.debug("ebird taxonomy: using env var path %s", csv_path)
        mapping = _parse_ebird_taxonomy_csv(csv_path)
        mapping.update(
            {k: v for k, v in _DEPRECATED_CODE_FALLBACK.items() if k not in mapping}
        )
        return mapping

    import json
    from pathlib import Path

    bundled = Path(__file__).resolve().parents[1] / "registry/ebird_taxonomy_2024.json"
    if bundled.is_file():
        mapping = json.loads(bundled.read_text())["species_code_to_scientific_name"]
        mapping.update(
            {k: v for k, v in _DEPRECATED_CODE_FALLBACK.items() if k not in mapping}
        )
        return mapping

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        downloaded = hf_hub_download(
            repo_id=_BIRDSET_HF_REPO,
            filename=_EBIRD_TAXONOMY_FILENAME,
            repo_type="dataset",
        )
        _LOG.debug("ebird taxonomy: downloaded to %s", downloaded)
        mapping = _parse_ebird_taxonomy_csv(downloaded)
        mapping.update(
            {k: v for k, v in _DEPRECATED_CODE_FALLBACK.items() if k not in mapping}
        )
        return mapping
    except Exception as exc:
        _LOG.debug("ebird taxonomy hf_hub_download failed: %s", exc)

    raise RuntimeError(
        "eBird taxonomy CSV not found. Provide one of:\n"
        f"  1. Set {_EBIRD_TAXONOMY_CSV_ENV} to the path of eBird_taxonomy_v2024.csv.\n"
        f"  2. Ensure huggingface_hub can download "
        f"{_EBIRD_TAXONOMY_FILENAME!r} from {_BIRDSET_HF_REPO!r}.\n"
        "The CSV is included in the BirdSet repository under resources/ebird_codes/."
    )


def _birdset_hf_labels(
    row: dict[str, Any],
    *,
    single_feat: object,
    multi_feat: object,
    taxonomy: dict[str, str],
) -> list[str] | None:
    """Extract scientific-name labels from a BirdSet HF row.

    Prefers ``ebird_code_multilabel`` (full multi-label ground truth) when
    populated, falling back to the single focal-species ``ebird_code``.

    Parameters
    ----------
    row
        A single decoded row from the BirdSet HF dataset.
    single_feat
        The ``ClassLabel`` feature for ``ebird_code``.
    multi_feat
        The inner ``ClassLabel`` feature for ``ebird_code_multilabel.feature``.
    taxonomy
        eBird species code → scientific name mapping.

    Returns
    -------
    list[str] or None
        Deduplicated list of scientific names, or ``None`` when not resolvable.

    Raises
    ------
    ValueError
        If a nonempty multi-label target contains an unknown species code.
    """

    def _canonical(sci: str) -> str:
        return _HF_TO_CANONICAL.get(sci, sci)

    multilabel_ints = row.get("ebird_code_multilabel")
    if isinstance(multilabel_ints, list) and multilabel_ints:
        codes = [
            i if isinstance(i, str) else multi_feat.int2str(i) for i in multilabel_ints
        ]
        unknown = set(codes) - taxonomy.keys()
        if unknown:
            raise ValueError(f"BirdSet taxonomy lacks species codes: {sorted(unknown)}")
        sci_names = list(
            dict.fromkeys(_canonical(taxonomy[c]) for c in codes if c in taxonomy)
        )
        if sci_names:
            return sci_names

    single_int = row.get("ebird_code")
    if single_int is not None:
        try:
            code = (
                single_int
                if isinstance(single_int, str)
                else single_feat.int2str(int(single_int))
            )
            sci = taxonomy.get(code)
            if sci:
                return [_canonical(sci)]
        except Exception:  # noqa: BLE001
            pass

    return None


# BirdSet's builder branch and payload branch have independent histories.
# Pin the actual payloads rather than the builder's mutable resolve/data URLs.
_BIRDSET_DATA_REVISION = "806ed2cda4ddcbe6efa194ccafff930aa0e557ce"


def _birdset_metadata(config: str, revision: str) -> list[dict[str, Any]]:
    """Read only the requested test_5s metadata from the HF payload branch.

    Returns
    -------
    list[dict[str, Any]]
        File names and multi-label species codes in metadata order.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        _BIRDSET_HF_REPO,
        f"{config}/{config}_metadata_test_5s.parquet",
        repo_type="dataset",
        revision=revision,
    )
    return pq.read_table(
        path, columns=["filepath", "ebird_code_multilabel"]
    ).to_pylist()


def _birdset_audio(config: str, revision: str, names: set[str]) -> dict[str, str]:
    """Materialize requested clips from pinned HF archives under a shared lock.

    Archive members are copied by basename; archive paths are never extracted.
    Existing complete clips are reused and no source or cache files are deleted.

    Returns
    -------
    dict[str, str]
        Requested basenames mapped to complete local audio files.

    Raises
    ------
    ValueError
        If a requested member cannot be read or is absent from all archives.
    """
    import shutil
    import tarfile
    from pathlib import Path

    from filelock import FileLock
    from huggingface_hub import hf_hub_download

    cache = (
        Path(
            os.environ.get(
                "BEANS_NEXT_HF_AUDIO_CACHE_DIR",
                str(
                    Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
                    / "birdset-audio"
                ),
            )
        )
        / "birdset"
        / revision
        / config
    )
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / "materialize.lock")):
        missing = {name for name in names if not (cache / name).is_file()}
        for shard in range(1, (4 if config == "SSW" else 1) + 1):
            if not missing:
                break
            archive = hf_hub_download(
                _BIRDSET_HF_REPO,
                f"{config}/{config}_test5s_shard_{shard:04d}.tar.gz",
                repo_type="dataset",
                revision=revision,
            )
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    if name not in missing or not member.isfile():
                        continue
                    stream = tar.extractfile(member)
                    if stream is None:
                        raise ValueError(
                            f"Unreadable BirdSet archive member: {member.name}"
                        )
                    temporary = cache / f".{name}.{os.getpid()}.tmp"
                    with stream, temporary.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    temporary.replace(cache / name)
                    missing.remove(name)
        if missing:
            raise ValueError(
                f"BirdSet archive lacks requested clips: {sorted(missing)[:10]}"
            )
    return {name: str(cache / name) for name in names}


def iter_hf_birdset_examples(
    *,
    subset: str,
    split: str = "test",
    task_id: str | None = None,
    limit: int | None = None,
    load_audio: bool = True,
    revision: str | None = None,
) -> Iterator[DatasetExample]:
    """Yield pinned BirdSet test metadata with optional HF audio materialization.

    Parameters
    ----------
    subset
        CONFIG-test_5s identifier, such as HSN-test_5s.
    split
        Split label stored on each example.
    task_id
        Evaluation task identifier.
    limit
        Maximum number of metadata rows; None uses the complete split.
    load_audio
        False reads metadata only and never opens audio archives.
    revision
        Builder revision recorded for provenance. Payloads use the independently
        pinned BEANS_NEXT_BIRDSET_DATA_REVISION, or the bundled default.

    Raises
    ------
    ValueError
        For unsupported splits, unknown species codes, or missing audio clips.

    Yields
    ------
    DatasetExample
        Same IDs, labels, and order in every modality condition.
    """
    from pathlib import Path

    config, hf_split = _parse_birdset_subset(subset)
    if hf_split != "test_5s":
        raise ValueError("BirdSet evaluation requires the test_5s split")
    data_revision = os.environ.get(
        "BEANS_NEXT_BIRDSET_DATA_REVISION", _BIRDSET_DATA_REVISION
    )
    rows = _birdset_metadata(config, data_revision)
    if limit is not None:
        rows = rows[:limit]
    names = {Path(row["filepath"]).name for row in rows}
    paths = _birdset_audio(config, data_revision, names) if load_audio else {}
    taxonomy = _ebird_taxonomy()
    for ordinal, row in enumerate(rows):
        name = Path(row["filepath"]).name
        labels = _birdset_hf_labels(
            row, single_feat=None, multi_feat=None, taxonomy=taxonomy
        )
        meta = {
            "file_name": name,
            "birdset_data_revision": data_revision,
            "birdset_builder_revision": revision,
            "birdset_target_present": bool(labels),
        }
        if load_audio:
            meta["audio_path"] = paths[name]
        yield DatasetExample(
            sample_id=synthesize_hf_sample_id(
                path_or_id=f"{_BIRDSET_HF_REPO}/{config}/{name}",
                split=subset,
                revision=data_revision,
                ordinal=ordinal,
            ),
            task_id=task_id,
            split=split,
            labels=labels,
            metadata=meta,
        )
