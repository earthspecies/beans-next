"""HuggingFace-backed loader for the BirdSet evaluation benchmark.

Loads BirdSet test splits from ``DBD-research-group/BirdSet`` on Hugging Face
Hub and converts eBird code labels to scientific names using the eBird taxonomy.

The eBird taxonomy CSV is resolved via:
1. ``BEANS_NEXT_EBIRD_TAXONOMY_CSV`` environment variable (explicit local path).
2. ``huggingface_hub.hf_hub_download`` from the BirdSet dataset repo
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
    _ensure_audio_path_from_array,
    require_datasets,
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
# used in the esp_data BirdSet labels. Only non-identity entries are listed.
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
            sci = (row.get("SCI_NAME") or "").strip()
            if code and sci:
                mapping[code] = sci
    return mapping


@lru_cache(maxsize=1)
def _ebird_taxonomy() -> dict[str, str]:
    """Return the cached eBird species-code → scientific-name mapping.

    Resolution order:
    1. ``BEANS_NEXT_EBIRD_TAXONOMY_CSV`` env var (must point to an existing file).
    2. ``huggingface_hub.hf_hub_download`` from the BirdSet dataset repo.

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
    """
    def _canonical(sci: str) -> str:
        return _HF_TO_CANONICAL.get(sci, sci)

    multilabel_ints = row.get("ebird_code_multilabel")
    if isinstance(multilabel_ints, list) and multilabel_ints:
        codes = [multi_feat.int2str(i) for i in multilabel_ints]
        sci_names = list(
            dict.fromkeys(
                _canonical(taxonomy[c]) for c in codes if c in taxonomy
            )
        )
        if sci_names:
            return sci_names

    single_int = row.get("ebird_code")
    if single_int is not None:
        try:
            code = single_feat.int2str(int(single_int))
            sci = taxonomy.get(code)
            if sci:
                return [_canonical(sci)]
        except Exception:  # noqa: BLE001
            pass

    return None


def iter_hf_birdset_examples(
    *,
    subset: str,
    split: str = "test",
    task_id: str | None = None,
    limit: int | None = None,
) -> Iterator[DatasetExample]:
    """Yield ``DatasetExample`` rows for a BirdSet subset from Hugging Face Hub.

    Loads ``DBD-research-group/BirdSet``, converts eBird code integer labels to
    scientific names via the eBird taxonomy, and materializes audio to local WAV
    files for prompt evaluation.

    Parameters
    ----------
    subset
        BirdSet subset of the form ``"CONFIG-SPLIT"`` (e.g. ``"HSN-test_5s"``).
    split
        Split label stored on each ``DatasetExample`` (default ``"test"``).
    task_id
        Optional eval-task id stored on each yielded example.
    limit
        Optional maximum number of examples to yield.

    Yields
    ------
    DatasetExample
        One normalized example per row, in dataset order.

    Raises
    ------
    TypeError
        If ``load_dataset`` returns a non-map-style dataset.
    """
    hf_config, hf_split = _parse_birdset_subset(subset)
    datasets = require_datasets()
    taxonomy = _ebird_taxonomy()

    loaded = datasets.load_dataset(
        _BIRDSET_HF_REPO,
        hf_config,
        split=hf_split,
        trust_remote_code=True,
    )

    if not hasattr(loaded, "__len__") or not hasattr(loaded, "__getitem__"):
        raise TypeError(
            f"load_dataset returned a non-map dataset for BirdSet subset {subset!r}. "
            "BirdSet test splits should be map-style."
        )

    single_feat = loaded.features["ebird_code"]
    multi_feat = loaded.features["ebird_code_multilabel"].feature

    n_rows = len(loaded)
    _LOG.debug(
        "hf_birdset: loaded %s config=%s split=%s rows=%d",
        _BIRDSET_HF_REPO,
        hf_config,
        hf_split,
        n_rows,
    )

    for ordinal in range(n_rows):
        if limit is not None and ordinal >= limit:
            break

        row: dict[str, Any] = loaded[ordinal]
        sample_id = synthesize_hf_sample_id(
            path_or_id=_BIRDSET_HF_REPO,
            split=subset,
            revision=None,
            ordinal=ordinal,
        )

        labels = _birdset_hf_labels(
            row,
            single_feat=single_feat,
            multi_feat=multi_feat,
            taxonomy=taxonomy,
        )

        audio_path = _ensure_audio_path_from_array(
            row.get("audio"),
            sample_id=sample_id,
        )
        meta: dict[str, Any] = {}
        if audio_path:
            meta["audio_path"] = audio_path

        yield DatasetExample(
            sample_id=sample_id,
            task_id=task_id,
            split=split,
            labels=labels,
            metadata=meta,
        )
