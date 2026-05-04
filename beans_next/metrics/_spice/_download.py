"""Download Stanford CoreNLP JARs required by SPICE.

JARs are cached to ``~/.cache/beans-next/spice/lib/`` and reused on subsequent
calls. Run via ``beans-next setup-spice`` or call :func:`download_stanford_models`
directly.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_CACHE_LIB = Path.home() / ".cache" / "beans-next" / "spice" / "lib"

_STANFORD_JARS: dict[str, str] = {
    "stanford-corenlp-3.6.0.jar": (
        "https://repo1.maven.org/maven2/edu/stanford/nlp/"
        "stanford-corenlp/3.6.0/stanford-corenlp-3.6.0.jar"
    ),
    "stanford-corenlp-3.6.0-models.jar": (
        "https://repo1.maven.org/maven2/edu/stanford/nlp/"
        "stanford-corenlp/3.6.0/stanford-corenlp-3.6.0-models.jar"
    ),
}


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, printing progress to stdout.

    Parameters
    ----------
    url : str
        Source URL.
    dest : Path
        Destination file path.

    """

    def _hook(count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            sys.stdout.write(f"\r  downloaded {count * block_size // (1024 * 1024)} MB")
        else:
            pct = min(100, count * block_size * 100 // total_size)
            mb = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:3d}%  {mb:.1f} MB")
        sys.stdout.flush()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_stanford_models(force: bool = False) -> None:
    """Download Stanford CoreNLP 3.6.0 JARs to the beans-next cache.

    Files are saved to ``~/.cache/beans-next/spice/lib/``. Already-present JARs
    are skipped unless *force* is ``True``.

    Parameters
    ----------
    force : bool
        Re-download even if the target file already exists.
    """
    _CACHE_LIB.mkdir(parents=True, exist_ok=True)
    for filename, url in _STANFORD_JARS.items():
        dest = _CACHE_LIB / filename
        if dest.exists() and not force:
            print(f"  already present: {dest}")
            continue
        print(f"  downloading {filename} …")
        _download_with_progress(url, dest)
        print(f"  saved → {dest}")


def stanford_jars_present() -> bool:
    """Return ``True`` when all required Stanford CoreNLP JARs are cached.

    Returns
    -------
    bool
        ``True`` iff every expected JAR exists in ``~/.cache/beans-next/spice/lib/``.
    """
    return all((_CACHE_LIB / name).exists() for name in _STANFORD_JARS)
