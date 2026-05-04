"""SPICE metric (Semantic Propositional Image Caption Evaluation).

Reference implementation adapted from:
    Anderson et al. (2016) — https://panderson.me/spice/
    https://github.com/salaniz/pycocoevalcap

Requires Java 8+ on ``PATH`` and Stanford CoreNLP 3.6.0 JARs.
Install the JARs with ``beans-next setup-spice``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from beans_next.metrics._spice._download import (
    _CACHE_LIB,
    _STANFORD_JARS,
)
from beans_next.metrics.base import MetricsError

_SPICE_JAR = Path(__file__).parent / "spice-1.0.jar"
_BUNDLED_LIB = Path(__file__).parent / "lib"


class SpiceUnavailableError(MetricsError):
    """Raised when SPICE cannot run.

    Possible reasons: Java is absent from ``PATH``, or the Stanford CoreNLP JARs
    have not been installed (run ``beans-next setup-spice``).
    """


def _require_java() -> None:
    """Raise :exc:`SpiceUnavailableError` when ``java`` is not on ``PATH``.

    Raises
    ------
    SpiceUnavailableError
        If the ``java`` executable cannot be found.
    """
    if shutil.which("java") is None:
        raise SpiceUnavailableError(
            "Java not found on PATH. Install Java 8+ to use SPICE."
        )


def _require_stanford_jars() -> None:
    """Raise :exc:`SpiceUnavailableError` when cached Stanford JARs are missing.

    Raises
    ------
    SpiceUnavailableError
        If one or more Stanford CoreNLP JARs are absent from the cache directory.
    """
    missing = [n for n in _STANFORD_JARS if not (_CACHE_LIB / n).exists()]
    if missing:
        raise SpiceUnavailableError(
            f"Stanford CoreNLP JAR(s) missing from {_CACHE_LIB}: {missing}. "
            "Run `beans-next setup-spice` to download them."
        )


def check_spice_available() -> None:
    """Check all SPICE prerequisites, raising on first failure."""
    _require_java()
    _require_stanford_jars()


class Spice:
    """Compute the SPICE metric over a set of hypothesis/reference pairs.

    SPICE measures semantic propositional similarity via scene-graph parsing.
    It requires Java 8+ and Stanford CoreNLP 3.6.0; call ``check_spice_available``
    or ``beans-next setup-spice`` before first use.
    """

    def _float_convert(self, obj: object) -> float:
        """Convert *obj* to float, returning ``nan`` on failure.

        Parameters
        ----------
        obj : object
            Value to convert.

        Returns
        -------
        float
            Parsed float, or ``nan`` if conversion fails.
        """
        try:
            return float(obj)  # type: ignore[arg-type]
        except Exception:
            return float("nan")

    def compute_score(
        self,
        gts: dict[str, list[str]],
        res: dict[str, list[str]],
    ) -> tuple[float, list[dict[str, dict[str, float]]]]:
        """Compute SPICE score for a corpus.

        Parameters
        ----------
        gts : dict[str, list[str]]
            Ground-truth captions, keyed by sample id. Each value is a list of
            one or more reference strings.
        res : dict[str, list[str]]
            Hypothesis captions, keyed by sample id. Each value must be a
            one-element list.

        Returns
        -------
        tuple[float, list[dict[str, dict[str, float]]]]
            ``(mean_score, per_sample_scores)`` where ``mean_score`` is the
            corpus-level SPICE F1 and each element of ``per_sample_scores``
            is a ``{category: {p, r, f}}`` dict.

        """
        check_spice_available()

        assert sorted(gts.keys()) == sorted(res.keys())
        img_ids = sorted(gts.keys())

        input_data = []
        for img_id in img_ids:
            hypo = res[img_id]
            ref = gts[img_id]
            assert isinstance(hypo, list) and len(hypo) == 1
            assert isinstance(ref, list) and len(ref) >= 1
            input_data.append({"image_id": img_id, "test": hypo[0], "refs": ref})

        work_dir = tempfile.mkdtemp(prefix="beans-next-spice-")
        try:
            jar_dest = os.path.join(work_dir, "spice-1.0.jar")
            shutil.copy2(str(_SPICE_JAR), jar_dest)

            lib_dest = os.path.join(work_dir, "lib")
            os.makedirs(lib_dest)
            for jar in _BUNDLED_LIB.glob("*.jar"):
                shutil.copy2(str(jar), os.path.join(lib_dest, jar.name))
            for jar_name in _STANFORD_JARS:
                src = str(_CACHE_LIB / jar_name)
                shutil.copy2(src, os.path.join(lib_dest, jar_name))

            in_fd, in_path = tempfile.mkstemp(suffix=".json", dir=work_dir)
            with os.fdopen(in_fd, "w") as fh:
                json.dump(input_data, fh, indent=2)

            out_fd, out_path = tempfile.mkstemp(suffix=".json", dir=work_dir)
            os.close(out_fd)

            spice_cmd = [
                "java",
                "-jar",
                "-Xmx8G",
                "spice-1.0.jar",
                in_path,
                "-out",
                out_path,
                "-subset",
                "-silent",
            ]
            subprocess.check_call(spice_cmd, cwd=work_dir)

            with open(out_path) as data_file:
                results = json.load(data_file)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        img_id_to_scores: dict[str, dict[str, object]] = {}
        spice_scores: list[float] = []
        for item in results:
            img_id_to_scores[item["image_id"]] = item["scores"]
            spice_scores.append(self._float_convert(item["scores"]["All"]["f"]))

        average_score = float(np.mean(np.array(spice_scores)))
        per_sample: list[dict[str, dict[str, float]]] = []
        for img_id in img_ids:
            score_set: dict[str, dict[str, float]] = {}
            for category, score_tuple in img_id_to_scores[img_id].items():
                assert isinstance(score_tuple, dict)
                score_set[category] = {
                    k: self._float_convert(v) for k, v in score_tuple.items()
                }
            per_sample.append(score_set)

        return average_score, per_sample

    def method(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            Always ``"SPICE"``.
        """
        return "SPICE"
