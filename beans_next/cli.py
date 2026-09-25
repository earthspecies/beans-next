"""Command-line interface for BEANS-Next (``beans-next``).

Subcommands dispatch to the benchmark runner and to bundled
registry assets (prompt YAMLs under ``beans_next/registry``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import yaml


def _workers_arg(raw: str) -> int:
    """Coerce a CLI workers argument into a positive integer.

    Parameters
    ----------
    raw
        Raw string from the CLI.

    Returns
    -------
    int
        Parsed worker count (at least ``1``).

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not an integer or is less than ``1``.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("--workers must be >= 1")
    return value


def _registry_root() -> Path:
    """Return the root ``beans_next/registry`` directory.

    Returns
    -------
    pathlib.Path
        Absolute path to bundled registry assets.
    """
    return Path(__file__).resolve().parent / "registry"


def _iter_registry_yaml_files(*, kind: str | None) -> Iterator[tuple[str, Path]]:
    """Yield ``(kind, path)`` for each bundled registry YAML file.

    Parameters
    ----------
    kind
        If set, restrict to that subdirectory name (for example ``prompt``).
        Otherwise all immediate subdirectories are scanned.

    Yields
    ------
    tuple[str, pathlib.Path]
        Registry kind label and absolute path to a ``*.yaml`` file.
    """
    root = _registry_root()
    if not root.is_dir():
        return
    if kind is not None:
        subdirs = [root / kind] if (root / kind).is_dir() else []
    else:
        subdirs = [p for p in root.iterdir() if p.is_dir()]
    for sub in sorted(subdirs, key=lambda p: p.name):
        label = sub.name
        for path in sorted(sub.glob("*.yaml")):
            yield label, path.resolve()


def _cmd_list(args: argparse.Namespace) -> int:
    """Print bundled registry YAML entries (one per line: ``kind relative_path``).

    Returns
    -------
    int
        ``0`` on success, ``1`` when the registry tree is missing or empty.
    """
    root = _registry_root()
    if not root.is_dir():
        print(f"No registry directory at {root}", file=sys.stderr)
        return 1
    empty = True
    for reg_kind, path in _iter_registry_yaml_files(kind=args.kind):
        rel = path.relative_to(root)
        print(f"{reg_kind}\t{rel.as_posix()}")
        empty = False
    if empty:
        print(
            "No YAML files found"
            + (f" for kind {args.kind!r}" if args.kind else "")
            + f" under {root}",
            file=sys.stderr,
        )
        return 1
    return 0


def _load_yaml_document(path: Path) -> object:
    """Load a single YAML document from ``path``.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    object
        Parsed YAML structure (``yaml.safe_load``).
    """
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _resolve_describe_yaml_path(args: argparse.Namespace, root: Path) -> Path | None:
    """Resolve the YAML path for ``describe``.

    Returns
    -------
    pathlib.Path or None
        Resolved file path, or ``None`` when required arguments are missing.
    """
    if args.yaml is not None:
        return Path(args.yaml).expanduser().resolve()
    if args.kind is None or args.name is None:
        return None
    sub = root / args.kind
    direct = (sub / args.name).resolve()
    if direct.is_file():
        return direct
    with_suffix = (sub / f"{args.name}.yaml").resolve()
    return with_suffix


def _cmd_describe(args: argparse.Namespace) -> int:
    """Print a YAML registry document in JSON for readability.

    Returns
    -------
    int
        ``0`` on success, ``1`` for missing/invalid files, ``2`` for bad CLI usage.
    """
    root = _registry_root()
    path = _resolve_describe_yaml_path(args, root)
    if path is None:
        print(
            "Either --yaml PATH or both KIND and NAME positional arguments "
            "are required.",
            file=sys.stderr,
        )
        return 2
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1
    try:
        doc = _load_yaml_document(path)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"Invalid YAML ({path}): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


_DEFAULT_LIMIT: Final[int] = sys.maxsize


def _resolve_predict_url(args: argparse.Namespace) -> None:
    """Populate ``args.predict_url`` from ``--predict-url-file`` when set.

    Reads the first non-empty line of the file and strips whitespace.  The
    ``--predict-url`` flag takes precedence: if both are given, the explicit
    URL wins and the file is ignored.

    Parameters
    ----------
    args
        Parsed ``run`` subcommand namespace; mutated in-place.

    Raises
    ------
    SystemExit
        If the file does not exist or is empty.
    """
    if args.predict_url:
        return
    url_file = getattr(args, "predict_url_file", None)
    if url_file is None:
        return
    path = Path(url_file).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"--predict-url-file not found: {path}")
    url = path.read_text(encoding="utf-8").strip()
    if not url:
        raise SystemExit(f"--predict-url-file is empty: {path}")
    args.predict_url = url


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a benchmark run via the runner package.

    Returns
    -------
    int
        ``0`` when the runner completes without raising ``SystemExit``.

    Raises
    ------
    SystemExit
        If ``--limit`` and ``--sample-fraction`` are both set.
    """
    if args.limit is not None and args.sample_fraction is not None:
        raise SystemExit("--limit and --sample-fraction cannot be used together.")
    _resolve_predict_url(args)
    from beans_next.runner.runner import run_from_cli_namespace

    run_from_cli_namespace(args)
    return 0


def _cmd_score_from_file(args: argparse.Namespace) -> int:
    """Rescore an existing ``predictions.jsonl`` file on CPU.

    Returns
    -------
    int
        ``0`` on success.

    Raises
    ------
    SystemExit
        If the predictions file is missing, empty, or lacks scoring targets.
    """
    from beans_next.runner.rescorer import rescore_predictions_file

    predictions_path = Path(args.predictions_jsonl).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    task_type: str | None = getattr(args, "task_type", None) or None
    try:
        rescore_predictions_file(
            predictions_path,
            output_dir=out_dir,
            task_type=task_type,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``beans-next`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``run``, ``list``, and ``describe`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="beans-next",
        description="BEANS-Next: HTTP-first bioacoustics audio-LM benchmark CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a benchmark through the HF evaluator.")
    p_run.add_argument(
        "--predict-url",
        default=None,
        help="Full URL to the launcher POST /predict endpoint.",
    )
    p_run.add_argument(
        "--predict-url-file",
        default=None,
        metavar="PATH",
        help=(
            "Read the predict URL from a file (e.g. written by a SLURM serving job). "
            "Ignored when --predict-url is also given."
        ),
    )
    p_run.add_argument(
        "--workers",
        type=_workers_arg,
        default=1,
        metavar="N",
        help=(
            "Number of CPU-side worker threads/processes used by the runner "
            "(when supported)."
        ),
    )
    resume_group = p_run.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a run from the output directory checkpoint when supported.",
    )
    resume_group.add_argument(
        "--resume-from",
        dest="resume_from",
        type=Path,
        default=None,
        metavar="OUTPUT_DIR",
        help=(
            "Resume from an existing run output directory (contains checkpoint.json) "
            "when supported."
        ),
    )
    p_run.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where run artifacts (JSONL, summary) are written.",
    )
    p_run.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Optional directory for SQLite inference + scoring caches. "
            "Omit for default uncached behavior."
        ),
    )
    p_run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of dataset examples to score.",
    )
    p_run.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "Deterministically select this fraction of each task after loading. "
            "Cannot be combined with --limit."
        ),
    )
    p_run.add_argument(
        "--exclude-sample-id",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "Exclude an exact dataset sample id before inference. Repeat the option "
            "to exclude multiple documented model-incompatible samples."
        ),
    )
    p_run.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic per-task sampling (default 0).",
    )
    p_run.add_argument(
        "--stratify-by-label",
        action="store_true",
        default=False,
        help=(
            "Stratify --sample-fraction by the original reference label "
            "(diagnostic use only)."
        ),
    )
    p_run.add_argument(
        "--suite",
        default=None,
        help="Optional suite id from the eval registry (when registry content exists).",
    )
    p_run.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a run configuration YAML.",
    )
    p_run.add_argument(
        "--hf-path",
        default="EarthSpeciesProject/BEANS-Zero",
        help=("HuggingFace dataset id for the built-in runner."),
    )
    p_run.add_argument(
        "--hf-config",
        default="BEANS-Zero",
        help=(
            "HuggingFace builder configuration name (default BEANS-Zero for the "
            "EarthSpeciesProject/BEANS-Zero dataset). Pass an empty string for "
            "single-config datasets."
        ),
    )
    p_run.add_argument(
        "--hf-revision",
        default=None,
        help="Optional exact Hugging Face dataset revision.",
    )
    p_run.add_argument(
        "--split",
        default="test",
        help="Dataset split name for HF loading (default test).",
    )
    p_run.add_argument(
        "--dataset-name",
        default="esc50",
        metavar="NAME",
        help=(
            "When set, keep only rows whose dataset_name column equals this "
            "value (default esc50 for the built-in BEANS-Zero slice)."
        ),
    )
    p_run.add_argument(
        "--task-id",
        default=None,
        help="Optional task id recorded on each DatasetExample.",
    )
    p_run.add_argument(
        "--run-id",
        default=None,
        help="Run directory name and RunSummary.run_id (default beans-next-cli).",
    )
    p_run.add_argument(
        "--prompt-yaml",
        default=None,
        metavar="PATH",
        help="Prompt spec YAML (default bundled classification_bioacoustic_v1).",
    )
    p_run.add_argument(
        "--modality-mode",
        choices=("audio", "gaussian-noise", "text-only", "text-only-informed"),
        default="audio",
        help=(
            "Input modality mode. Gaussian-noise preserves the audio pathway but "
            "replaces every slot with deterministic noise. Text-only modes remove "
            "audio placeholders and send no audio."
        ),
    )
    p_run.add_argument(
        "--gaussian-noise-cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Shared deterministic noise cache. Defaults to "
            "~/.cache/beans-next/gaussian-noise."
        ),
    )
    p_run.add_argument(
        "--gaussian-noise-seed",
        type=int,
        default=0,
        help="Recorded global Gaussian-noise seed (default 0).",
    )
    p_run.add_argument(
        "--gaussian-noise-rms-dbfs",
        type=float,
        default=-20.0,
        help="Gaussian-noise RMS in dBFS (full protocol default -20).",
    )
    p_run.add_argument(
        "--gaussian-noise-protocol-version",
        default="beans-next.gaussian-noise.v1",
        help="Version string included in deterministic noise seeds and manifests.",
    )
    p_run.add_argument(
        "--preserve-file-paths",
        action="store_true",
        default=False,
        help=(
            "Send file_path audio payloads without base64 conversion. Use only "
            "when the runner and model server can read the same filesystem paths."
        ),
    )
    p_run.set_defaults(_handler=_cmd_run)

    p_list = sub.add_parser(
        "list",
        help="List bundled registry YAML files (prompts, future dataset/suite ids).",
    )
    p_list.add_argument(
        "--kind",
        default=None,
        metavar="KIND",
        help="Restrict to one registry subdirectory (for example 'prompt').",
    )
    p_list.set_defaults(_handler=_cmd_list)

    p_desc = sub.add_parser(
        "describe",
        help="Show a registry YAML document as formatted JSON.",
    )
    p_desc.add_argument(
        "kind",
        nargs="?",
        default=None,
        help="Registry subdirectory (for example 'prompt').",
    )
    p_desc.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Stem or file name under that subdirectory (for example "
        "'classification_bioacoustic_v1').",
    )
    p_desc.add_argument(
        "--yaml",
        dest="yaml",
        metavar="PATH",
        default=None,
        help="Describe this YAML path directly (bypasses KIND/NAME lookup).",
    )
    p_desc.set_defaults(_handler=_cmd_describe)

    p_score = sub.add_parser(
        "score-from-file",
        help=(
            "Rescore an existing predictions.jsonl by running post-process + metrics "
            "and writing scored artifacts (CPU-only)."
        ),
    )
    p_score.add_argument(
        "predictions_jsonl",
        metavar="PREDICTIONS_JSONL",
        help="Path to a predictions.jsonl file produced by beans-next run.",
    )
    p_score.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write artifacts into (default: predictions file directory).",
    )
    p_score.add_argument(
        "--task-type",
        default=None,
        metavar="TYPE",
        help=(
            "Task type for post-processing and scoring "
            "(e.g. classification, detection, captioning)."
        ),
    )
    p_score.set_defaults(_handler=_cmd_score_from_file)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the ``beans-next`` console script.

    Parameters
    ----------
    argv
        Arguments excluding the program name (like ``sys.argv[1:]``). Uses
        ``sys.argv`` when ``None``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = args._handler
    exit_code = handler(args)
    if os.environ.get("BEANS_NEXT_HARD_EXIT") == "1":
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
