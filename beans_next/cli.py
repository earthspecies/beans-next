"""Command-line interface for BEANS-Next (``beans-next``).

Subcommands dispatch to the benchmark runner (increment I3-B) and to bundled
registry assets (prompt YAMLs under ``beans_next/registry``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import yaml

from beans_next.api.types import DatasetExample
from beans_next.post_process.pipeline import StepSpec

_RUN_HOOK_NAMES: Final[tuple[str, ...]] = (
    "run_from_cli_namespace",
    "main_run_from_cli",
    "cli_run",
)


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


def _effective_run_limit(args: argparse.Namespace) -> int:
    """Return the sample cap for a CLI run.

    Returns
    -------
    int
        At least ``1``; defaults to unlimited (``sys.maxsize``) when ``--limit`` is
        omitted.
    """
    if args.limit is not None:
        return max(1, int(args.limit))
    return _DEFAULT_LIMIT


def _legacy_builtin_task_type(args: argparse.Namespace) -> str | None:
    """Best-effort task type for the legacy ``beans-next run`` path (no suite YAML).

    Normal runs use :func:`~beans_next.runner.runner.run_from_cli_namespace`, which
    reads ``task_type`` from eval-task YAML. This hook only applies when the CLI
    falls back to :func:`_run_benchmark_cli_builtin`.

    Returns
    -------
    str or None
        ``\"captioning\"`` when ``--dataset-name`` is the BEANS-Zero captioning
        subset so post-processing does not comma-split or fuzzy-match prose.
    """
    dn = str(getattr(args, "dataset_name", "") or "").strip().lower()
    if dn == "captioning":
        return "captioning"
    return None


def _build_postprocess_tuples(
    examples: list[DatasetExample],
    *,
    task_type: str | None = None,
) -> tuple[tuple[StepSpec, ...], tuple[StepSpec, ...]]:
    """Build parser and cleaner :class:`~beans_next.post_process.pipeline.StepSpec` rows.

    Delegates to :func:`~beans_next.runner.runner._postprocess_steps_for_examples`
    so behavior matches suite/config runs (captioning and other free-text tasks
    skip label parsing).

    Parameters
    ----------
    examples
        Loaded dataset rows.
    task_type
        Optional eval-task type (e.g. ``\"captioning\"``, ``\"classification\"``).

    Returns
    -------
    tuple
        ``(parser_steps, cleaner_steps)`` for
        :class:`~beans_next.runner.runner.RunnerConfig`.
    """
    from beans_next.runner.runner import _postprocess_steps_for_examples

    return _postprocess_steps_for_examples(examples, task_type=task_type)


def _load_examples_for_run(args: argparse.Namespace) -> list[DatasetExample]:
    """Load HuggingFace rows as :class:`~beans_next.api.types.DatasetExample`.

    Stops after the effective ``--limit`` (see :func:`_effective_run_limit`).

    Returns
    -------
    list of DatasetExample
        Up to ``limit`` normalized rows.
    """
    from beans_next.datasets import (
        dataset_name_equals,
        iter_esp_data_beans_zero_examples,
        iter_hf_beans_next_examples,
        iter_hf_dataset_examples,
    )

    limit = _effective_run_limit(args)
    row_filter = (
        dataset_name_equals(args.dataset_name) if args.dataset_name else None
    )
    hf_config = args.hf_config if args.hf_config else None
    rows: list[DatasetExample] = []
    data_source = getattr(args, "data_source", None)
    if data_source == "esp_data":
        # esp_data path is BEANS-Zero specific: we use subset via dataset_name.
        # Row filtering is unnecessary because esp_data already yields by subset.
        for ex in iter_esp_data_beans_zero_examples(
            subset=str(args.dataset_name),
            split=str(args.split),
            task_id=args.task_id,
            limit=limit,
        ):
            rows.append(ex)
            if len(rows) >= limit:
                break
    elif data_source == "huggingface":
        from beans_next.datasets.beans_next_hub import BEANS_NEXT_HUB_REPO_ID

        hf_path_arg = (getattr(args, "hf_path", None) or "").strip()
        repo_id = hf_path_arg or BEANS_NEXT_HUB_REPO_ID
        subset_name = str(args.dataset_name).strip()
        for ex in iter_hf_beans_next_examples(
            repo_id,
            subset=subset_name,
            split=str(args.split),
            task_id=args.task_id,
            limit=limit,
        ):
            rows.append(ex)
            if len(rows) >= limit:
                break
    else:
        for ex in iter_hf_dataset_examples(
            args.hf_path,
            split=args.split,
            config_name=hf_config,
            task_id=args.task_id,
            row_filter=row_filter,
        ):
            rows.append(ex)
            if len(rows) >= limit:
                break
    return rows


def _default_output_dir(args: argparse.Namespace, run_id: str) -> Path:
    """Resolve the artifact output directory for ``run``.

    Returns
    -------
    pathlib.Path
        Absolute output path (``./results/<run_id>`` when ``-o`` is omitted).
    """
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    return (Path.cwd() / "results" / run_id).resolve()


def _prompt_path_from_args(args: argparse.Namespace) -> Path:
    """Resolve the prompt YAML path for ``run``.

    Returns
    -------
    pathlib.Path
        Absolute path to the prompt specification file.
    """
    from beans_next.prompts.renderer import builtin_prompt_registry_path

    if args.prompt_yaml is not None:
        return Path(args.prompt_yaml).expanduser().resolve()
    return (
        builtin_prompt_registry_path() / "classification_bioacoustic_v1.yaml"
    ).resolve()


def _validate_builtin_run_args(args: argparse.Namespace) -> None:
    """Validate flags for the built-in ``beans-next run`` path.

    Raises
    ------
    SystemExit
        When ``--predict-url`` is missing for non-config runs.
    """
    if args.config is None and not args.predict_url:
        raise SystemExit(
            "--predict-url is required for beans-next run (unless --config is set)."
        )


def _run_benchmark_cli_builtin(args: argparse.Namespace) -> None:
    """Execute the default HuggingFace-backed ``beans-next run`` path.

    Wires :class:`~beans_next.runner.runner.BenchmarkRunner` with a small batch
    and the bundled classification prompt unless ``--prompt-yaml`` is set.

    Raises
    ------
    SystemExit
        On invalid CLI combinations, empty example lists, or missing optional
        dependencies such as the ``datasets`` package.
    """
    from beans_next.models.http import HttpClient
    from beans_next.prompts.renderer import PromptRenderer, load_prompt_spec_from_path
    from beans_next.runner.runner import BenchmarkRunner, RunnerConfig

    _validate_builtin_run_args(args)
    if args.suite is not None:
        print(
            "warning: --suite is not yet wired to suite YAML; "
            "using HF path/config instead.",
            file=sys.stderr,
        )
    run_id = (args.run_id or "beans-next-cli").strip() or "beans-next-cli"
    try:
        examples = _load_examples_for_run(args)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    if not examples:
        raise SystemExit("No dataset examples were loaded; check HF parameters.")
    parsers, cleaners = _build_postprocess_tuples(
        examples,
        task_type=_legacy_builtin_task_type(args),
    )
    out_dir = _default_output_dir(args, run_id)
    spec = load_prompt_spec_from_path(_prompt_path_from_args(args))
    renderer = PromptRenderer(spec)
    cfg = RunnerConfig(
        output_dir=out_dir,
        run_id=run_id,
        parser_steps=parsers,
        cleaner_steps=cleaners,
    )
    with HttpClient(args.predict_url, probe_on_init=True) as client:
        runner = BenchmarkRunner(client, renderer, cfg)
        runner.run(examples)


def _dispatch_run(args: argparse.Namespace) -> None:
    """Forward ``beans-next run`` to ``beans_next.runner.runner``.

    If the runner module defines one of ``run_from_cli_namespace``,
    ``main_run_from_cli``, or ``cli_run``, that hook receives the
    :class:`argparse.Namespace` and owns execution. Otherwise this CLI builds a
    minimal :class:`~beans_next.runner.runner.BenchmarkRunner` from the
    namespace (HuggingFace slice + bundled prompt + HTTP endpoint).

    Parameters
    ----------
    args
        Parsed ``run`` subcommand arguments.

    Raises
    ------
    SystemExit
        If the runner module cannot be imported.
    """
    try:
        mod = importlib.import_module("beans_next.runner.runner")
    except ImportError as exc:
        msg = (
            "Benchmark runner is unavailable: could not import "
            "'beans_next.runner.runner'. Merge increment I3-B (BenchmarkRunner) "
            "before using 'beans-next run'."
        )
        raise SystemExit(msg) from exc
    for name in _RUN_HOOK_NAMES:
        fn = getattr(mod, name, None)
        if callable(fn):
            fn(args)
            return
    _run_benchmark_cli_builtin(args)


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
    """
    _resolve_predict_url(args)
    _dispatch_run(args)
    return 0


def _cmd_setup_spice(args: argparse.Namespace) -> int:
    """Download Stanford CoreNLP JARs required by SPICE.

    Returns
    -------
    int
        ``0`` on success.

    Raises
    ------
    SystemExit
        If the download fails.
    """
    from beans_next.metrics._spice._download import download_stanford_models

    force = getattr(args, "force", False)
    print("Downloading Stanford CoreNLP 3.6.0 JARs for SPICE …")
    try:
        download_stanford_models(force=force)
    except Exception as exc:
        raise SystemExit(f"Download failed: {exc}") from exc
    print("Done. SPICE is ready.")
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
    judge_url: str | None = getattr(args, "judge_url", None) or None
    judge_extract_url: str | None = getattr(args, "judge_extract_url", None) or None
    task_type: str | None = getattr(args, "task_type", None) or None
    try:
        rescore_predictions_file(
            predictions_path,
            output_dir=out_dir,
            task_type=task_type,
            judge_url=judge_url,
            judge_extract_url=judge_extract_url,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def _cmd_pairs(args: argparse.Namespace) -> int:
    """Generate prompt/answer/ground-truth pairs for BeansPro subsets.

    Returns
    -------
    int
        ``0`` on success.
    """
    from beans_next.scripts.generate_pairs import generate_pairs_main

    return int(generate_pairs_main(args))


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

    p_run = sub.add_parser("run", help="Run a benchmark via BenchmarkRunner (I3-B).")
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
            "Optional directory for SQLite inference + scoring caches (I6-A). "
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
        "--backend",
        dest="data_source",
        default=None,
        choices=("esp_data", "huggingface", "hf"),
        help=(
            "Dataset backend. `esp_data` (default) loads BEANS-Next audio from GCS "
            "via the esp_data library. `huggingface` loads from the two-table Parquet "
            "bundle on the HuggingFace Hub (no private credentials needed). "
            "`hf` is the legacy streaming backend for other HF datasets."
        ),
    )
    p_run.add_argument(
        "--hf-path",
        default="EarthSpeciesProject/BEANS-Zero",
        help=(
            "HuggingFace dataset id for the built-in runner "
            "(ignored by custom hooks)."
        ),
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
        "--judge-url",
        default=None,
        metavar="URL",
        help=(
            "URL for the judge POST endpoint (enables LLM-as-judge scoring). "
            "When set, judge_outputs.jsonl is written after the run."
        ),
    )
    p_run.add_argument(
        "--upload-gcs",
        action="store_true",
        default=False,
        help=(
            "Upload run artifacts to GCS after the run completes. "
            "Uses --gcs-prefix (default foundation-model-data bucket). "
            "Requires google-cloud-storage to be installed."
        ),
    )
    p_run.add_argument(
        "--gcs-prefix",
        default="gs://foundation-model-data/synthetic/predictions",
        metavar="GCS_PREFIX",
        help=(
            "Base GCS prefix for artifact uploads (used with --upload-gcs). "
            "The run id is appended automatically. "
            "Default: gs://foundation-model-data/synthetic/predictions"
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
        "--judge-url",
        default=None,
        metavar="URL",
        help=(
            "URL for the YES/NO judge model POST /predict endpoint. When set, "
            "judge_scored_predictions.jsonl, judge_summary.json, and "
            "judge_outputs.jsonl are written alongside normal rescorer artifacts."
        ),
    )
    p_score.add_argument(
        "--judge-extract-url",
        default=None,
        metavar="URL",
        help=(
            "URL for the extractor judge model POST /predict endpoint. When set, "
            "the judge converts each raw prediction into a structured prediction "
            "using task-specific templates, then scores with the normal pipeline. "
            "Writes judge_extracted_scored_predictions.jsonl and "
            "judge_extracted_summary.json. Use --task-type to select the right "
            "extraction template (classification / detection / captioning)."
        ),
    )
    p_score.add_argument(
        "--task-type",
        default=None,
        metavar="TYPE",
        help=(
            "Task type for post-processing and judge extraction template selection "
            "(e.g. classification, detection, captioning)."
        ),
    )
    p_score.set_defaults(_handler=_cmd_score_from_file)

    p_pairs = sub.add_parser(
        "pairs",
        help=(
            "Generate prompt/answer/ground-truth pairs for BeansPro subsets "
            "(stores raw + post-processed predictions)."
        ),
    )
    from beans_next.scripts.generate_pairs import build_pairs_arg_parser

    build_pairs_arg_parser(p_pairs)
    p_pairs.set_defaults(_handler=_cmd_pairs)

    p_setup_spice = sub.add_parser(
        "setup-spice",
        help=(
            "Download Stanford CoreNLP 3.6.0 JARs required by the SPICE metric. "
            "JARs are cached in ~/.cache/beans-next/spice/lib/."
        ),
    )
    p_setup_spice.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if the JARs are already present.",
    )
    p_setup_spice.set_defaults(_handler=_cmd_setup_spice)

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
    if os.environ.get("BEANS_PRO_HARD_EXIT") == "1":
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
