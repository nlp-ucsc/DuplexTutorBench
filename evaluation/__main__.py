"""Unified CLI for the evaluation module.

Four explicit subcommands:

    uv run python -m evaluation run --run-name <name> [...]
    uv run python -m evaluation report --run-name <name> [...]
    uv run python -m evaluation leaderboard [--judge {bea,naive}] [...]
    uv run python -m evaluation import-mtb [--hard] [--limit N] [...]

`run` scores a duplex run; `report` re-renders per-dialog text reports from
existing scores; `leaderboard` aggregates across runs; `import-mtb` converts
MathTutorBench dialogues into the duplex conversation schema so the same
judges can score them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evaluation import leaderboard as _leaderboard_mod
from evaluation import mathtutorbench as _mtb_mod
from evaluation.runner import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_EVALUATORS = "stats,turn_taking,llm_judge"
ALL_EVALUATOR_NAMES = (
    "stats",
    "turn_taking",
    "naturalness",
    "llm_judge",
    "llm_judge_bea",
)


def _add_run_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "run",
        help="Score one duplex run with the named evaluators.",
        description="Evaluate duplex_output/{run_name}/ conversations.",
    )
    p.add_argument("--run-name", required=True, help="Duplex run to evaluate")
    p.add_argument(
        "--evaluators",
        default=DEFAULT_EVALUATORS,
        help=(
            "Comma-separated evaluator names. Choices: "
            + ", ".join(ALL_EVALUATOR_NAMES)
            + f" (default: {DEFAULT_EVALUATORS})"
        ),
    )
    p.add_argument(
        "--conv-index",
        type=int,
        default=None,
        help="If set, evaluate only this single conversation index.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a result for this evaluator already exists.",
    )
    p.add_argument(
        "--judge-provider",
        default="openai",
        choices=["openai", "gemini", "claude"],
        help="LLM judge provider (default: openai)",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Judge model id (defaults per provider: gpt-5-mini / "
            "gemini-2.5-flash / claude-sonnet-4-6)"
        ),
    )
    p.add_argument("--duplex-root", default="duplex_output")
    p.add_argument("--eval-root", default="eval_output")
    p.add_argument(
        "--align-variant",
        default=None,
        help=(
            "Score an aligned-transcript variant produced by the alignment "
            "module (e.g. whisper_mms, whisper_mms_no_bias, whisper_mfa, "
            "whisper_mfa_no_bias, text_mms). Reads from "
            "{duplex-root}/{run_name}/aligned/<variant>.jsonl and writes to "
            "{eval-root}/{run_name}__<variant>/scores.jsonl so original and "
            "each variant accumulate side-by-side."
        ),
    )
    p.add_argument(
        "--eval-subdir",
        default=None,
        help=(
            "Explicit output subdir under {eval-root} (overrides the default "
            "{run_name} / {run_name}__<variant>). Use to keep per-judge-model "
            "results separate, e.g. {run_name}__judge-claude-sonnet-4-6."
        ),
    )
    p.add_argument(
        "--report",
        action="store_true",
        help=(
            "After scoring, write per-dialog text reports to "
            "{eval-root}/{run_name}/reports/conv_NNN.txt and a summary.txt."
        ),
    )
    p.set_defaults(_cmd_fn=_cmd_run, _cmd_parser=p)


def _add_report_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "report",
        help="Render per-dialog text reports from existing scores.jsonl.",
        description=(
            "Re-render per-dialog text reports without re-scoring. Useful "
            "after editing the report formatter."
        ),
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--align-variant", default=None)
    p.add_argument("--eval-subdir", default=None)
    p.add_argument("--conv-index", type=int, default=None)
    p.add_argument("--duplex-root", default="duplex_output")
    p.add_argument("--eval-root", default="eval_output")
    p.set_defaults(_cmd_fn=_cmd_report)


def _cmd_run(args: argparse.Namespace) -> None:
    names = [n.strip() for n in args.evaluators.split(",") if n.strip()]
    unknown = [n for n in names if n not in ALL_EVALUATOR_NAMES]
    if unknown:
        args._cmd_parser.error(
            f"Unknown evaluator(s): {unknown}. "
            f"Known: {', '.join(ALL_EVALUATOR_NAMES)}"
        )

    duplex_root = Path(args.duplex_root)
    eval_root = Path(args.eval_root)

    scores_path = run_evaluation(
        run_name=args.run_name,
        evaluator_names=names,
        duplex_root=duplex_root,
        eval_root=eval_root,
        conv_index=args.conv_index,
        force=args.force,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        align_variant=args.align_variant,
        eval_subdir=args.eval_subdir,
    )
    logger.info("Done. Scores at %s", scores_path)

    if args.report:
        _write_reports(args, scores_path=scores_path)


def _cmd_report(args: argparse.Namespace) -> None:
    duplex_root = Path(args.duplex_root)
    eval_root = Path(args.eval_root)
    eval_subdir = args.eval_subdir or (
        f"{args.run_name}__{args.align_variant}"
        if args.align_variant
        else args.run_name
    )
    scores_path = eval_root / eval_subdir / "scores.jsonl"
    if not scores_path.is_file():
        raise SystemExit(
            f"No scores at {scores_path}. Run `python -m evaluation run --run-name "
            f"{args.run_name}` first."
        )
    _write_reports(
        args, scores_path=scores_path, _eval_root=eval_root, _duplex_root=duplex_root
    )


def _write_reports(
    args: argparse.Namespace,
    *,
    scores_path: Path,
    _eval_root: Path | None = None,
    _duplex_root: Path | None = None,
) -> None:
    from evaluation.report import write_reports

    duplex_root = _duplex_root or Path(args.duplex_root)
    eval_root = _eval_root or Path(args.eval_root)
    src_name = (
        f"aligned/{args.align_variant}.jsonl"
        if args.align_variant
        else "conversations.jsonl"
    )
    conversations_path = duplex_root / args.run_name / src_name
    eval_subdir = args.eval_subdir or (
        f"{args.run_name}__{args.align_variant}"
        if args.align_variant
        else args.run_name
    )
    reports_dir = eval_root / eval_subdir / "reports"
    write_reports(
        scores_path,
        conversations_path,
        reports_dir,
        conv_index=args.conv_index,
    )
    logger.info("Reports at %s", reports_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Evaluation CLI: score runs, render reports, build leaderboards, import datasets.",
    )
    subparsers = p.add_subparsers(dest="cmd", required=True)
    _add_run_subparser(subparsers)
    _add_report_subparser(subparsers)
    _leaderboard_mod.add_subparser(subparsers)
    _mtb_mod.add_subparser(subparsers)
    args = p.parse_args()
    args._cmd_fn(args)


if __name__ == "__main__":
    main()
