"""Batch-score all (or selected) duplex runs, then print the leaderboard.

Usage:
    # Score every run under duplex_output/ and print leaderboard
    uv run python evaluation/scripts/eval_runs.py

    # Score specific runs only
    uv run python evaluation/scripts/eval_runs.py --runs gemini_gpt_v1 gemini_gemini_v1

    # Skip scoring (leaderboard only, using already-scored data)
    uv run python evaluation/scripts/eval_runs.py --skip-scoring

    # Score + write per-dialog text reports + leaderboard
    uv run python evaluation/scripts/eval_runs.py --report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-score duplex runs + leaderboard.")
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run names to score (default: all dirs under --duplex-root)",
    )
    p.add_argument("--duplex-root", default="duplex_output")
    p.add_argument("--eval-root", default="eval_output")
    p.add_argument(
        "--evaluators",
        default="stats,turn_taking,llm_judge",
        help="Comma-separated evaluator names",
    )
    p.add_argument("--judge-model", default=None)
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute already-cached evaluator results",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Write per-dialog text reports after scoring each run",
    )
    p.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip evaluation; go straight to leaderboard using existing scores",
    )
    p.add_argument(
        "--sort-by",
        default=None,
        help="Rubric key to sort leaderboard by (default: 'overall' for naive judge, 'guidance_quality' for BEA)",
    )
    p.add_argument(
        "--leaderboard-out",
        default=None,
        help="Path for leaderboard output (default: eval_output/leaderboard.txt)",
    )
    return p.parse_args()


def discover_runs(duplex_root: Path) -> list[str]:
    return [
        d.name
        for d in sorted(duplex_root.iterdir())
        if d.is_dir() and (d / "conversations.jsonl").is_file()
    ]


def main() -> None:
    args = parse_args()
    duplex_root = Path(args.duplex_root)
    eval_root = Path(args.eval_root)
    evaluator_names = [n.strip() for n in args.evaluators.split(",") if n.strip()]

    runs = args.runs or discover_runs(duplex_root)
    if not runs:
        raise SystemExit(f"No runs found under {duplex_root}")

    logger.info("Runs to process: %s", runs)

    if not args.skip_scoring:
        from evaluation.report import write_reports
        from evaluation.runner import run_evaluation

        for run_name in runs:
            logger.info("=== Scoring: %s ===", run_name)
            try:
                scores_path = run_evaluation(
                    run_name=run_name,
                    evaluator_names=evaluator_names,
                    duplex_root=duplex_root,
                    eval_root=eval_root,
                    force=args.force,
                    judge_model=args.judge_model,
                )
            except FileNotFoundError as e:
                logger.warning("Skipping %s: %s", run_name, e)
                continue

            if args.report:
                convs_path = duplex_root / run_name / "conversations.jsonl"
                reports_dir = eval_root / run_name / "reports"
                write_reports(scores_path, convs_path, reports_dir)
                logger.info("Reports: %s", reports_dir)

    # Leaderboard across all scored runs.
    # Infer judge mode from whichever judge evaluator was run.
    from evaluation.leaderboard import run_leaderboard

    judge_mode = "bea" if "llm_judge_bea" in evaluator_names else "naive"
    sort_by = args.sort_by or ("guidance_quality" if judge_mode == "bea" else "overall")
    lb_out = Path(args.leaderboard_out) if args.leaderboard_out else None
    try:
        out_path = run_leaderboard(
            run_names=None,  # auto-discover from eval_root
            duplex_root=duplex_root,
            eval_root=eval_root,
            out=lb_out,
            sort_by=sort_by,
            judge_mode=judge_mode,
        )
        print("\n" + out_path.read_text())
    except FileNotFoundError as e:
        logger.warning("Leaderboard skipped: %s", e)


if __name__ == "__main__":
    main()
