"""Run the BEA judge with multiple judge models and compare their scores.

Each (provider, model) judge scores into its own
eval_output/{run}__judge-{model}/ dir so results never overwrite each other,
then a side-by-side table shows how much the judge models agree per run.

Usage:
    # Default 3 judges over one run
    uv run python evaluation/scripts/judge_sweep.py --runs gemini_gemini_v1

    # All runs under duplex_output/
    uv run python evaluation/scripts/judge_sweep.py

    # Custom judges (provider:model), force recompute
    uv run python evaluation/scripts/judge_sweep.py --judges openai:gpt-5-mini claude:claude-sonnet-4-6 --force
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

from evaluation.evaluators.llm_judge_bea import RUBRIC_KEYS_BEA
from evaluation.leaderboard import _load_jsonl
from evaluation.runner import run_evaluation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# (provider, model) judges run by default.
DEFAULT_JUDGES = [
    ("openai", "gpt-5-mini"),
    ("gemini", "gemini-2.5-flash"),
    ("claude", "claude-sonnet-4-6"),
]


def _slug(model: str) -> str:
    return model.replace("/", "-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model judge sweep + comparison.")
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run names (default: all dirs under --duplex-root)",
    )
    p.add_argument(
        "--judges",
        nargs="*",
        default=None,
        help="provider:model pairs (default: openai:gpt-5-mini gemini:gemini-2.5-flash claude:claude-sonnet-4-6)",
    )
    p.add_argument(
        "--evaluator", default="llm_judge_bea", choices=["llm_judge", "llm_judge_bea"]
    )
    p.add_argument("--duplex-root", default="duplex_output")
    p.add_argument("--eval-root", default="eval_output")
    p.add_argument("--force", action="store_true", help="Recompute cached results")
    p.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring; just print the comparison from existing dirs",
    )
    return p.parse_args()


def _judges_from_args(judge_args: list[str] | None) -> list[tuple[str, str]]:
    if not judge_args:
        return DEFAULT_JUDGES
    out: list[tuple[str, str]] = []
    for spec in judge_args:
        if ":" not in spec:
            raise SystemExit(f"--judges entry {spec!r} must be provider:model")
        provider, model = spec.split(":", 1)
        out.append((provider, model))
    return out


def _discover_runs(duplex_root: Path) -> list[str]:
    return [
        d.name
        for d in sorted(duplex_root.iterdir())
        if d.is_dir() and (d / "conversations.jsonl").is_file()
    ]


def _rubric_keys(evaluator: str) -> tuple[str, ...]:
    if evaluator == "llm_judge_bea":
        return RUBRIC_KEYS_BEA
    return (
        "answer_correctness",
        "scaffolding_quality",
        "student_realism",
        "phrasing_naturalness",
        "overall",
    )


def _mean(vals: list[float]) -> float | None:
    return round(statistics.mean(vals), 2) if vals else None


def build_comparison(
    runs: list[str],
    judges: list[tuple[str, str]],
    evaluator: str,
    eval_root: Path,
) -> str:
    """Per run, mean component score + per-criterion means for each judge model."""
    rubric_keys = _rubric_keys(evaluator)
    lines: list[str] = []
    thick = "=" * 88
    sep = "-" * 88

    for run in runs:
        lines.append(thick)
        lines.append(f"RUN: {run}   (evaluator: {evaluator})")
        lines.append(thick)

        # Header: model columns.
        model_w = 22
        header = f"{'criterion':<24}" + "".join(
            f"{_slug(m):>{model_w}}" for _, m in judges
        )
        lines.append(header)
        lines.append(sep)

        # Collect per-judge score lists.
        per_judge: dict[str, dict[str, list[int]]] = {}
        per_judge_overall: dict[str, list[float]] = {}
        for _, model in judges:
            subdir = eval_root / f"{run}__judge-{_slug(model)}"
            scores_path = subdir / "scores.jsonl"
            crit: dict[str, list[int]] = {k: [] for k in rubric_keys}
            overall: list[float] = []
            if scores_path.is_file():
                for rec in _load_jsonl(scores_path):
                    judge = rec.get("evaluator_results", {}).get(evaluator, {})
                    if judge.get("status") != "ok":
                        continue
                    if isinstance(judge.get("mean_component_score"), (int, float)):
                        overall.append(judge["mean_component_score"])
                    for k in rubric_keys:
                        entry = judge.get("scores", {}).get(k)
                        if isinstance(entry, dict) and isinstance(
                            entry.get("score"), int
                        ):
                            crit[k].append(entry["score"])
            else:
                logger.warning("Missing %s — was it scored?", scores_path)
            per_judge[model] = crit
            per_judge_overall[model] = overall

        # Per-criterion rows.
        for k in rubric_keys:
            row = f"{k:<24}"
            for _, model in judges:
                mu = _mean(per_judge[model][k])
                row += f"{(f'{mu:.2f}' if mu is not None else '--'):>{model_w}}"
            lines.append(row)

        # Mean component score row.
        lines.append(sep)
        row = f"{'MEAN COMPONENT':<24}"
        for _, model in judges:
            mu = _mean(per_judge_overall[model])
            row += f"{(f'{mu:.2f}' if mu is not None else '--'):>{model_w}}"
        lines.append(row)

        # n row.
        row = f"{'n (ok convs)':<24}"
        for _, model in judges:
            row += f"{len(per_judge_overall[model]):>{model_w}}"
        lines.append(row)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    duplex_root = Path(args.duplex_root)
    eval_root = Path(args.eval_root)
    judges = _judges_from_args(args.judges)
    runs = args.runs or _discover_runs(duplex_root)
    if not runs:
        raise SystemExit(f"No runs found under {duplex_root}")

    logger.info("Runs: %s", runs)
    logger.info("Judges: %s", [f"{p}:{m}" for p, m in judges])

    if not args.skip_scoring:
        for run in runs:
            for provider, model in judges:
                subdir = f"{run}__judge-{_slug(model)}"
                logger.info(
                    "=== Scoring %s with %s:%s -> %s ===", run, provider, model, subdir
                )
                try:
                    run_evaluation(
                        run_name=run,
                        evaluator_names=[args.evaluator],
                        duplex_root=duplex_root,
                        eval_root=eval_root,
                        force=args.force,
                        judge_model=model,
                        judge_provider=provider,
                        eval_subdir=subdir,
                    )
                except FileNotFoundError as e:
                    logger.warning("Skipping %s: %s", run, e)

    text = build_comparison(runs, judges, args.evaluator, eval_root)
    out_path = eval_root / "judge_comparison.txt"
    out_path.write_text(text)
    logger.info("Comparison written to %s", out_path)
    print("\n" + text)


if __name__ == "__main__":
    main()
