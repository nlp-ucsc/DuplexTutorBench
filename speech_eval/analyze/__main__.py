"""CLI for the speech_eval cross-system analysis module.

Usage:
    uv run python -m speech_eval.analyze [--runs r1 r2 ...] [--group NAME]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from speech_eval.analyze.loader import discover_runs, load_all
from speech_eval.analyze.report import build

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m speech_eval.analyze",
        description=(
            "Cross-system comparison report over speech_eval JSONL outputs. "
            "Writes summary.csv, system_table.md, report.md, figures/, and "
            "pairwise/ under <out-root>/<group>/."
        ),
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help=(
            "Runs to compare. Default: every non-underscore-prefixed subdir of "
            "--input-root."
        ),
    )
    p.add_argument(
        "--group",
        default=None,
        help=(
            "Subdir name under --out-root for this report. Default: "
            "YYYY-MM-DD_<N>runs."
        ),
    )
    p.add_argument(
        "--reference",
        default=None,
        help=(
            "Human-reference run (e.g. 'maptask_ref') whose vap.jsonl timing "
            "distributions anchor the distributional-realism distances (KS / "
            "EMD / JS, report §14). It is NOT included in --runs and needs only "
            "a vap.jsonl (no audiobox/judge). Omit to skip §14."
        ),
    )
    p.add_argument(
        "--input-root",
        default="speech_eval_output",
        help="Default: speech_eval_output",
    )
    p.add_argument(
        "--out-root",
        default="speech_eval_output/_analysis",
        help="Default: speech_eval_output/_analysis",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)

    runs = args.runs or discover_runs(input_root)
    if not runs:
        raise SystemExit(f"No runs found under {input_root}")
    logger.info("analyze: %d runs: %s", len(runs), ", ".join(runs))

    group = args.group or (
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_{len(runs)}runs"
    )

    data = load_all(input_root, runs)
    logger.info(
        "analyze: loaded audiobox=%d rows, judge=%d rows, vap=%d rows",
        len(data.audiobox),
        len(data.judge),
        len(data.vap),
    )

    group_dir = build(data, out_root=out_root, group=group, reference=args.reference)
    print(f"speech_eval analysis: {group_dir}")


if __name__ == "__main__":
    main()
