"""Cross-run leaderboard: compare tutor backends on shared llm_judge scores.

Reads all scored runs under eval_output/ and produces a backend-vs-backend
comparison table. Supports two judge modes:
  --judge bea    BEA 2025 taxonomy (Maurya et al., NAACL 2025) — 8 dimensions (default)
  --judge naive  Original 5-dimension rubric for comparison

Usage:
    uv run python -m evaluation leaderboard                        # auto-discover, BEA judge
    uv run python -m evaluation leaderboard --judge naive          # naive rubric
    uv run python -m evaluation leaderboard --runs gemini_gpt_v1 gemini_gemini_v1
    uv run python -m evaluation leaderboard --out eval_output/leaderboard.txt
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

# Naive judge rubric (original 5 keys)
_RUBRIC_KEYS_NAIVE = (
    "answer_correctness",
    "scaffolding_quality",
    "student_realism",
    "phrasing_naturalness",
    "overall",
)

_RUBRIC_LABELS_NAIVE = {
    "answer_correctness": "Correctness",
    "scaffolding_quality": "Scaffolding",
    "student_realism": "Stu.realism",
    "phrasing_naturalness": "Naturalness",
    "overall": "Overall",
}

_AXIS_GROUPS_NAIVE = {
    "Naturalness": ("phrasing_naturalness", "student_realism"),
    "Effectiveness": ("answer_correctness", "scaffolding_quality"),
    "Engagement": ("overall",),
}

# BEA 2025 judge rubric (8 keys) — imported to avoid duplication
from evaluation.evaluators.llm_judge_bea import (
    _AXIS_GROUPS_BEA,
)
from evaluation.evaluators.llm_judge_bea import RUBRIC_KEYS_BEA as _RUBRIC_KEYS_BEA

_RUBRIC_LABELS_BEA = {
    "confusion_identification": "Confusion ID",
    "confusion_location": "Conf.location",
    "answer_withheld": "Ans.withheld",
    "guidance_quality": "Guidance",
    "actionability": "Actionability",
    "coherence": "Coherence",
    "tutor_tone": "Tutor tone",
    "human_likeness": "Human-like",
}


def _rubric_config(judge_mode: str) -> tuple:
    """Return (evaluator_key, rubric_keys, labels, axis_groups) for a judge mode."""
    if judge_mode == "bea":
        return "llm_judge_bea", _RUBRIC_KEYS_BEA, _RUBRIC_LABELS_BEA, _AXIS_GROUPS_BEA
    return "llm_judge", _RUBRIC_KEYS_NAIVE, _RUBRIC_LABELS_NAIVE, _AXIS_GROUPS_NAIVE


_SEP = "-" * 80
_THICK = "=" * 80


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _backend_label(t_back: str, s_back: str) -> str:
    short = {
        "gpt-realtime": "GPT",
        "gemini-live": "Gemini",
        "personaplex": "PersonaPlex",
        "moshivis": "MoshiVis",
        "human": "Human",
    }
    t = short.get(t_back, t_back)
    s = short.get(s_back, s_back)
    return f"{t}↑ / {s}↓"


def _row_mean(vals: list[int]) -> float | None:
    return round(statistics.mean(vals), 2) if vals else None


def build_leaderboard(
    run_names: list[str],
    duplex_root: Path,
    eval_root: Path,
    *,
    judge_mode: str = "bea",
) -> tuple[dict, dict]:
    """Return (board, meta) keyed by backend-label with per-criterion score lists."""
    ev_key, rubric_keys, _, _ = _rubric_config(judge_mode)
    board: dict[str, dict[str, list[int]]] = {}
    meta: dict[str, dict] = {}

    for run_name in run_names:
        scores_path = eval_root / run_name / "scores.jsonl"
        convs_path = duplex_root / run_name / "conversations.jsonl"

        if not scores_path.is_file():
            logger.warning("No scores found for %s — run evaluation first", run_name)
            continue
        if not convs_path.is_file():
            logger.warning("No conversations.jsonl for %s, skipping", run_name)
            continue

        scores = _load_jsonl(scores_path)
        convs = _load_jsonl(convs_path)
        convs_by_idx = {int(c.get("conv_index", i)): c for i, c in enumerate(convs)}

        for rec in scores:
            idx = int(rec.get("conv_index", -1))
            conv = convs_by_idx.get(idx, {})
            t_back = conv.get("tutor_backend", "?")
            s_back = conv.get("student_backend", "?")
            label = _backend_label(t_back, s_back)

            judge = rec.get("evaluator_results", {}).get(ev_key, {})
            if judge.get("status") != "ok":
                continue

            if label not in board:
                board[label] = {k: [] for k in rubric_keys}
                meta[label] = {"run": run_name, "tutor": t_back, "student": s_back}

            for k in rubric_keys:
                entry = judge.get("scores", {}).get(k)
                if isinstance(entry, dict) and isinstance(entry.get("score"), int):
                    board[label][k].append(entry["score"])

    return board, meta


def format_leaderboard(
    board: dict,
    meta: dict,
    run_names: list[str],
    *,
    sort_by: str = "overall",
    judge_mode: str = "bea",
) -> str:
    _, rubric_keys, rubric_labels, axis_groups = _rubric_config(judge_mode)
    judge_tag = (
        "BEA 2025 (Maurya et al., NAACL 2025)" if judge_mode == "bea" else "Naive"
    )

    lines: list[str] = []
    lines.append(_THICK)
    lines.append(f"TUTOR BACKEND LEADERBOARD  [{judge_tag}]")
    lines.append(f"Runs: {', '.join(run_names)}")
    lines.append(_THICK)

    if not board:
        lines.append(
            "No scored runs found. Run: uv run python -m evaluation run --run-name <name>"
        )
        return "\n".join(lines)

    # Sort backends by sort_by axis mean (descending).
    def _sort_key(label: str) -> float:
        vals = board[label].get(sort_by, [])
        return -statistics.mean(vals) if vals else 0.0

    sorted_labels = sorted(board.keys(), key=_sort_key)

    # --- Axis summary table ---
    lines.append("")
    lines.append("Axis summary  (mean score / 5, higher is better)")
    lines.append(_SEP)

    col_w = 14
    header = f"{'Backend':<28}" + "".join(f"{ax:>{col_w}}" for ax in axis_groups)
    header += f"{'n':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for label in sorted_labels:
        row = f"{label:<28}"
        n = None
        for ax, keys in axis_groups.items():
            axis_vals: list[int] = []
            for k in keys:
                axis_vals.extend(board[label].get(k, []))
            mu = _row_mean(axis_vals)
            if n is None and axis_vals:
                n = len(board[label].get(keys[0], []))
            row += f"{(f'{mu:.2f}' if mu is not None else '--'):>{col_w}}"
        row += f"{(n or 0):>6}"
        lines.append(row)

    # --- Per-criterion detail table ---
    lines.append("")
    lines.append("")
    lines.append("Per-criterion detail")
    lines.append(_SEP)

    crit_col_w = 12
    header2 = f"{'Backend':<28}" + "".join(
        f"{rubric_labels[k]:>{crit_col_w}}" for k in rubric_keys
    )
    lines.append(header2)
    lines.append("-" * len(header2))

    for label in sorted_labels:
        row = f"{label:<28}"
        for k in rubric_keys:
            vals = board[label].get(k, [])
            mu = _row_mean(vals)
            row += f"{(f'{mu:.2f}' if mu is not None else '--'):>{crit_col_w}}"
        lines.append(row)

    # --- Score distributions per criterion ---
    lines.append("")
    lines.append("")
    lines.append("Score distributions (count per 0–5 bucket)")
    lines.append(_SEP)

    for label in sorted_labels:
        lines.append(f"  {label}  (run: {meta[label]['run']})")
        for k in rubric_keys:
            vals = board[label].get(k, [])
            if not vals:
                continue
            dist = {i: vals.count(i) for i in range(6)}
            dist_str = "  ".join(f"{i}:{dist[i]}" for i in range(6))
            mu = statistics.mean(vals)
            lines.append(f"    {rubric_labels[k]:<14} mean={mu:.2f}   [{dist_str}]")
        lines.append("")

    lines.append(_THICK)
    return "\n".join(lines)


def run_leaderboard(
    run_names: list[str] | None = None,
    *,
    duplex_root: Path = Path("duplex_output"),
    eval_root: Path = Path("eval_output"),
    out: Path | None = None,
    sort_by: str = "guidance_quality",
    judge_mode: str = "bea",
) -> Path:
    """Build leaderboard from scored runs, write to file, return path."""
    if run_names is None:
        run_names = [
            d.name
            for d in sorted(eval_root.iterdir())
            if d.is_dir()
            and not d.name.endswith("__aligned")
            and (d / "scores.jsonl").is_file()
        ]
        if not run_names:
            raise FileNotFoundError(
                f"No scored runs found under {eval_root}. "
                "Run: uv run python -m evaluation run --run-name <name>"
            )

    board, meta = build_leaderboard(
        run_names, duplex_root, eval_root, judge_mode=judge_mode
    )
    text = format_leaderboard(
        board, meta, run_names, sort_by=sort_by, judge_mode=judge_mode
    )

    if out is None:
        eval_root.mkdir(parents=True, exist_ok=True)
        suffix = "_bea" if judge_mode == "bea" else "_naive"
        out = eval_root / f"leaderboard{suffix}.txt"

    out.write_text(text)
    logger.info("Leaderboard written to %s", out)
    return out


def add_subparser(subparsers) -> None:
    """Register the `leaderboard` subcommand on the unified CLI."""
    p = subparsers.add_parser(
        "leaderboard",
        help="Cross-run tutor backend leaderboard from existing scores.jsonl files.",
        description="Cross-run tutor backend leaderboard.",
    )
    p.add_argument(
        "--runs", nargs="*", help="Run names (default: all scored runs in eval-root)"
    )
    p.add_argument("--duplex-root", default="duplex_output")
    p.add_argument("--eval-root", default="eval_output")
    p.add_argument(
        "--out",
        default=None,
        help="Output path (default: eval_output/leaderboard_{bea|naive}.txt)",
    )
    p.add_argument(
        "--judge",
        default="bea",
        choices=["bea", "naive"],
        help="Which judge's scores to use: 'bea' (BEA 2025, default) or 'naive' (original 5-key rubric)",
    )
    p.add_argument(
        "--sort-by",
        default=None,
        help="Rubric key to sort backends by (descending). Defaults to 'guidance_quality' for BEA, 'overall' for naive.",
    )
    p.set_defaults(_cmd_fn=_run, _cmd_parser=p)


def _run(args) -> None:
    _, rubric_keys, _, _ = _rubric_config(args.judge)
    sort_by = args.sort_by or ("guidance_quality" if args.judge == "bea" else "overall")
    if sort_by not in rubric_keys:
        args._cmd_parser.error(
            f"--sort-by {sort_by!r} not valid for --judge {args.judge}. "
            f"Choices: {list(rubric_keys)}"
        )

    out_path = run_leaderboard(
        run_names=args.runs or None,
        duplex_root=Path(args.duplex_root),
        eval_root=Path(args.eval_root),
        out=Path(args.out) if args.out else None,
        sort_by=sort_by,
        judge_mode=args.judge,
    )
    print(out_path.read_text())
