"""Export the multi-model BEA judge sweep (eval_output/{run}__judge-{model}/)
to CSV.

Produces two files under eval_output/:
  judge_scores_all.csv   — one row per (run, judge_model, conversation)
  judge_scores_summary.csv — one row per (run, judge_model): averaged BEA scores

Usage:
  uv run python evaluation/scripts/export_judge_csv.py
  uv run python evaluation/scripts/export_judge_csv.py --evaluator llm_judge_bea
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.evaluators.llm_judge_bea import RUBRIC_KEYS_BEA

EVAL_DIR = _REPO_ROOT / "eval_output"
_MARKER = "__judge-"


def _discover(eval_root: Path) -> list[tuple[str, str, Path]]:
    """Return (run_name, judge_model, scores_path) for each judge-sweep dir."""
    out = []
    for d in sorted(eval_root.iterdir()):
        if not d.is_dir() or _MARKER not in d.name:
            continue
        scores = d / "scores.jsonl"
        if not scores.is_file():
            continue
        run_name, model = d.name.split(_MARKER, 1)
        out.append((run_name, model, scores))
    return out


def _load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _row(
    run: str, model: str, rec: dict, evaluator: str, keys: tuple[str, ...]
) -> dict:
    judge = rec.get("evaluator_results", {}).get(evaluator, {})
    scores = judge.get("scores", {})
    row = {
        "run_name": run,
        "judge_model": model,
        "conv_index": rec.get("conv_index", ""),
        "pid": rec.get("pid", ""),
        "status": judge.get("status", ""),
        "mean_component_score": judge.get("mean_component_score", ""),
    }
    for k in keys:
        row[k] = scores.get(k, {}).get("score", "")
    return row


def _mean(values) -> float | str:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 3) if vals else ""


def main() -> None:
    p = argparse.ArgumentParser(description="Export judge-sweep scores to CSV.")
    p.add_argument("--evaluator", default="llm_judge_bea")
    p.add_argument("--eval-root", default=str(EVAL_DIR))
    args = p.parse_args()

    eval_root = Path(args.eval_root)
    keys = tuple(RUBRIC_KEYS_BEA)  # BEA 8-dimension rubric
    detail_cols = (
        ["run_name", "judge_model", "conv_index", "pid", "status"]
        + list(keys)
        + ["mean_component_score"]
    )

    dirs = _discover(eval_root)
    if not dirs:
        raise SystemExit(f"No {_MARKER} dirs found under {eval_root}")

    all_rows = []
    for run, model, scores_path in dirs:
        recs = _load(scores_path)
        for rec in recs:
            all_rows.append(_row(run, model, rec, args.evaluator, keys))
        print(f"  {run} / {model}: {len(recs)} conversations")

    # Detail CSV.
    all_path = eval_root / "judge_scores_all.csv"
    with all_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {all_path} ({len(all_rows)} rows)")

    # Summary CSV: averages per (run, judge_model), ok rows only.
    summary_cols = (
        ["run_name", "judge_model", "n_ok"] + list(keys) + ["mean_component_score"]
    )
    summary_path = eval_root / "judge_scores_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for run, model, _ in dirs:
            rows = [
                r
                for r in all_rows
                if r["run_name"] == run
                and r["judge_model"] == model
                and r["status"] == "ok"
            ]
            summary = {"run_name": run, "judge_model": model, "n_ok": len(rows)}
            for col in list(keys) + ["mean_component_score"]:
                summary[col] = _mean([r[col] for r in rows])
            w.writerow(summary)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
