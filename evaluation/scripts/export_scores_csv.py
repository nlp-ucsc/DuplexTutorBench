"""
Export eval_output scores to CSV files.

Usage:
  uv run python evaluation/scripts/export_scores_csv.py

Outputs two files in eval_output/:
  summary.csv   — one row per run, averages for every metric
  all_runs.csv  — one row per conversation, all runs combined
"""

import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_output"

LLM_SCORE_KEYS = [
    "answer_correctness",
    "scaffolding_quality",
    "student_realism",
    "phrasing_naturalness",
    "overall",
]

DETAIL_COLS = (
    ["run_name", "conv_index", "pid"]
    + ["duration_s", "num_segments", "role_switches"]
    + ["tutor_words", "tutor_speaking_time_s", "tutor_talk_ratio"]
    + ["student_words", "student_speaking_time_s", "student_talk_ratio"]
    + ["latency_mean_s", "latency_median_s", "latency_p90_s"]
    + ["overlap_s", "overlap_ratio", "silence_s", "silence_ratio"]
    + ["backchannel_count", "backchannel_ratio"]
    + LLM_SCORE_KEYS
    + ["mean_component_score"]
)


def _load_run(run_dir: Path) -> list[dict]:
    with (run_dir / "scores.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_row(run_name: str, record: dict) -> dict:
    ev = record["evaluator_results"]
    stats = ev.get("stats", {})
    tt = ev.get("turn_taking", {})
    latency = tt.get("response_latency_s", {})
    judge = ev.get("llm_judge", {})
    scores = judge.get("scores", {})

    row = {
        "run_name": run_name,
        "conv_index": record.get("conv_index", ""),
        "pid": record.get("pid", ""),
        "duration_s": stats.get("duration_s", ""),
        "num_segments": stats.get("num_segments", ""),
        "role_switches": stats.get("role_switches", ""),
        "tutor_words": stats.get("tutor_words", ""),
        "tutor_speaking_time_s": stats.get("tutor_speaking_time_s", ""),
        "tutor_talk_ratio": stats.get("tutor_talk_ratio", ""),
        "student_words": stats.get("student_words", ""),
        "student_speaking_time_s": stats.get("student_speaking_time_s", ""),
        "student_talk_ratio": stats.get("student_talk_ratio", ""),
        "latency_mean_s": latency.get("mean", ""),
        "latency_median_s": latency.get("median", ""),
        "latency_p90_s": latency.get("p90", ""),
        "overlap_s": tt.get("overlap_s", ""),
        "overlap_ratio": tt.get("overlap_ratio", ""),
        "silence_s": tt.get("silence_s", ""),
        "silence_ratio": tt.get("silence_ratio", ""),
        "backchannel_count": tt.get("backchannel_count", ""),
        "backchannel_ratio": tt.get("backchannel_ratio", ""),
        "mean_component_score": judge.get("mean_component_score", ""),
    }
    for key in LLM_SCORE_KEYS:
        row[key] = scores.get(key, {}).get("score", "")
    return row


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 3) if vals else ""


def main():
    run_dirs = sorted(
        d for d in EVAL_DIR.iterdir() if d.is_dir() and (d / "scores.jsonl").exists()
    )
    if not run_dirs:
        print(f"No runs found under {EVAL_DIR}")
        return

    all_rows = []
    for run_dir in run_dirs:
        records = _load_run(run_dir)
        for record in records:
            all_rows.append(_extract_row(run_dir.name, record))
        print(f"  Loaded {run_dir.name} ({len(records)} conversations)")

    numeric_cols = DETAIL_COLS[3:]  # everything after run_name, conv_index, pid

    # all_runs.csv
    all_runs_path = EVAL_DIR / "all_runs.csv"
    with all_runs_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {all_runs_path}")

    # summary.csv — averages per run
    summary_cols = ["run_name"] + numeric_cols
    summary_path = EVAL_DIR / "summary.csv"
    run_names = [d.name for d in run_dirs]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for run_name in run_names:
            run_rows = [r for r in all_rows if r["run_name"] == run_name]
            summary_row = {"run_name": run_name}
            for col in numeric_cols:
                summary_row[col] = _mean([r[col] for r in run_rows])
            writer.writerow(summary_row)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
