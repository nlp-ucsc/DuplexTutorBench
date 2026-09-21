"""
Upload eval_output scores.jsonl files to a new Google Spreadsheet.

Setup (one-time):
  1. Enable Google Sheets API + Google Drive API in Google Cloud Console.
  2. Create OAuth2 credentials (Desktop App) and download as credentials.json
     into the project root (Mock_AITutor/credentials.json).
  3. uv run python evaluation/scripts/upload_scores_to_sheets.py

The script will open a browser URL for OAuth authorization on first run, then
cache the token in token.json so subsequent runs skip the auth prompt.
"""

import json
import os
import statistics
from pathlib import Path

import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_output"
CREDS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"

LLM_SCORE_KEYS = [
    "answer_correctness",
    "scaffolding_quality",
    "student_realism",
    "phrasing_naturalness",
    "overall",
]

# Column labels for the detail sheets (one row per conversation)
DETAIL_COLS = (
    ["conv_index", "pid"]
    + ["duration_s", "num_segments", "role_switches"]
    + ["tutor_words", "tutor_speaking_time_s", "tutor_talk_ratio"]
    + ["student_words", "student_speaking_time_s", "student_talk_ratio"]
    + ["latency_mean_s", "latency_median_s", "latency_p90_s"]
    + ["overlap_s", "overlap_ratio", "silence_s", "silence_ratio"]
    + ["backchannel_count", "backchannel_ratio"]
    + LLM_SCORE_KEYS
    + ["mean_component_score"]
)


def _get_creds() -> Credentials:
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_FILE.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDS_FILE}.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def _load_run(run_dir: Path) -> list[dict]:
    scores_file = run_dir / "scores.jsonl"
    with scores_file.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_row(record: dict) -> list:
    ev = record["evaluator_results"]
    stats = ev.get("stats", {})
    tt = ev.get("turn_taking", {})
    latency = tt.get("response_latency_s", {})
    judge = ev.get("llm_judge", {})
    scores = judge.get("scores", {})

    return [
        record.get("conv_index", ""),
        record.get("pid", ""),
        # stats
        stats.get("duration_s", ""),
        stats.get("num_segments", ""),
        stats.get("role_switches", ""),
        stats.get("tutor_words", ""),
        stats.get("tutor_speaking_time_s", ""),
        stats.get("tutor_talk_ratio", ""),
        stats.get("student_words", ""),
        stats.get("student_speaking_time_s", ""),
        stats.get("student_talk_ratio", ""),
        # turn-taking
        latency.get("mean", ""),
        latency.get("median", ""),
        latency.get("p90", ""),
        tt.get("overlap_s", ""),
        tt.get("overlap_ratio", ""),
        tt.get("silence_s", ""),
        tt.get("silence_ratio", ""),
        tt.get("backchannel_count", ""),
        tt.get("backchannel_ratio", ""),
        # llm judge scores
        *[scores.get(k, {}).get("score", "") for k in LLM_SCORE_KEYS],
        judge.get("mean_component_score", ""),
    ]


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 3) if vals else ""


def build_summary_rows(
    run_names: list[str], all_records: dict[str, list[dict]]
) -> list[list]:
    """One row per run with averages for every numeric metric."""
    header = ["run_name"] + DETAIL_COLS[2:]  # drop conv_index and pid
    rows = [header]
    for run_name in run_names:
        records = all_records[run_name]
        extracted = [_extract_row(r) for r in records]
        # col indices align with DETAIL_COLS; skip first two (conv_index, pid)
        avgs = [
            _mean([row[i] for row in extracted]) for i in range(2, len(DETAIL_COLS))
        ]
        rows.append([run_name] + avgs)
    return rows


def main():
    creds = _get_creds()
    gc = gspread.authorize(creds)

    run_dirs = sorted(
        d for d in EVAL_DIR.iterdir() if d.is_dir() and (d / "scores.jsonl").exists()
    )
    if not run_dirs:
        print(f"No run directories with scores.jsonl found under {EVAL_DIR}")
        return

    run_names = [d.name for d in run_dirs]
    all_records = {name: _load_run(d) for name, d in zip(run_names, run_dirs)}

    print(f"Loaded {len(run_dirs)} runs: {run_names}")
    print("Creating spreadsheet...")

    spreadsheet = gc.create("Mock AITutor Eval Scores")

    # Share as read-only link — anyone with the URL can view but not edit
    spreadsheet.share(None, perm_type="anyone", role="reader")

    # --- Summary sheet ---
    summary_ws = spreadsheet.sheet1
    summary_ws.update_title("Summary")
    summary_rows = build_summary_rows(run_names, all_records)
    summary_ws.update(summary_rows, value_input_option="USER_ENTERED")
    # Bold header row
    summary_ws.format("1:1", {"textFormat": {"bold": True}})
    print(f"  Written Summary sheet ({len(summary_rows)-1} runs)")

    # --- Per-run sheets ---
    for run_name in run_names:
        records = all_records[run_name]
        ws = spreadsheet.add_worksheet(
            title=run_name, rows=len(records) + 2, cols=len(DETAIL_COLS)
        )
        rows = [DETAIL_COLS] + [_extract_row(r) for r in records]
        ws.update(rows, value_input_option="USER_ENTERED")
        ws.format("1:1", {"textFormat": {"bold": True}})
        print(f"  Written sheet '{run_name}' ({len(records)} rows)")

    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
    print(f"\nDone! Spreadsheet URL:\n{url}")


if __name__ == "__main__":
    main()
