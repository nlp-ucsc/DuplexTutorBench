# Tooling

Beyond the core `run` / `report` subcommands, the module ships several entry points for running the judge at scale and turning the results into comparison artifacts.

## `leaderboard` subcommand

```bash
uv run python -m evaluation leaderboard [--judge {bea,naive}] [--runs ...] [--out path]
```

Cross-run backend comparison table. Auto-discovers scored runs under `eval_output/`, groups by tutor backend, prints per-axis means. `--judge bea` uses the 8-dimension rubric; `--judge naive` uses the 5-key. `--out path/to/leaderboard.txt` saves the table instead of printing.

## `import-mtb` subcommand

```bash
uv run python -m evaluation import-mtb [--hard] [--limit N] [--all] [--seed S] [--source <path-or-URL>]
```

Downloads [MathTutorBench](https://github.com/eth-lre/mathtutorbench) (Macina et al., EMNLP 2025) and emits a fake `duplex_output/<run>/conversations.jsonl` with synthetic 1-second-spaced timestamps so the existing judge pipeline scores it unchanged.

- `--limit N` (default 30), `--all`, `--hard` (the `_hard` split), `--seed S`, `--source <path-or-URL>`.
- **Text-only data** — score only with `--evaluators llm_judge,llm_judge_bea`. `stats` / `turn_taking` / `naturalness` numbers from the synthetic timestamps are not meaningful.

## `scripts/eval_runs.py`

Batch-score every duplex run with the default evaluators, then print the leaderboard.

- `--report` also writes per-dialog text reports.
- `--skip-scoring` jumps straight to the leaderboard from existing scores.

## `scripts/judge_sweep.py`

Run a judge evaluator (configurable via `--evaluator`, defaults to BEA) under multiple `(provider, model)` judges over each run, routing each into its own `eval_output/<run>__judge-<model>/scores.jsonl` so judges never overwrite each other.

- Defaults: 3 judges (`gpt-5-mini`, `gemini-2.5-flash`, `claude-sonnet-4-6`) × all runs.
- Prints a side-by-side per-criterion comparison and writes it to `eval_output/judge_comparison.txt`.

## `scripts/judge_human_agreement.py`

Score both rubrics against `dmacjam/pedagogical-rewardmodel-data` preference pairs (positive vs negative tutor turn over a shared dialogue prefix) and report pairwise accuracy + Wilson 95% CI per `(judge_model, rubric, comparison_key)`.

- Results cached in `eval_output/human_agreement/` so re-runs are free.
- Useful for validating that a new rubric / judge model tracks human pedagogical preferences.

## CSV / Sheets export

| Script | Output |
|---|---|
| `scripts/export_scores_csv.py` | `summary.csv` (one row per run, averaged metrics) + `all_runs.csv` (one row per conversation). |
| `scripts/export_judge_csv.py` | Discovers `eval_output/<run>__judge-<model>/` dirs and emits `judge_scores_all.csv` + `judge_scores_summary.csv` keyed by `(run, judge_model)`. |
| `scripts/upload_scores_to_sheets.py` | OAuth utility (one-time `credentials.json` from Google Cloud Console). Creates a new Google Spreadsheet with a Summary tab + one tab per run and prints the URL. Shares as a public read-only link by default. |

## Programmatic reports

`evaluation.report.write_reports(scores_path, conversations_path, reports_dir)` is importable directly if you want per-dialog text reports without re-running scoring — the `--report` flag and `report` subcommand are thin wrappers around it.
