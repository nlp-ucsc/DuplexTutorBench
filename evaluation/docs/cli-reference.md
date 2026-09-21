# CLI Reference

`uv run python -m evaluation <subcommand> [...]` — subcommands are `run`, `report`, `leaderboard`, `import-mtb`. This page documents `run` (the main scoring entry point) and the on-disk output format. `leaderboard` and `import-mtb` are covered in [Tooling](tooling.md).

## `run` flags

| Flag | Default | Description |
|---|---|---|
| `--run-name` | *required* | Duplex run to evaluate (reads `duplex_output/{run_name}/`) |
| `--evaluators` | `stats,turn_taking,llm_judge` | Comma-separated subset of `stats`, `turn_taking`, `naturalness`, `llm_judge`, `llm_judge_bea` |
| `--conv-index` | — | If set, score only this single conversation index (good for prompt iteration) |
| `--force` | off | Recompute even if a result for this evaluator already exists |
| `--judge-provider` | `openai` | Judge backend: `openai` / `gemini` / `claude`. Reads `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` from `.env` accordingly. |
| `--judge-model` | per provider | Override the judge model id. Provider defaults: `openai`→`gpt-5-mini`, `gemini`→`gemini-2.5-flash`, `claude`→`claude-sonnet-4-6` |
| `--duplex-root` | `duplex_output` | Override the root dir holding `{run_name}/` duplex datasets |
| `--eval-root` | `eval_output` | Override the root dir to write `{run_name}/scores.jsonl` into |
| `--align-variant` | — | Score an aligned-transcript variant from the [`alignment/`](../../alignment/README.md) module (e.g. `whisper_mms`, `whisper_mms_no_bias`, `whisper_mfa`, `whisper_mfa_no_bias`, `text_mms`). Reads `duplex_output/{run}/aligned/<variant>.jsonl` and writes to `eval_output/{run}__<variant>/scores.jsonl` so original + each variant accumulate side-by-side. |
| `--eval-subdir` | — | Explicit output subdir under `--eval-root` (overrides the default `{run_name}` / `{run_name}__<align-variant>`). Used by the judge sweep to keep per-judge results in their own dirs, e.g. `{run_name}__judge-claude-sonnet-4-6`. |
| `--report` | off | After scoring, also write per-dialog text reports + `summary.txt` under `{eval-root}/{eval-subdir}/reports/` (calls `evaluation.report.write_reports`). |

## Output layout

```
eval_output/{run_name}/
├── scores.jsonl                # one JSON line per conversation
└── reports/                    # written by --report (or `report` subcommand)
    ├── conv_000.txt
    ├── conv_001.txt
    └── summary.txt
```

When `--align-variant` or `--eval-subdir` is used, the output dir gets a suffix (`{run_name}__whisper_mms`, `{run_name}__judge-claude-sonnet-4-6`) so different variants/judges accumulate side by side rather than overwriting.

The runner does a full rewrite of `scores.jsonl` each run (atomic via `.tmp` rename), so partial updates are safe.

## JSONL schema

One JSON line per conversation, keyed by the same `conv_index` used by `duplex/storage.py`:

```json
{
  "pid": "1",
  "conv_index": 0,
  "timestamp": "2026-04-29T03:31:11.915+00:00",
  "evaluator_results": {
    "stats": {
      "duration_s": 92.181,
      "num_segments": 9,
      "tutor_talk_ratio": 0.346,
      "student_talk_ratio": 0.084,
      "tutor_words_per_min": 244.3,
      "student_words_per_min": 1258.8,
      "...": "..."
    },
    "turn_taking": {
      "response_latency_s": {"mean": 3.69, "median": 1.816, "p90": 7.625, "count": 8},
      "overlap_s": 0.0,
      "silence_ratio": 0.569,
      "backchannel_count": 0,
      "...": "..."
    },
    "llm_judge": {
      "status": "ok",
      "model": "gpt-5-mini",
      "scores": {
        "answer_correctness":   {"score": 4, "rationale": "..."},
        "scaffolding_quality":  {"score": 5, "rationale": "..."},
        "student_realism":      {"score": 4, "rationale": "..."},
        "phrasing_naturalness": {"score": 4, "rationale": "..."},
        "overall":              {"score": 4, "rationale": "..."}
      },
      "mean_component_score": 4.25
    },
    "llm_judge_bea": {
      "status": "ok",
      "model": "gpt-5-mini",
      "scores": {
        "confusion_identification": {"score": 4, "rationale": "..."},
        "confusion_location":       {"score": 3, "rationale": "..."},
        "answer_withheld":          {"score": 5, "rationale": "..."},
        "guidance_quality":         {"score": 4, "rationale": "..."},
        "actionability":            {"score": 4, "rationale": "..."},
        "coherence":                {"score": 5, "rationale": "..."},
        "tutor_tone":               {"score": 5, "rationale": "..."},
        "human_likeness":           {"score": 4, "rationale": "..."}
      },
      "mean_component_score": 4.25
    },
    "naturalness": {
      "status": "ok",
      "metric": "dnsmos",
      "tutor_dnsmos":   {"ovrl": 2.65, "sig": 2.96, "bak": 4.08, "p808": 2.95},
      "student_dnsmos": {"ovrl": 3.26, "sig": 3.49, "bak": 4.14, "p808": 3.73}
    }
  }
}
```

Each evaluator owns its own metric schema under `evaluator_results[<evaluator name>]`. Failed evaluations are stored with `status: "error"` (or `"parse_error"` when an LLM judge returns unparseable JSON) and a plain re-run will retry them automatically.
