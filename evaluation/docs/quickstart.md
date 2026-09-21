# Quickstart

Common invocations of `uv run python -m evaluation`. For every flag see [CLI reference](cli-reference.md); for what each evaluator does see [Evaluators](evaluators.md).

```bash
# Default: stats + turn_taking + LLM judge (OpenAI, gpt-5-mini)
uv run python -m evaluation run --run-name example_dataset

# Free, instant subset (no API calls)
uv run python -m evaluation run --run-name example_dataset --evaluators stats,turn_taking

# Single conversation only — handy when iterating on the rubric prompt
uv run python -m evaluation run --run-name example_dataset --conv-index 0 --evaluators llm_judge

# DNSMOS naturalness — speechmos / librosa / onnxruntime are standard deps
uv run python -m evaluation run --run-name example_dataset --evaluators naturalness

# BEA 2025 8-dimension rubric alongside the original 5-key rubric
uv run python -m evaluation run --run-name example_dataset \
    --evaluators llm_judge,llm_judge_bea

# Score with Gemini or Claude as judge instead of OpenAI
uv run python -m evaluation run --run-name example_dataset --judge-provider gemini
uv run python -m evaluation run --run-name example_dataset --judge-provider claude

# Score an aligned-transcript variant produced by the alignment/ module
uv run python -m evaluation run --run-name example_dataset --align-variant whisper_mms

# After scoring, write per-dialog text reports + a per-run summary.txt
uv run python -m evaluation run --run-name example_dataset --report

# Re-render reports later without re-scoring
uv run python -m evaluation report --run-name example_dataset
```

The CLI uses explicit subcommands: `run`, `report`, `leaderboard`, `import-mtb`. The first two are above; for `leaderboard` and `import-mtb`, see [Tooling](tooling.md).

## Idempotency

Re-running the same `run --run-name <name>` is **safe**:

- Already-computed `(conv_index, evaluator)` pairs are skipped.
- Records with `status: "error"` or `status: "parse_error"` are treated as not-done, so a plain re-run retries only failures.
- Use `--force` to recompute (e.g. after editing the rubric prompt or upgrading the judge model).
- Use `--conv-index N` to score one conversation only — fast iteration on prompts.

`scores.jsonl` is rewritten atomically (`.tmp` rename) so partial runs are safe to interrupt.
