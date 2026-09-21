# Evaluation Module

Post-hoc evaluation framework for full-duplex conversations produced by the [`duplex/`](../duplex/README.md) module. Operates on a finished `duplex_output/{run_name}/` dataset (JSONL + per-role audio) and writes per-conversation scores to `eval_output/{run_name}/scores.jsonl`. Runs entirely offline against the saved dataset — never live during generation.

> **Scope: duplex only.** This module evaluates the simultaneous-speech conversations under `duplex_output/`. The turn-based conversations under `output/` (from `main.py` / `viewer.py`) are a different format — utterance-level rather than full-duplex with overlapping segments — and are out of scope here. Several v0 metrics (overlap, response latency, backchannel ratio) don't make sense on strict turn-by-turn data, so evaluating those would need a separate framework.

The framework is **deliberately minimal**: five evaluators ship today, each is a single small file, and the `Evaluator` ABC contract is loose enough that adding a new axis (emotion, prosody, multilingual, factuality, …) is one new file plus one line in `runner.py`.

## What gets scored

| Evaluator | Inputs | What it measures | Cost / deps |
|---|---|---|---|
| `stats` | JSONL only | Duration, segment counts, words/min per role, talk-time ratio, role switches | none |
| `turn_taking` | JSONL segment timestamps | Response latency (mean / median / p90), overlap, silence ratio, backchannel count | none |
| `naturalness` | `{role}_full.wav` | DNSMOS naturalness scores per role — SIG / BAK / OVRL / P.808 | `speechmos` + `librosa` + `onnxruntime` + `soundfile` (standard deps) |
| `llm_judge` | Transcript + question + answer | Original 5-key rubric (`answer_correctness`, `scaffolding_quality`, `student_realism`, `phrasing_naturalness`, `overall`) | OpenAI / Gemini / Claude API call |
| `llm_judge_bea` | Transcript + question + answer | 8-dimension BEA 2025 / Maurya et al. pedagogical taxonomy (NAACL 2025) | Same provider plumbing as `llm_judge` |

See [Evaluators](docs/evaluators.md) for the per-metric details, scoring keys, and caveats.

## Quick start

```bash
# Default: stats + turn_taking + LLM judge (OpenAI, gpt-5-mini)
uv run python -m evaluation run --run-name example_dataset

# After scoring, write per-dialog text reports + a per-run summary
uv run python -m evaluation run --run-name example_dataset --report
```

The CLI uses explicit subcommands: `run`, `report`, `leaderboard`, `import-mtb`. Re-running is **idempotent** — already-computed `(conv_index, evaluator)` pairs are skipped; `--force` to recompute.

For other invocations (subset evaluators, alternative judges, aligned variants), see [Quickstart](docs/quickstart.md).

## Docs

| Doc | What's in it |
|---|---|
| [Quickstart](docs/quickstart.md) | Common command recipes + idempotency |
| [CLI reference](docs/cli-reference.md) | All flags, output format, JSONL schema |
| [Evaluators](docs/evaluators.md) | Per-evaluator details, LLM judge providers, caveats |
| [Architecture](docs/architecture.md) | `Evaluator` contract, module layout, web UI integration, adding a new evaluator |
| [Tooling](docs/tooling.md) | Leaderboard, multi-judge sweep, human-agreement, CSV/Sheets export, MathTutorBench import |

## Sources

- [Full-Duplex-Bench (Lin et al., 2025) — turn-taking dimensions](https://arxiv.org/abs/2503.04721)
- [`speechmos` — bundled UTMOS / DNSMOS / NISQA](https://pypi.org/project/speechmos/)
- [Pedagogical evaluation of LLM tutors (CEUR Workshop, 2025)](https://ceur-ws.org/Vol-4006/paper3short.pdf)
- [TutorBench — rubric-per-example tutoring evaluation](https://arxiv.org/html/2510.02663)
- [Maurya et al., "Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors" (NAACL 2025)](https://aclanthology.org/2025.naacl-long.57.pdf) — 8-dimension taxonomy used by `llm_judge_bea`
- [MathTutorBench (Macina et al., EMNLP 2025) — pedagogical math-tutoring benchmark](https://github.com/eth-lre/mathtutorbench)
- [Audio MultiChallenge — voice-agent multi-turn benchmark](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/Audio_Multichallenge_Scale_LB%20(1).pdf)
