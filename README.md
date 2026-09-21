# Mock AI Tutor — Synthetic Conversation Generator

A research framework for generating synthetic student–AI tutor conversations grounded in math questions from the [MathVista](https://mathvista.github.io/) dataset. Two agents (a tutor and a student) interact in dialogue, producing realistic tutoring transcripts.

> **Project status (2026-05):** active development is on the full-duplex pipeline — [`duplex/`](duplex/README.md) (generation) → [`alignment/`](alignment/README.md) (ASR alignment) → [`evaluation/`](evaluation/README.md) (scoring). The turn-based mode (`main.py` → `output/`, viewed via `viewer.py`) is stable and still works, but is no longer the primary focus.

## Modes

| Mode | Entry point | Output | When to use |
|---|---|---|---|
| **Turn-based** | `python main.py` | `output/{run}/conversations.jsonl` | Utterance-by-utterance dialogue. Supports text-only, +image, +audio, +audio-history. Backed by a Qwen3-Omni server (or any OpenAI-compatible model). |
| **Full-duplex** | `python -m duplex` | `duplex_output/{run}/` | Simultaneous speech via real-time speech models (PersonaPlex, MoshiVis, GPT Realtime, Gemini Live, human-in-the-loop). Where new work is happening. See [`duplex/README.md`](duplex/README.md). |

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install dependencies
echo "OPENAI_API_KEY=sk-..." > .env      # only if using OpenAI / GPT Realtime
echo "GEMINI_API_KEY=..." >> .env        # only if using Gemini Live
echo "ANTHROPIC_API_KEY=..." >> .env     # only if using Claude as LLM judge
```

## Quick start

```bash
# Turn-based, default Qwen server (no API key needed)
uv run python main.py --run-name testmini_baseline --n 5

# Full-duplex, GPT Realtime (cloud, no remote GPU needed)
uv run python -m duplex --web --tutor-backend gpt-realtime --student-backend gpt-realtime
```

For more recipes (audio, image, audio-history, alternative backends), see the docs below.

## Docs

### Turn-based mode (this README's main subject)

| Doc | What's in it |
|---|---|
| [Quickstart](docs/quickstart.md) | Usage recipes: default Qwen, OpenAI, custom server, `--image`, `--audio`, `--audio-history` |
| [CLI reference](docs/cli-reference.md) | All `main.py` flags, output format, JSONL schema, prompt customization |
| [Architecture](docs/architecture.md) | How turns are generated, image handling, extending to other models (`ChatModel` protocol) |
| [Audio mode](docs/audio-mode.md) | Audio architecture + audio-history mode + Omni-server client |
| [Servers](docs/servers.md) | vLLM (text/text+image) and Omni server (text+audio) startup |

### Per-module READMEs

| Module | What it does |
|---|---|
| [`duplex/`](duplex/README.md) | Full-duplex conversation generation |
| [`alignment/`](alignment/README.md) | ASR forced alignment of duplex audio (faithful word-level timestamps) |
| [`evaluation/`](evaluation/README.md) | Post-hoc scoring: stats, turn-taking, naturalness, LLM judges (transcript-side) |
| [`speech_eval/`](speech_eval/README.md) | Audio-side scoring: Audiobox quality, ESPnet turn-taking judge, VAP turn-taking scorer |

## Project Structure

```
Mock_AITutor/
├── main.py                 # Turn-based CLI entry point
├── src/                    # Turn-based mode internals
│   ├── schemas.py          # Dataclasses: MathVistaQuestion, Utterance, Conversation
│   ├── models.py           # ChatModel protocol + OpenAI-compatible implementation
│   ├── audio.py            # OmniChatModel client for text+audio generation
│   ├── data.py             # MathVista dataset loading (HuggingFace)
│   └── conversation.py     # Turn-by-turn conversation simulation
├── viewer.py               # Flask web viewer for output/ datasets
├── duplex/                 # Full-duplex mode (see duplex/README.md)
├── alignment/              # ASR forced alignment (see alignment/README.md)
├── evaluation/             # Post-hoc transcript-side scoring (see evaluation/README.md)
├── speech_eval/            # Audio-side scoring (see speech_eval/README.md)
├── server/                 # Reference copies of remote server scripts
├── prompts/                # System prompt templates (turn-based + duplex)
├── docs/                   # Turn-based mode docs (this README's link map)
├── output/                 # Turn-based runs (gitignored)
├── duplex_output/          # Full-duplex runs (gitignored)
├── eval_output/            # Evaluation scores (gitignored)
└── speech_eval_output/     # Audio-side scores (gitignored)
```

## Development

```bash
uv run pre-commit run --all-files
```

Code style: Black (line-length 88) + isort (Black profile), enforced via pre-commit hooks.
