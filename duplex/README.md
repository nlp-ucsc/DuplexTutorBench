# Full-Duplex Conversation Module

Simultaneous-speech (full-duplex) student/tutor conversations driven by real-time speech models. Unlike the turn-by-turn parent project, both agents listen and speak concurrently. The relay is model-agnostic — backends plug in via a single ABC (`ModelBackend` in `backend.py`).

> **Evaluating these conversations:** see [`../evaluation/README.md`](../evaluation/README.md). The evaluation web UI is mounted into this module's web app via `evaluation.web_integration.register_routes` — per-conversation scores show up in the History panel and a run-level summary lives at `/eval-summary`.

## Supported backends

| Backend | Model | Audio Codec | Accepts Image | Infrastructure |
|---------|-------|-------------|:-------------:|----------------|
| `personaplex` | [PersonaPlex 7B](https://huggingface.co/nvidia/personaplex-7b-v1) | Opus (via `sphn`) | no | Self-hosted, 2 GPU instances |
| `moshivis` | [Moshika-Vis](https://huggingface.co/kyutai/moshika-vis-pytorch-bf16) | Opus (via `sphn`) | **yes** | Self-hosted, 2 GPU instances |
| `gpt-realtime` | [GPT Realtime](https://platform.openai.com/docs/guides/realtime) | PCM16 24kHz | yes | OpenAI cloud API |
| `gemini-live` | [Gemini Live](https://ai.google.dev/gemini-api/docs/live-api) | PCM16 16kHz in / 24kHz out | yes | Google cloud API |
| `human` | Browser microphone (human-in-the-loop) | PCM16 24kHz over WebSocket | no | Web UI only |

Backends can be mixed per-role (e.g., PersonaPlex tutor + GPT Realtime student) for cross-model comparison. At most one role may be `human`, and `human` requires `--web`.

## Quick start

Web UI with two cloud backends (no remote GPU needed; requires `OPENAI_API_KEY` in `.env`):

```bash
uv run python -m duplex --web --tutor-backend gpt-realtime --student-backend gpt-realtime
```

Open <http://localhost:5002>, pick a question, click **Start**. Recordings land under `duplex_output/<run>/`.

For PersonaPlex/MoshiVis (which need remote GPU servers), batch runs, and the human-in-the-loop seat, see the [Quickstart](docs/quickstart.md).

## Docs

| Doc | What's in it |
|---|---|
| [Quickstart](docs/quickstart.md) | Per-backend setup + first run (PersonaPlex, MoshiVis, GPT, Gemini, Human, Mixed) |
| [Web UI](docs/web-ui.md) | Live monitor + history playback walkthrough |
| [Batch mode](docs/batch-mode.md) | Scaling: attempts, resume, manifest, sweeps |
| [CLI reference](docs/cli-reference.md) | All flags, output format, JSONL schema, termination conditions |
| [Backends](docs/backends.md) | Per-backend internals: wire protocols, VAD tuning, image handling |
| [Architecture](docs/architecture.md) | Relay design, turn-taking, barge-in, module layout |

## Adding a new backend

1. Create `duplex/backend_<name>.py` implementing `ModelBackend` (see [Architecture](docs/architecture.md) for the contract and [Backends](docs/backends.md) for examples).
2. Add a case to `create_backend()` in `run.py`.
3. Add the name to `BACKEND_CHOICES` in `run.py`.

If the backend's server does VAD-based response cancellation, yield `("interrupt", None)` from `recv()` on cancel — the relay's flush + recording truncation + web-UI plumbing are backend-agnostic and one yield hooks into all three. See [Architecture → Barge-in](docs/architecture.md#barge-in).
