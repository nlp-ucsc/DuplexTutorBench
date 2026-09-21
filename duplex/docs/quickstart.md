# Quickstart

One section per backend. Each one shows the minimum needed to get a single conversation running, in batch or web mode. For the full flag set, see [CLI reference](cli-reference.md); for what each backend does under the hood, see [Backends](backends.md).

## PersonaPlex

Requires two PersonaPlex server instances on the remote GPU machine (one per role):

```bash
# On remote server (2x GPUs required)
cd ~/repo/personaplex/moshi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m moshi.server --host 0.0.0.0 --port 8998
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m moshi.server --host 0.0.0.0 --port 8999
```

Then on the Mac:

```bash
# Batch mode
uv run python -m duplex --run-name pp_test --pid 1 \
  --tutor-backend personaplex --student-backend personaplex

# Web UI
uv run python -m duplex --web --run-name pp_test \
  --tutor-backend personaplex --student-backend personaplex
```

## MoshiVis

Image-aware sibling of PersonaPlex — same Opus/24kHz WebSocket, plus the MathVista image is uploaded as the first frame of each session. Requires two server instances on the remote GPU machine (one per role):

```bash
# On remote server (2x GPUs required, Python 3.11 venv — torch 2.2.0's
# Dynamo does NOT support 3.12+, so the venv must be created with
# `uv sync --python 3.11`)
cd ~/repo/moshivis/kyuteye_pt
CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run server configs/moshika-vis.yaml --host 0.0.0.0 --port 8088 --ssl False
CUDA_VISIBLE_DEVICES=1 ~/.local/bin/uv run server configs/moshika-vis.yaml --host 0.0.0.0 --port 8089 --ssl False
```

**First-time setup:** the server refuses to start without `client/dist/`. Fetch the static client once:

```bash
cd ~/repo/moshivis
~/.local/bin/uv run --project kyuteye_pt scripts/get_static_client.py
```

Then on the Mac:

```bash
# Batch mode
uv run python -m duplex --run-name mv_test --pid 1 \
  --tutor-backend moshivis --student-backend moshivis

# Web UI
uv run python -m duplex --web --run-name mv_test \
  --tutor-backend moshivis --student-backend moshivis
```

Every question must have an image — the backend raises on image-less questions; the web UI returns HTTP 400.

## GPT Realtime

Requires `OPENAI_API_KEY` in `.env`:

```bash
# Batch mode
uv run python -m duplex --run-name gpt_test --pid 1 \
  --tutor-backend gpt-realtime --student-backend gpt-realtime

# Web UI
uv run python -m duplex --web --run-name gpt_test \
  --tutor-backend gpt-realtime --student-backend gpt-realtime
```

## Gemini Live

Requires `GEMINI_API_KEY` in `.env`:

```bash
# Batch mode
uv run python -m duplex --run-name gem_test --pid 1 \
  --tutor-backend gemini-live --student-backend gemini-live

# Web UI
uv run python -m duplex --web --run-name gem_test \
  --tutor-backend gemini-live --student-backend gemini-live
```

## Mixed backends

```bash
uv run python -m duplex --run-name mixed_test --pid 1 \
  --tutor-backend personaplex --student-backend gpt-realtime
```

## Human-in-the-loop

Drop a real human into either the tutor or student seat and let them speak with the opposite AI. Web-only — the mic comes from the browser. Headphones are strongly recommended so the AI's voice doesn't bleed back into the mic (browser AEC helps but is not perfect).

```bash
# Start web mode with human tutor seat (actual choice is made in the dropdown)
uv run python -m duplex --web --run-name human_test \
  --tutor-backend human --student-backend gpt-realtime
```

Then in the browser:

- Pick `human` for at most one role in the header dropdowns (the other role's `human` option is disabled — you cannot put a human on both sides).
- Click **Start** and grant microphone permission.
- Speak. The human role's own live playback is muted automatically to avoid echo; you still hear the other agent at full volume.
- The human's audio is recorded to `tutor_full.wav` + `tutor_full.raw` (or `student_*`) just like any other backend; `tutor_voice` / `student_voice` is saved as `"human"` in the JSONL.

Barge-in still works both ways: the AI's server-side VAD detects when the human starts speaking and cancels its own response exactly as it does in AI-vs-AI runs — no extra interrupt plumbing is needed on the human side.

## Testing tools

`duplex/tools/talk_gpt_realtime.py` — talk to GPT Realtime directly via your microphone to test VAD behavior and turn-taking dynamics. Requires `pyaudio`.

```bash
uv run python duplex/tools/talk_gpt_realtime.py
```
