# CLI Reference

`uv run python -m duplex [...]` — see [Quickstart](quickstart.md) for typical invocations and [Batch mode](batch-mode.md) for scaling-related flags.

## Backend selection

| Flag | Default | Description |
|---|---|---|
| `--tutor-backend` | `personaplex` | Backend for tutor (`personaplex`, `moshivis`, `gpt-realtime`, `gemini-live`, `human`) |
| `--student-backend` | `personaplex` | Backend for student (`personaplex`, `moshivis`, `gpt-realtime`, `gemini-live`, `human`) |

`human` is valid only with `--web` (the microphone is browser-driven) and cannot be set on both roles at once.

## PersonaPlex options

| Flag | Default | Description |
|---|---|---|
| `--tutor-host` | `100.116.140.1` | Tutor server host |
| `--tutor-port` | `8998` | Tutor server port |
| `--student-host` | `100.116.140.1` | Student server host |
| `--student-port` | `8999` | Student server port |
| `--tutor-voice` | `NATM1.pt` | Tutor voice prompt file |
| `--student-voice` | `NATF2.pt` | Student voice prompt file |

Available PersonaPlex voices: `NATF0-F3.pt`, `NATM0-M3.pt`, `VARF0-F4.pt`, `VARM0-M4.pt`.

## MoshiVis options

| Flag | Default | Description |
|---|---|---|
| `--moshivis-host` | `100.116.140.1` | MoshiVis server host (shared for both roles) |
| `--moshivis-tutor-port` | `8088` | Tutor server port |
| `--moshivis-student-port` | `8089` | Student server port |

Single voice only (Moshika). The MathVista image for the selected question is uploaded once at session start via the `0x08` binary frame.

## GPT Realtime options

| Flag | Default | Description |
|---|---|---|
| `--gpt-model` | `gpt-realtime` | GPT Realtime model name (image input requires `gpt-realtime`; the older `gpt-4o-realtime-preview` is audio-only) |
| `--tutor-gpt-voice` | `ash` | GPT voice for tutor |
| `--student-gpt-voice` | `shimmer` | GPT voice for student |

Available GPT voices: `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse` (plus `marin`, `cedar` on `gpt-realtime`). Requires `OPENAI_API_KEY` environment variable. If the MathVista question has an image, it is attached once at session start and persists for the whole conversation.

## Gemini Live options

| Flag | Default | Description |
|---|---|---|
| `--gemini-model` | `gemini-3.1-flash-live-preview` | Gemini Live model name (image input requires a Live model that accepts images — `gemini-3.1-flash-live-preview` does; earlier 2.5 native-audio previews may not) |
| `--tutor-gemini-voice` | `Charon` | Gemini voice for tutor |
| `--student-gemini-voice` | `Leda` | Gemini voice for student |

Available Gemini voices: `Kore`, `Puck`, `Charon`, `Fenrir`, `Aoede`, `Leda`, `Orus`, `Zephyr`. Requires `GEMINI_API_KEY` environment variable. If the MathVista question has an image, it is attached once at session start via `send_client_content` (`turn_complete=False`) and persists for the whole conversation. The initiator's kickoff goes through `send_realtime_input(text=...)` rather than `send_client_content` — see [Backends → Gemini Live](backends.md#gemini-live) for the empirical reason.

## General options

| Flag | Default | Description |
|---|---|---|
| `--run-name` | required in batch / optional in `--web` | Output directory name (`duplex_output/{run_name}/`). In `--web` mode the header dropdown can pick or create one after launch, so the CLI flag just seeds the initial selection. |
| `--tutor-prompt` | `prompts/duplex_tutor_system.txt` | Tutor prompt template |
| `--student-prompt` | `prompts/duplex_student_system.txt` | Student prompt template |
| `--max-duration` | `300` | Max conversation duration in seconds |
| `--split` | `testmini` | MathVista split |
| `--pid` | — | One or more question PIDs, space-separated (batch mode), e.g. `--pid 1 2 4` |
| `--n` | all | First N questions in dataset order (batch mode) |
| `--sample` | — | Random N questions (mutually exclusive with `--n` / `--pid`); seeded by `--seed` |
| `--seed` | `0` | RNG seed used by `--sample` |
| `--attempts` | `1` | Independent generations per question (batch mode) |
| `--concurrency` | `1` | Parallel conversations — cloud-API backends only (`gpt-realtime`, `gemini-live`) |
| `--force` | off | Override manifest mismatch on resume |
| `--replay` | — | Path to a `manifest.json` from a prior run; reproduces that dataset |
| `--web` | off | Launch web UI instead of batch mode |
| `--port` | `5002` | Web UI port |

`--attempts`, `--sample`, `--seed`, `--concurrency`, `--force`, and `--replay` are batch-only — the web UI ignores them with a warning.

## Output format

```
duplex_output/{run_name}/
├── manifest.json                 # batch-mode only: snapshot of args + prompts + target pids
├── conversations.jsonl
├── audio/
│   └── {idx}/                    # idx = JSONL line number (positional)
│       ├── tutor_full.wav        # WAV (universal playback)
│       ├── student_full.wav
│       ├── combined.wav          # stereo: tutor=L, student=R (time-aligned)
│       ├── tutor_full.opus       # native format (PersonaPlex / MoshiVis)
│       └── student_full.raw      # native format (GPT Realtime / Gemini Live, PCM16)
└── aligned/                      # optional, written by the alignment module
    ├── whisper_mms.jsonl         # one file per (mode, no_bias) variant
    ├── whisper_mms_no_bias.jsonl
    ├── whisper_mfa.jsonl
    ├── whisper_mfa_no_bias.jsonl
    └── text_mms.jsonl
```

Audio is saved in both WAV (universal) and the backend's native format. All three WAVs (`tutor_full.wav`, `student_full.wav`, `combined.wav`) are conversation-aligned: each one starts at conversation t=0 and is exactly `conv.duration` long, with leading + trailing silence padding as needed so a per-role file and the stereo mix line up sample-for-sample.

`manifest.json` is written by the first batch invocation of a `--run-name` and used to enforce reproducibility on resume + power `--replay`. The web UI does not write or validate it.

## JSONL schema

One conversation per line:

```json
{
  "pid": "1",
  "attempt_index": 0,
  "question": "What is the spring constant?",
  "answer": "150",
  "tutor_voice": "NATM1.pt",
  "student_voice": "default",
  "tutor_backend": "personaplex",
  "student_backend": "gpt-realtime",
  "tutor_uses_image": false,
  "student_uses_image": true,
  "duration": 45.2,
  "segments": [
    {"role": "student", "text": "I'm not sure how to start...", "start_time": 2.1, "end_time": 5.3},
    {"role": "tutor", "text": "Let's look at Hooke's law...", "start_time": 4.8, "end_time": 9.1}
  ],
  "metadata": {}
}
```

Segments can overlap in time — this is expected in full-duplex conversation. All segment timestamps and per-role audio files share a single conversation timeline (t=0 = conversation start), so no offset bookkeeping is needed for playback or alignment.

`tutor_uses_image` / `student_uses_image` are `true` when the question has an image **and** that role's backend is image-capable (`gpt-realtime`, `gemini-live`, `moshivis`); `personaplex` and `human` always read `false`. `attempt_index` is the 0-based generation number for this `(run, pid)`; rows are deduped by `(pid, attempt_index)` on resume.

> **Caveat:** `segments[].start_time` / `end_time` reflect when the model's *text deltas* arrived, not when the audio was actually played. For cloud backends that stream all text in <1 s, the timestamps are systematically off, and barge-in only truncates the audio buffer — the buffered text isn't trimmed. Use the [alignment module](../../alignment/README.md) if you need timestamps that match what was actually heard.

## Termination conditions

A conversation stops when any of these occur:

1. **Max duration** reached (`--max-duration`, default 300 s)
2. **Silence timeout** — no activity from either agent for 10 s
3. **`[END]` token** — either agent emits `[END]` in its text output
