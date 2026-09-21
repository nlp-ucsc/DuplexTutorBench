# Architecture

How the relay drives two model backends concurrently, how turn-taking works without explicit hand-off, and how barge-in propagates from a VAD signal back to the saved WAV and the browser.

## Relay

```
Tutor ModelBackend              Relay (model-agnostic)              Student ModelBackend
  connect()           -->   cross-connect audio queues   <--    connect()
  recv() -> PCM/text         segmentation, callbacks            recv() -> PCM/text
  send_audio(PCM)            watchdog, recording                send_audio(PCM)
  disconnect()                                                  disconnect()
```

The relay works in **float32 PCM at 24 kHz** as the canonical internal format. Each backend converts to/from its native codec internally. Audio (including silence) is fed continuously at real-time pace (`FRAME_SIZE` per `frame_interval` in `_send_to`) so models with server-side VAD (GPT Realtime, Gemini Live) see natural speech boundaries.

The metering loop anchors to `time.monotonic()` (`next_tick += frame_interval`) rather than sleeping a flat `frame_interval` after each iteration — this keeps the effective cadence wall-clock accurate regardless of per-iteration work time. A flat-sleep schedule drifts slower than real-time by the per-iteration work time; for AI-vs-AI this is invisible (both sides drift equally) but against a real-time mic it causes the cross-feed queue to slowly accumulate, making barge-in feel seconds late.

## Turn-taking

There is **no manual turn-taking** between agents. Each model's own server-side VAD decides when to speak:

- **GPT Realtime**: `server_vad` with `create_response: true` + `interrupt_response: true`. On `silence_duration_ms` of quiet, the server auto-generates a response; on new speech detected mid-reply, the in-flight response is cancelled.
- **Gemini Live**: SDK default automatic activity detection. Google's server handles speech start/end and auto-generates replies on silence.
- **PersonaPlex**: Turn-taking is internal to the Moshi model (token-level full-duplex).

The only explicit cue our code issues is the **initial kickoff** on the `--initiate` side (the student by default), done once during `connect()` so the first speaker starts the conversation. After that, VAD drives every turn. Real-time pacing in the relay is what keeps VAD boundaries aligned with natural conversation flow; without it, responses would arrive faster than real-time and VAD would race.

## Barge-in

VAD-based cloud backends (GPT Realtime, Gemini Live) generate their response audio faster than real-time and hand all of it to us up-front. When the other agent interrupts mid-response, three separate buffers hold pre-generated audio that would otherwise keep playing:

1. The relay's `_audio_queues[other]` + local metering buffer inside `_send_to` — audio heading to the other agent.
2. The browser's Web Audio graph — `AudioBufferSourceNode`s scheduled at future `audioCtx.currentTime`s.
3. The per-role recording buffers (`_audio_buffers[role]` for WAV, `_native_audio_buffers[role]` for native format).

On interrupt, the VAD backend yields `("interrupt", None)` from `recv()`. The relay (`_recv_from`):

- Drains `_audio_queues[other]`; sets `_flush_send[other] = True` so `_send_to(other)` drops its partial frame on the next tick.
- Calls `_truncate_recording_to_realtime(role)` — trims both recording buffers back to `(wall_now - start_time - audio_start[role])` samples' worth. WAV is always trimmed; the native buffer is trimmed only when `backend.get_native_audio_ext() == ".raw"` (PCM16), because Opus streams can't be safely truncated mid-packet.
- Fires the optional `on_interrupt(role)` callback. The web UI wires this to a `{"type":"interrupt","role":role}` WebSocket broadcast; the browser then stops all tracked `AudioBufferSourceNode`s for that role and resets `audioState[role].nextTime`.

This fixes a three-way divergence that otherwise happens: the live browser would cut off, but the saved WAV would contain seconds of cancelled audio, and the other agent would keep hearing the interrupted voice for hundreds of ms. Both web and batch modes now produce recordings that match what was actually heard.

Full-duplex backends (PersonaPlex, MoshiVis) do not emit interrupt events because the Moshi architecture handles speech cessation internally and doesn't pre-generate audio ahead of real-time.

The `human` backend also does not emit interrupts — there is no "server" deciding it was interrupted; instead, the AI on the other side hears the human start talking (via its own VAD) and emits its own interrupt, which flushes cross-feed back toward the human exactly as in AI-vs-AI. In addition, `HumanBackend.is_realtime_source = True` causes `_recv_from` to cap the downstream queue at 2 frames (160 ms). Without that cap, even a small producer/consumer rate mismatch would let stale mic audio pile up ahead of the AI — it can't be "pre-streamed" from a mic, so any backlog is pure latency.

### Adding a new VAD-based backend

If you add a cloud backend whose server does VAD-based cancellation, yield `("interrupt", None)` whenever the server signals "my in-flight response was cancelled by the other side starting to talk." The relay, recording truncation, and web UI plumbing are backend-agnostic — one yield hooks into all three.

## Real-time source flag

`ModelBackend.is_realtime_source` (default `False`) — set `True` for a backend whose audio comes at wall-clock pace (mic) so the relay caps the downstream queue at 2 frames in `_recv_from`. AI backends leave it `False` so pre-generated bursts queue legitimately for metered playout.

## Module layout

```
duplex/
├── backend.py          # ModelBackend ABC + PersonaPlexBackend
├── backend_moshivis.py # MoshiVisBackend (image-aware)
├── backend_gpt.py      # GPTRealtimeBackend
├── backend_gemini.py   # GeminiLiveBackend
├── backend_human.py    # HumanBackend (browser mic, web-only)
├── relay.py            # Model-agnostic async relay (cross-connects two backends)
├── web.py              # aiohttp web UI (live monitor + history playback)
├── schemas.py          # DuplexSegment, DuplexConversation dataclasses
├── storage.py          # Save conversations (JSONL) and audio (WAV + native)
├── manifest.py         # Per-run config snapshot (args + prompts + target pids) for resume / replay
├── run.py              # CLI entry point (batch + web modes)
└── tools/
    └── talk_gpt_realtime.py  # Testing script: talk to GPT Realtime via mic
```
