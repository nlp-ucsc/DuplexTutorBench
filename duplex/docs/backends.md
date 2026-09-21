# Backends

Per-backend internals: wire protocols, VAD tuning, image handling, quirks. For the contract a new backend must satisfy and how the relay drives them, see [Architecture](architecture.md).

## PersonaPlex

- Binary WebSocket protocol (`0x00` handshake, `0x01` audio, `0x02` text).
- Opus audio via `sphn` library at 24 kHz, 1920-sample frames.
- Requires continuous audio input (silence when idle) to step the model forward.
- Two separate server instances needed (one per role), each locks to one client.
- ~19.4 GB VRAM per instance.

## MoshiVis

- Wire-compatible with PersonaPlex: same `/api/chat` WebSocket, same Opus@24kHz 1920-sample frames, same `sphn` codec, same `0x00` handshake / `0x01` audio tagging.
- New frame type `0x08 + raw_png_bytes` is sent as the **first** client message before the server issues its handshake; the MoshiVis server decodes it with `torchvision.io.decode_image` and precomputes cross-attention K/V for the image.
- Text frames use `0x07` (PT backend): 1-byte gate-color prefix + UTF-8 token (the backend also accepts `0x02` for forward compatibility with the MLX server).
- Always sends `?xa_start=0` as a URL query param — the shipped `moshika-vis.yaml` has `xa_start: "start"` (a string) but the server compares it numerically to the step offset; the query param hits the server's `int(aux)` override path.
- One image per session — reconnect to change images.
- ~20 GB VRAM per BF16 instance on the PT backend; one instance per role, each holds an `asyncio.Lock` (same single-client pattern as PersonaPlex).
- Single voice (Moshika, female) — no tutor/student voice contrast.
- **No system-prompt / persona input**: the server accepts only three inbound frame types (`0x08` image, `0x01` audio, `0x0a` rating) and exposes only generation/VAD knobs via URL query params. Persona is baked into the Moshika-vis weights, so tutor and student share the same persona — we can't steer one into "patient tutor" and the other into "curious student" the way we do for PersonaPlex (voice prompt) or the cloud backends (system instructions). `MoshiVisBackend.__init__` takes no `text_prompt` for this reason, and `run.py`'s prompt-template formatting is a no-op for MoshiVis runs.

## GPT Realtime

- OpenAI Realtime API via `openai` Python SDK.
- PCM16 audio at 24 kHz, base64-encoded in JSON events.
- Server-side `server_vad` with `create_response: true` + `interrupt_response: true` — the server auto-responds on end-of-speech and cancels its own in-flight response if it hears the other agent start talking (barge-in).
- On barge-in: yields `("interrupt", None)` from `recv()` when the server sends `input_audio_buffer.speech_started`. That hooks into the relay's buffer-flush + recording-truncation machinery (see [Architecture → Barge-in](architecture.md#barge-in)). The server stops generating after cancellation, so no source-level audio suppression is needed.
- VAD tuning: `threshold 0.5`, `prefix_padding_ms 300`, `silence_duration_ms 400`.
- Audio fed continuously (including silence) at real-time pace for natural VAD behavior.
- If the MathVista question has an image, it is sent once at session start via `conversation.item.create` as a single `input_image` part (no accompanying user-text cue — an explanatory cue was biasing the model and occasionally flipped the student↔tutor roles) and persists for the whole conversation — requires the GA `gpt-realtime` model; the older `gpt-4o-realtime-preview` is audio-only.
- Cloud API — no GPU required, needs `OPENAI_API_KEY`.
- Student role auto-initiates the conversation via `response.create()` (once, at session start).

## Gemini Live

- Google Gemini Live API via `google-genai` Python SDK (async WebSocket).
- PCM16 audio: input resampled from 24 kHz → 16 kHz, output 24 kHz (matches relay's internal 24 kHz format).
- SDK-default automatic activity detection — Google's server handles speech boundaries and auto-generates replies on detected silence.
- On barge-in: yields `("interrupt", None)` when the server sets `server_content.interrupted=True`. Unlike GPT, Gemini's server keeps delivering already-generated in-flight audio chunks *after* the interrupt flag, so the backend latches a `suppress` flag inside `recv()` that drops subsequent audio and output-transcription deltas from the same turn until `generation_complete` / `turn_complete` (or the SDK's `receive()` iterator ends). Truncating the recording buffer alone is not enough — new chunks would extend it again.
- Audio fed continuously at real-time pace, same pattern as GPT Realtime.
- Cloud API — no GPU required, needs `GEMINI_API_KEY`.
- Student role auto-initiates by calling `send_realtime_input(text="Please begin.")` (once, at session start) since VAD cannot fire without prior audio. **Empirical observation (not documented Google behavior)**: in our testing with `google-genai==1.73.1` against `gemini-3.1-flash-live-preview`, the `send_client_content` text kickoff produced no `server_content` for the full silence-timeout window — only `session_resumption_update` heartbeats. Routing the same kickoff through `send_realtime_input(text=...)` produced `model_turn` + audio immediately. Root cause not confirmed: it could be a model-side change, an SDK-version interaction, or a conflict with our session config (`response_modalities=["AUDIO"]`, the prior `send_client_content(turn_complete=False)` image-attach turn, default `automatic_activity_detection`, etc.). The 2.5 native-audio model accepted the `send_client_content` path.
- If the MathVista question has an image, it is sent once at session start via `send_client_content` with `turn_complete=False` as a single `inline_data` PNG part (no accompanying user-text cue — an explanatory cue was biasing the model and occasionally flipped the student↔tutor roles), then persists in the session for every subsequent VAD-driven turn. The non-terminal turn adds the image to context without triggering generation, so the initiator's kickoff (or the other side's first VAD turn) is the first actual response.

## Human

- Browser microphone via `getUserMedia` → inline `AudioWorkletNode` (`MicFramer`) in the existing 24 kHz `AudioContext`. Frames are accumulated to 1920 samples (80 ms), converted to int16, and streamed as binary WebSocket messages to `/ws`.
- Server: `DuplexWebApp._handle_ws` re-chunks inbound int16 bytes into 1920-sample frames (per-connection residual buffer) and calls `HumanBackend.push_mic_frame`. The relay consumes them like any other backend.
- No remote server, no API key, no GPU — mic is client-side only.
- Echo/feedback: the human role's own browser playback gain is muted while the mic is active. AEC/noise-suppression/AGC are enabled in the `getUserMedia` constraints, but **headphones are strongly recommended** — AEC alone is not enough to fully prevent the other agent's voice from bleeding back into the mic.
- Barge-in is not emitted from the human side. The AI's own server-side VAD detects the human speaking and emits its own `("interrupt", None)`, which the relay handles through the existing path.
- `HumanBackend.is_realtime_source = True` — the relay caps the downstream cross-feed queue at 2 frames in `_recv_from`. Unlike AI backends there is no pre-generated audio to stream later; any backlog is pure latency on the AI's ear.
- Recording: the human audio is saved to `{role}_full.wav` + `{role}_full.raw` (PCM16) just like GPT/Gemini. `voice` is stored as `"human"` in the JSONL.
