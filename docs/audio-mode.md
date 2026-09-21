# Audio mode (`--audio`)

When `--audio` is enabled, the framework uses a dedicated Qwen3-Omni server that generates text and speech in a single forward pass (thinker + talker), avoiding paraphrasing issues that would arise from a separate TTS step. For invocation recipes see [Quickstart → With audio generation](quickstart.md#with-audio-generation); for the server start command see [Servers → Omni server](servers.md#omni-server-textaudio).

## Architecture

```
Mac (client)                       Remote GPU Server
┌──────────────────┐              ┌─────────────────────────────┐
│  main.py         │              │  omni_server.py (port 8902) │
│  ↓               │ POST json   │  Qwen3-Omni via transformers│
│  conversation.py ├────────────→│  thinker + talker            │
│  ↓               │ text+audio  │  returns text + base64 WAV   │
│  OmniChatModel   │←────────────│                              │
│  ↓               │              └─────────────────────────────┘
│  save .wav files │
└──────────────────┘
```

Without `--audio`, the framework uses vLLM (text only) — see [Servers → vLLM](servers.md#vllm-texttextimage).

## Audio history mode (`--audio-history`)

By default, conversation history is passed as text — the model reads prior turns. With `--audio-history`, each turn's generated audio is embedded back into the message history as an audio content block (`{"type": "audio", "audio": "data:audio/wav;base64,..."}`) instead of text. The server extracts and processes these audio inputs via `process_mm_info()` from `qwen_omni_utils`, which handles decoding and resampling to 16 kHz for the model's audio encoder.

This mode creates separate `OmniChatModel` instances for tutor and student with different speaker voices, so the model can distinguish who is speaking from the audio alone.

**Key behaviors:**

- **System prompts remain text** — only turn-level history uses audio.
- **Text is still saved** in the JSONL output for every utterance.
- **Separate speaker voices** — tutor and student use different voices (configurable via `--tutor-speaker` / `--student-speaker`) so the model can distinguish roles by voice.
- **Images still work** — combine with `--image` to include question images alongside audio history.

## Client side

`src/audio.py` provides `OmniChatModel`, which calls the Omni server and handles both text and audio. It implements the same `ChatModel` protocol as `OpenAIChatModel` (see [Architecture → Extending to other models](architecture.md#extending-to-other-models)), so it plugs into `simulate_conversation()` without changes. After each `generate()` call, audio is automatically saved to `output/audio/{pid}/turn_{n}.wav` when an `audio_dir` is provided.

## Server side

The Omni server's HTTP contract, start commands, and dependency pins live in [Servers → Omni server](servers.md#omni-server-textaudio). The local `server/omni_server.py` is a reference copy of the remote `~/repo/mock_aitutor_server/omni_server.py` — when editing, always SSH to the remote box first; that's the source of truth.
