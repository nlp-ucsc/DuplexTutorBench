# Quickstart — turn-based mode

Usage recipes for `python main.py`. For all flags see [CLI reference](cli-reference.md); for how turns are generated see [Architecture](architecture.md); for the audio-mode internals see [Audio mode](audio-mode.md).

## Default — Qwen model on remote server

```bash
uv run python main.py --run-name testmini_baseline --n 5
```

Uses `Qwen/Qwen3-Omni-30B-A3B-Instruct` served at `http://100.116.140.1:8901/v1` by default (vLLM). No API key needed. Output goes to `output/testmini_baseline/conversations.jsonl`. If the server isn't running, see [Servers](servers.md).

## Using OpenAI models

```bash
uv run python main.py --run-name gpt4o_test --model gpt-4o-mini --n 5
```

Uses the OpenAI API directly (requires `OPENAI_API_KEY` in `.env`).

## Using a custom server

```bash
uv run python main.py --run-name custom_test --model my-model --base-url http://localhost:8000/v1
```

Anything OpenAI-API-compatible works (vLLM, Ollama, llama.cpp, …).

## With question images

```bash
uv run python main.py --run-name gpt4o_image --model gpt-4o-mini --n 5 --image
```

Includes each question's image (diagram, graph, chart, …) in the conversation. Images are saved to `output/gpt4o_image/image/{pid}.png`. Works with any vision-capable model (GPT-4o-mini, GPT-4o, GPT-5, Qwen3-Omni via the Omni server). For how images are injected per agent, see [Architecture → Image handling](architecture.md#image-handling).

## With audio generation

```bash
uv run python main.py --run-name testmini_audio --n 5 --audio --omni-url http://100.116.140.1:8902
```

Uses the [Qwen3-Omni server](servers.md#omni-server-textaudio) to generate both text and speech in a single forward pass. Audio files are saved to `output/testmini_audio/audio/{pid}/turn_{n}.wav`. Architecture and audio-history sub-mode are in [Audio mode](audio-mode.md).

## With images and audio

```bash
uv run python main.py --run-name full --n 5 --image --audio --omni-url http://100.116.140.1:8902
```

The image is sent to the Omni server (converted from OpenAI format to Qwen's native format internally), and audio is generated for each utterance.

## With audio history

```bash
uv run python main.py --run-name audio_hist --n 5 --audio --audio-history
```

In the default audio mode, generated audio is saved to disk but only **text** is fed back as conversation history. With `--audio-history`, the generated audio is passed back as conversation history instead, so the model "hears" prior turns as speech rather than reading transcriptions. Uses Qwen3-Omni's native audio input support. See [Audio mode → Audio history](audio-mode.md#audio-history-mode---audio-history).

```bash
# Customize speaker voices (defaults: tutor=Ethan, student=Chelsie)
uv run python main.py --run-name audio_hist --n 5 --audio --audio-history \
    --tutor-speaker Aiden --student-speaker Chelsie

# With images and audio history
uv run python main.py --run-name full_audio_hist --n 5 --image --audio --audio-history
```
