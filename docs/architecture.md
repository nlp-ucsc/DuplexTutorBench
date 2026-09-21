# Architecture — turn-based mode

How `main.py` simulates one tutor↔student conversation, how images are wired in, and how to swap in a different model backend.

## How it works

1. **Load questions** from MathVista via HuggingFace Datasets.
2. **Format system prompts** — the tutor receives the question *and* the correct answer; the student receives only the question.
3. **Simulate conversation** — the student speaks first, then the tutor and student alternate. Each agent sees the other's messages as `role: "user"` (standard ChatML framing). The conversation ends when either agent emits `[END]` or the `--max-turns` limit is reached.
4. **Write output** — each conversation is appended as a single JSON line to a JSONL file, streamed as it completes (no full-buffer write at the end, so interrupted runs preserve their finished conversations).

```
Student prompt (no answer) ──► Student LLM ──┐
                                              ├──► turn-by-turn dialogue ──► JSONL
Tutor prompt (with answer)  ──► Tutor LLM ───┘
```

## Image handling

When `--image` is enabled, each question's image (diagram, graph, chart, …) is included in the conversation using OpenAI's vision content-array format:

```json
[
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "..."}
]
```

The image is injected differently for each agent:

- **Student:** receives the image as the first `user` message with the text *"Here is the math problem. Look at the image and begin."*
- **Tutor:** receives the image bundled with the student's first reply (image + student text in the same message).

After that first exchange, subsequent turns are text-only. The Qwen3-Omni server converts this OpenAI format to Qwen's native image format internally.

## Extending to other models

The `ChatModel` protocol in `src/models.py` requires a single method:

```python
def generate(self, messages: list[dict]) -> str: ...
```

Any class implementing this interface can be used as a tutor or student model. Two implementations ship:

- `OpenAIChatModel` (`src/models.py`) — works with any OpenAI-compatible API (OpenAI, vLLM, Ollama, llama.cpp, …) via the `base_url` parameter.
- `OmniChatModel` (`src/audio.py`) — adds audio generation as a side effect (text comes back the same way, plus a `.wav` is written to disk per `generate()` call). See [Audio mode](audio-mode.md).

To add a third backend (a local llama.cpp build, a new cloud API, …), implement the protocol in a new file and wire it into `main.py`'s model construction.
