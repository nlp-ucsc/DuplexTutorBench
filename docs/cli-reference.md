# CLI Reference — turn-based mode

`uv run python main.py [...]` — see [Quickstart](quickstart.md) for typical invocations.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--run-name` | *required* | Name for this run (creates `output/{run_name}/`) |
| `--model` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Model name |
| `--base-url` | Auto-detected | API base URL (set for custom servers, omit for OpenAI) |
| `--split` | `testmini` | MathVista split (`testmini` or `test`) |
| `--n` | all | Number of questions to process |
| `--max-turns` | `10` | Maximum turns per conversation |
| `--tutor-prompt` | `prompts/tutor_system.txt` | Path to tutor prompt template |
| `--student-prompt` | `prompts/student_system.txt` | Path to student prompt template |
| `--image` | off | Include question image in conversation (requires vision-capable model) |
| `--audio` | off | Enable audio generation (uses [Omni server](servers.md#omni-server-textaudio) instead of [vLLM](servers.md#vllm-texttextimage)) |
| `--audio-history` | off | Pass generated audio as conversation history instead of text (requires `--audio`). See [Audio mode → Audio history](audio-mode.md#audio-history-mode---audio-history) |
| `--omni-url` | `http://100.116.140.1:8902` | Omni server URL |
| `--speaker` | `Chelsie` | Voice for audio (`Ethan`, `Chelsie`, or `Aiden`) |
| `--tutor-speaker` | `Ethan` | Tutor voice when `--audio-history` is set |
| `--student-speaker` | `Chelsie` | Student voice when `--audio-history` is set |

## Output format

Each run creates a directory under `output/`:

```
output/{run_name}/
├── conversations.jsonl       # all conversations, one per line
├── image/                    # only when --image is enabled
│   ├── 1.png
│   ├── 42.png
│   └── ...
└── audio/                    # only when --audio is enabled
    └── {pid}/
        ├── turn_0.wav
        ├── turn_1.wav
        └── ...
```

## JSONL schema

One conversation per line:

```json
{
  "pid": "42",
  "question": "What is the area of the triangle?",
  "answer": "12",
  "tutor_model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "student_model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "utterances": [
    {"role": "student", "content": "I don't get how to find the area here...", "turn": 0},
    {"role": "tutor", "content": "Let's start with what you know about triangles...", "turn": 1}
  ],
  "num_turns": 6,
  "metadata": {}
}
```

When `--image` is enabled, the conversation includes an `image_path` field:

```json
{"pid": "42", "image_path": "image/42.png", "...": "..."}
```

When `--audio` is enabled, each utterance also includes an `audio_path` field (relative to the run directory):

```json
{"role": "student", "content": "I don't get how to...", "turn": 0, "audio_path": "audio/42/turn_0.wav"}
```

## Customizing prompts

System prompts live in `prompts/` as plain-text files with template variables:

- `{question}` — the math question text
- `{answer}` — the correct answer (tutor prompt only)
- `{choices_block}` — formatted multiple-choice options (empty if not applicable)

Edit `prompts/tutor_system.txt` / `prompts/student_system.txt` directly, or point at your own files with `--tutor-prompt` / `--student-prompt`.
