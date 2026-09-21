# Architecture

How the runner dispatches to evaluators, what an evaluator must implement, and how the module mounts into the duplex web UI.

## Module layout

```
evaluation/
├── __init__.py
├── __main__.py             # Unified CLI: subcommands `run`, `report`, `leaderboard`, `import-mtb`
├── base.py                 # Evaluator ABC + EvaluationResult dataclass
├── runner.py               # Loads conversations, dispatches to evaluators, writes scores.jsonl
├── report.py               # write_reports(): per-dialog text reports + summary.txt
├── leaderboard.py          # Cross-run backend comparison table (subcommand: leaderboard)
├── mathtutorbench.py       # MathTutorBench dialogues → duplex schema (subcommand: import-mtb)
├── web_integration.py      # register_routes / render_panel_* called by duplex/web.py
├── prompts/
│   ├── tutor_judge.txt     # 5-key rubric prompt template
│   └── tutor_judge_bea.txt # 8-dim BEA / Maurya et al. rubric prompt template
├── evaluators/             # Concrete Evaluator implementations
│   ├── __init__.py
│   ├── stats.py            # ConversationStats (zero deps)
│   ├── turn_taking.py      # TurnTaking (zero deps)
│   ├── naturalness.py      # SpeechNaturalness — DNSMOS via `speechmos`
│   ├── providers.py        # JudgeProvider Protocol + OpenAIJudge / GeminiJudge / ClaudeJudge
│   │                       #   + make_judge_provider() factory + transient-error retry
│   ├── llm_judge.py        # LLMJudge (5-key naive rubric) + parser
│   └── llm_judge_bea.py    # LLMJudgeBEA (8-dim BEA taxonomy) + parser
└── scripts/                # Standalone CLI scripts (see Tooling docs)
```

The framework is fully decoupled from `duplex/`'s runtime — the only thing it imports from `duplex` is `storage.load_conversations` for parsing the JSONL. It does **not** depend on any of the live backends (`backend_gpt.py`, `backend_gemini.py`, etc.).

## `Evaluator` contract

```python
class Evaluator(ABC):
    name: str                      # short id, e.g. "turn_taking"
    requires_audio: bool = False   # if True, runner skips conversations whose audio_dir is missing

    @abstractmethod
    def evaluate(self, conv: dict, audio_dir: Path) -> dict[str, Any]:
        """Return a JSON-serializable dict of metrics for this conversation.

        `conv` is the raw JSONL dict (as produced by duplex.storage.load_conversations);
        `audio_dir` is `duplex_output/{run_name}/audio/{idx}/` and may be missing for
        text-only evaluators.
        """
```

Each evaluator owns its own metric schema — there's no central registry of metric keys to keep in sync. The runner stores each evaluator's output dict under its `name` key in the per-conversation result, and the aggregate / web summary code walks numeric leaves generically.

## Adding a new evaluator

1. Create `evaluation/evaluators/{your_axis}.py` with a class subclassing `Evaluator`. Pick a short `name`, set `requires_audio` if needed, return a JSON-serializable dict from `evaluate(conv, audio_dir)`.
2. Add a case to `build_evaluators` in `runner.py` (lazy-imported, so it doesn't trigger optional deps when unused).
3. Add the name to `ALL_EVALUATOR_NAMES` in `__main__.py` so it shows up in `--evaluators` validation.
4. Optional: add a small render function in `web_integration.py`'s `render_panel_script` so the new metrics show up in the per-conversation panel. The summary page picks up numeric leaves automatically (no change needed).

**Heavy deps:** if your new evaluator needs a torch model or a PyPI package that isn't lightweight, follow the lazy-import pattern from `evaluators/naturalness.py`: import the dep lazily inside `__init__` or `evaluate`, and self-skip with a clean `{"status": "skipped", "reason": "..."}` dict when the import fails. The DNSMOS deps (`speechmos` / `librosa` / `onnxruntime`) are standard now, but the lazy import keeps the framework robust against partial installs.

**LLM-judge rubric:** for a new rubric, you only need the rubric keys + a JSON parser + a prompt template — the provider plumbing already lives in `evaluators/providers.py` (see `evaluators/llm_judge_bea.py` for the pattern).

## Web UI integration

The evaluation module owns the per-conversation panel and the run-level summary page. `duplex/web.py` mounts it via three calls:

```python
# duplex/web.py
from evaluation.web_integration import (
    register_routes as register_eval_routes,
    render_panel_html, render_panel_script,
)

# In create_app():
register_eval_routes(app, get_run_dir=lambda: self.run_dir)

# In _handle_index() before returning:
html = HTML_TEMPLATE.replace("<!--EVAL_PANEL-->", render_panel_html())
html = html.replace("<!--EVAL_SCRIPT-->", render_panel_script())
```

The injected JS exposes `window.loadEvalScores(idx)`, which the existing `selectConversation(idx)` calls when the user picks a conversation in the History view. `get_run_dir` is invoked per request, so when the user switches the active run via the duplex header dropdown the eval panel automatically points at the new run's `eval_output/{run}/scores.jsonl` with no re-registration needed.

Routes added by `register_routes`:

| Route | Returns |
|---|---|
| `GET /api/eval/scores` | All scores for the active run |
| `GET /api/eval/scores/{idx}` | One conversation's scores (404 if not yet evaluated) |
| `GET /api/eval/summary` | Aggregate stats — numeric leaves rolled up to mean / min / max across the run |
| `GET /eval-summary` | Standalone HTML page: aggregate panel + sortable per-conversation table |

The `Run summary →` link inside the per-conversation panel opens `/eval-summary` in a new tab. The summary page is fully self-contained (no shared CSS dependency on `duplex/web.py`) so it survives template changes on either side.

Evaluation is currently **CLI-driven** — there's no "Evaluate this run" button in the UI yet. Run the CLI, then refresh the History view; the panel will populate.

## Concurrency

The runner is **sequential**, matching the `duplex/run.py` batch-mode shape. For thousands of conversations + the LLM judge, an asyncio batch with a semaphore would help; not needed at current dataset sizes.
