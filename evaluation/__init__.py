"""Post-hoc evaluation framework for full-duplex conversations.

Scope: duplex only. Consumes datasets produced by the `duplex/` module
(`duplex_output/{run_name}/conversations.jsonl` + per-conversation audio)
and emits scores to `eval_output/{run_name}/scores.jsonl`. Turn-based
datasets under `output/` (produced by `main.py`) are not handled here —
several v0 metrics (overlap, response latency, backchannel ratio) only
make sense on full-duplex segment data.

Each evaluator is a small subclass of `Evaluator` that returns a JSON-serializable
dict; new axes (e.g. emotion, prosody, multilingual) are added by dropping a new
file next to `stats.py` / `turn_taking.py` / `naturalness.py` / `llm_judge.py`
and registering it in `runner.EVALUATORS`.
"""

from evaluation.base import EvaluationResult, Evaluator
from evaluation.runner import run_evaluation

__all__ = ["Evaluator", "EvaluationResult", "run_evaluation"]
