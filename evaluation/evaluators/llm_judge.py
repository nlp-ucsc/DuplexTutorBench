"""LLM-as-judge evaluator (naive 5-key rubric) over the conversation transcript.

Provider plumbing (OpenAI/Gemini/Claude clients, retry, transcript formatting)
lives in `evaluation.evaluators.providers`; this file only defines the rubric
and the parser.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from evaluation.base import Evaluator
from evaluation.evaluators.providers import (
    DEFAULT_MODELS,
    JudgeProvider,
    _format_transcript,
    make_judge_provider,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "tutor_judge.txt"
DEFAULT_MODEL = DEFAULT_MODELS["openai"]

RUBRIC_KEYS = (
    "answer_correctness",
    "scaffolding_quality",
    "student_realism",
    "phrasing_naturalness",
    "overall",
)


def _parse_rubric(raw: str) -> dict[str, dict[str, Any]]:
    """Parse the judge's JSON response, tolerating stray ```json fences."""
    text = raw.strip()
    # Strip code fences if the model added them despite response_format.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    out: dict[str, dict[str, Any]] = {}
    for k in RUBRIC_KEYS:
        v = obj.get(k)
        if isinstance(v, dict) and "score" in v:
            out[k] = {
                "score": int(v["score"]),
                "rationale": str(v.get("rationale", "")),
            }
    return out


class LLMJudge(Evaluator):
    name = "llm_judge"
    requires_audio = False

    def __init__(
        self,
        provider: JudgeProvider | None = None,
        provider_name: str = "openai",
        model: str | None = None,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
    ):
        model = model or DEFAULT_MODELS.get(provider_name, DEFAULT_MODEL)
        self.provider = provider or make_judge_provider(provider_name, model)
        self.model = model
        self.prompt_template = prompt_path.read_text()

    def evaluate(self, conv: dict, audio_dir: Path) -> dict[str, Any]:
        transcript = _format_transcript(conv.get("segments", []))
        prompt = self.prompt_template.format(
            question=conv.get("question", ""),
            answer=conv.get("answer", ""),
            transcript=transcript,
        )
        try:
            raw = self.provider.complete(prompt)
        except Exception as e:
            logger.exception("LLM judge call failed for pid=%s", conv.get("pid"))
            return {"status": "error", "error": str(e), "model": self.model}

        try:
            scores = _parse_rubric(raw)
        except Exception as e:
            logger.warning(
                "LLM judge JSON parse failed for pid=%s: %s", conv.get("pid"), e
            )
            return {
                "status": "parse_error",
                "error": str(e),
                "raw": raw[:1000],
                "model": self.model,
            }

        # Flat numeric mean across the rubric (excluding `overall`) so the UI
        # can sort/filter on a single column.
        component_scores = [
            scores[k]["score"] for k in RUBRIC_KEYS if k != "overall" and k in scores
        ]
        mean_component = (
            round(sum(component_scores) / len(component_scores), 3)
            if component_scores
            else 0.0
        )

        return {
            "status": "ok",
            "model": self.model,
            "scores": scores,
            "mean_component_score": mean_component,
        }
