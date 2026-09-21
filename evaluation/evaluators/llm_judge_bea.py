"""BEA 2025-aligned LLM-as-judge evaluator.

Applies the 8-dimension pedagogical taxonomy from:

  Maurya et al., "Unifying AI Tutor Evaluation: An Evaluation Taxonomy for
  Pedagogical Ability Assessment of LLM-Powered AI Tutors," NAACL 2025.
  https://aclanthology.org/2025.naacl-long.57.pdf

Adapted for whole-conversation scoring on MathVista (0–5 Likert per dimension).
Kept alongside the original `llm_judge` evaluator so both can be compared.

Reuses OpenAIJudge and JudgeProvider from llm_judge — no duplication.
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

DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "tutor_judge_bea.txt"
DEFAULT_MODEL = DEFAULT_MODELS["openai"]

RUBRIC_KEYS_BEA = (
    "confusion_identification",
    "confusion_location",
    "answer_withheld",
    "guidance_quality",
    "actionability",
    "coherence",
    "tutor_tone",
    "human_likeness",
)

_AXIS_GROUPS_BEA = {
    "Error diagnosis": ("confusion_identification", "confusion_location"),
    "Pedagogical quality": ("answer_withheld", "guidance_quality", "actionability"),
    "Dialogue quality": ("coherence", "tutor_tone", "human_likeness"),
}


def _parse_rubric_bea(raw: str) -> dict[str, dict[str, Any]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    out: dict[str, dict[str, Any]] = {}
    for k in RUBRIC_KEYS_BEA:
        v = obj.get(k)
        if isinstance(v, dict) and "score" in v:
            out[k] = {
                "score": int(v["score"]),
                "rationale": str(v.get("rationale", "")),
            }
    return out


class LLMJudgeBEA(Evaluator):
    name = "llm_judge_bea"
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
            logger.exception("BEA judge call failed for pid=%s", conv.get("pid"))
            return {"status": "error", "error": str(e), "model": self.model}

        try:
            scores = _parse_rubric_bea(raw)
        except Exception as e:
            logger.warning(
                "BEA judge JSON parse failed for pid=%s: %s", conv.get("pid"), e
            )
            return {
                "status": "parse_error",
                "error": str(e),
                "raw": raw[:1000],
                "model": self.model,
            }

        component_scores = [scores[k]["score"] for k in RUBRIC_KEYS_BEA if k in scores]
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
