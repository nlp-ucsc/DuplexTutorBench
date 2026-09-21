"""Judge-provider clients and shared helpers used by every LLM-judge evaluator.

Provider plumbing lives here so that adding a new rubric (a new evaluator)
only requires the rubric definition + a parser — never re-implementing the
retry loop or the transcript formatter.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Default model id per judge provider (used when --judge-model is not given).
DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-6",
}


def _is_transient(exc: Exception) -> bool:
    """True for retryable API failures (rate limits, 5xx, timeouts)."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(
        t in name
        for t in (
            "timeout",
            "connection",
            "servererror",
            "unavailable",
            "ratelimit",
            "overloaded",
        )
    )


def _with_retry(
    fn: Callable[[], str], *, attempts: int = 5, base_delay: float = 2.0
) -> str:
    """Call fn, retrying transient errors with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — re-raised below if not transient
            if i == attempts - 1 or not _is_transient(e):
                raise
            delay = base_delay * (2**i)
            logger.warning(
                "Transient judge error (%s); retry %d/%d in %.1fs",
                e,
                i + 1,
                attempts - 1,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def _format_transcript(segments: list[dict]) -> str:
    """Render segments as `[t=  1.23s] role: text` lines for the judge prompt."""
    rows = []
    for s in sorted(segments, key=lambda x: float(x.get("start_time", 0.0))):
        t = float(s.get("start_time", 0.0))
        role = s.get("role", "?")
        text = (s.get("text", "") or "").strip()
        rows.append(f"[t={t:6.2f}s] {role}: {text}")
    return "\n".join(rows)


class JudgeProvider(Protocol):
    """Returns a JSON-string response for a given prompt."""

    def complete(self, prompt: str) -> str: ...


class OpenAIJudge:
    """OpenAI Chat Completions backend."""

    def __init__(
        self, model: str = DEFAULT_MODELS["openai"], api_key: str | None = None
    ):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the LLM judge "
                "(loaded from .env via duplex/run.py-style dotenv)."
            )
        self.model = model
        self.client = OpenAI(api_key=key)

    def complete(self, prompt: str) -> str:
        def _call() -> str:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""

        return _with_retry(_call)


class GeminiJudge:
    """Google Gemini backend (text-only) via the google-genai SDK."""

    def __init__(
        self, model: str = DEFAULT_MODELS["gemini"], api_key: str | None = None
    ):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini judge.")
        self.model = model
        self.client = genai.Client(api_key=key)

    def complete(self, prompt: str) -> str:
        def _call() -> str:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            return resp.text or ""

        return _with_retry(_call)


class ClaudeJudge:
    """Anthropic Claude backend via the anthropic SDK."""

    def __init__(
        self, model: str = DEFAULT_MODELS["claude"], api_key: str | None = None
    ):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the Claude judge.")
        self.model = model
        self.client = anthropic.Anthropic(api_key=key)

    def complete(self, prompt: str) -> str:
        # Claude has no JSON-mode flag; the prompt instructs JSON-only output
        # and each evaluator's parser tolerates stray fences / surrounding prose.
        def _call() -> str:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if msg.content else ""

        return _with_retry(_call)


def make_judge_provider(provider: str, model: str) -> JudgeProvider:
    """Instantiate a judge provider by name. SDK imports are lazy per-class."""
    if provider == "openai":
        return OpenAIJudge(model=model)
    if provider == "gemini":
        return GeminiJudge(model=model)
    if provider == "claude":
        return ClaudeJudge(model=model)
    raise ValueError(
        f"Unknown judge provider: {provider!r} (known: openai, gemini, claude)"
    )
