"""Chat model protocol and implementations."""

import logging
import time
from typing import Any, Protocol

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds


class ChatModel(Protocol):
    """Protocol for chat-based language models."""

    def generate(self, messages: list[dict[str, Any]]) -> str: ...


class OpenAIChatModel:
    """OpenAI chat completion model."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.kwargs = kwargs  # temperature, max_tokens, etc.

    def generate(self, messages: list[dict[str, Any]]) -> str:
        """Generate a response with exponential backoff on transient errors."""
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self.kwargs,
                )
                return response.choices[0].message.content
            except (RateLimitError, APIConnectionError, APIError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
        # Unreachable, but satisfies type checkers
        raise RuntimeError("Exhausted retries")
