"""OmniChatModel: client for the Qwen3-Omni text+audio server."""

import base64
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0


class OmniChatModel:
    """Chat model that returns text and saves audio via the omni server.

    Satisfies the ChatModel protocol (has generate(messages) -> str),
    so it can be used as a drop-in replacement for OpenAIChatModel.
    """

    def __init__(
        self,
        base_url: str,
        speaker: str = "Chelsie",
        audio_dir: Path | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.speaker = speaker
        self.audio_dir = audio_dir
        self.model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        self._last_audio_base64: str = ""

    def generate(self, messages: list[dict[str, Any]]) -> str:
        """Send messages to omni server, return text. Audio stored internally."""
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={"messages": messages, "speaker": self.speaker},
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                self._last_audio_base64 = data.get("audio_base64", "")
                return data["text"]
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "Omni server error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Exhausted retries")

    def get_last_audio_data_uri(self) -> str:
        """Return the last generated audio as a base64 data URI."""
        if self._last_audio_base64:
            return f"data:audio/wav;base64,{self._last_audio_base64}"
        return ""

    def save_last_audio(self, path: Path) -> str:
        """Save the audio from the last generate() call. Returns path string."""
        if self._last_audio_base64:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(self._last_audio_base64))
            return str(path)
        return ""
