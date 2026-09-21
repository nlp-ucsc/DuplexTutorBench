"""GPT Realtime API backend for full-duplex conversations."""

import base64
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from duplex.backend import (
    ModelBackend,
    pcm_float32_to_int16,
    pcm_int16_bytes_to_float32,
)

logger = logging.getLogger(__name__)


class GPTRealtimeBackend(ModelBackend):
    """Backend using OpenAI's GPT Realtime API via WebSocket.

    Audio is exchanged as PCM16 (24kHz mono) base64-encoded in JSON events.

    Uses server-side VAD with auto-response: each model detects when the other
    stopped speaking and generates its own response. Audio (including silence)
    must be fed at real-time pace so VAD sees speech boundaries naturally —
    the relay handles this metering.
    """

    def __init__(
        self,
        instructions: str,
        role: str,
        voice: str | None = None,
        model: str = "gpt-realtime",
        api_key: str | None = None,
        initiate: bool = False,
        image_bytes: bytes | None = None,
    ):
        super().__init__(role)
        self._instructions = instructions
        self._voice = voice
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._initiate = initiate
        self._image_bytes = image_bytes
        self._client: AsyncOpenAI | None = None
        self._connection = None
        self._connection_cm = None

    @property
    def needs_continuous_input(self) -> bool:
        return True

    def get_native_audio_ext(self) -> str:
        return ".raw"

    async def connect(self) -> None:
        logger.info(
            "[%s] Connecting to GPT Realtime (model=%s)...", self._role, self._model
        )
        self._client = AsyncOpenAI(api_key=self._api_key)
        self._connection_cm = self._client.realtime.connect(model=self._model)
        self._connection = await self._connection_cm.__aenter__()

        session_config: dict[str, Any] = {
            "type": "realtime",
            "instructions": self._instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 400,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
            },
        }
        if self._voice:
            session_config["audio"]["output"] = {"voice": self._voice}
        await self._connection.session.update(session=session_config)

        # Wait for session.updated so turn_detection is in effect before any
        # audio is sent or the initial response is triggered.
        async for event in self._connection:
            if event.type == "session.updated":
                break
        logger.info("[%s] GPT Realtime session configured.", self._role)

        # Attach the question image (if any) before the kickoff so the first
        # response can reason over it. The image persists in the conversation
        # for the whole session — Realtime treats it as a user message item.
        # Image-only (no accompanying text part): an explanatory user-text
        # cue was previously biasing the model's role — both sides would
        # sometimes respond to the cue as if it were teacher framing,
        # flipping student↔tutor.
        if self._image_bytes is not None:
            b64 = base64.b64encode(self._image_bytes).decode("ascii")
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    ],
                }
            )
            logger.info(
                "[%s] GPT Realtime image attached (%d bytes).",
                self._role,
                len(self._image_bytes),
            )

        # Kickoff for the initiator: VAD can't fire without incoming audio,
        # so one side must speak first. After this, VAD drives every turn.
        if self._initiate:
            logger.info("[%s] Initiating conversation...", self._role)
            await self._connection.response.create()

    async def disconnect(self) -> None:
        if self._connection_cm is not None:
            try:
                await self._connection_cm.__aexit__(None, None, None)
            except Exception:
                logger.debug(
                    "[%s] Error closing GPT Realtime connection",
                    self._role,
                    exc_info=True,
                )
        self._connection = None
        self._connection_cm = None
        self._client = None

    async def send_audio(self, pcm: np.ndarray) -> None:
        if self._connection is None:
            return
        pcm_int16 = pcm_float32_to_int16(pcm)
        audio_b64 = base64.b64encode(pcm_int16.tobytes()).decode("ascii")
        await self._connection.input_audio_buffer.append(audio=audio_b64)

    async def recv(self) -> AsyncGenerator[tuple[str, Any], None]:
        if self._connection is None:
            return

        try:
            async for event in self._connection:
                event_type = event.type

                if event_type == "response.output_audio.delta":
                    raw_bytes = base64.b64decode(event.delta)
                    yield ("audio", pcm_int16_bytes_to_float32(raw_bytes))
                    yield ("native_audio", raw_bytes)

                elif event_type == "response.output_audio_transcript.delta":
                    yield ("text", event.delta)

                elif event_type == "conversation.item.input_audio_transcription.delta":
                    pass

                elif event_type == "input_audio_buffer.speech_started":
                    # Server VAD detected the other agent speaking. With
                    # interrupt_response=true, the server cancels our own
                    # in-flight response, but audio we've already streamed
                    # is still sitting in the relay's downstream buffer and
                    # would otherwise keep playing over the interrupter.
                    yield ("interrupt", None)

                elif event_type == "error":
                    logger.error("[%s] GPT Realtime error: %s", self._role, event.error)

        except Exception:
            logger.exception("[%s] GPT Realtime recv loop error", self._role)

        yield ("closed", None)
