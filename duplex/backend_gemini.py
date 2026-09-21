"""Gemini Live API backend for full-duplex conversations."""

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from duplex.backend import (
    SAMPLE_RATE,
    ModelBackend,
    pcm_float32_to_int16,
    pcm_int16_bytes_to_float32,
)

logger = logging.getLogger(__name__)

# Gemini Live expects 16kHz PCM16 input, outputs 24kHz PCM16
_GEMINI_INPUT_RATE = 16000


def _resample_24k_to_16k(pcm: np.ndarray) -> np.ndarray:
    """Resample float32 PCM from 24kHz to 16kHz using linear interpolation."""
    if len(pcm) == 0:
        return pcm
    ratio = _GEMINI_INPUT_RATE / SAMPLE_RATE  # 2/3
    out_len = int(len(pcm) * ratio)
    indices = np.arange(out_len) / ratio
    indices = np.clip(indices, 0, len(pcm) - 1)
    idx_floor = indices.astype(np.intp)
    idx_ceil = np.minimum(idx_floor + 1, len(pcm) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    return pcm[idx_floor] * (1 - frac) + pcm[idx_ceil] * frac


class GeminiLiveBackend(ModelBackend):
    """Backend using Google's Gemini Live API for full-duplex audio.

    Uses server-VAD auto-response (default Gemini Live behavior): each model
    detects when the other side's audio goes silent and generates its own
    response. The relay feeds audio at real-time pace so VAD sees natural
    speech boundaries.
    """

    def __init__(
        self,
        instructions: str,
        role: str,
        voice: str = "Kore",
        model: str = "gemini-3.1-flash-live-preview",
        api_key: str | None = None,
        initiate: bool = False,
        image_bytes: bytes | None = None,
    ):
        super().__init__(role)
        self._instructions = instructions
        self._voice = voice
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._initiate = initiate
        self._image_bytes = image_bytes
        self._client: genai.Client | None = None
        self._session = None
        self._session_cm = None

    @property
    def needs_continuous_input(self) -> bool:
        return True

    def get_native_audio_ext(self) -> str:
        return ".raw"

    async def connect(self) -> None:
        logger.info(
            "[%s] Connecting to Gemini Live (model=%s)...", self._role, self._model
        )
        self._client = genai.Client(api_key=self._api_key)

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice
                    )
                )
            ),
            system_instruction=self._instructions,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        self._session_cm = self._client.aio.live.connect(
            model=self._model,
            config=config,
        )
        self._session = await self._session_cm.__aenter__()
        logger.info("[%s] Gemini Live session established.", self._role)

        # Attach the question image (if any) before the kickoff so the first
        # response can reason over it. `turn_complete=False` adds the image
        # to conversation context without triggering generation — the
        # initiator's kickoff below drives the first turn, and for the
        # non-initiator the image just sits in context until VAD fires.
        # Image-only (no accompanying text part): an explanatory user-text
        # cue was previously biasing the model's role — both sides would
        # sometimes respond to the cue as if it were teacher framing,
        # flipping student↔tutor.
        if self._image_bytes is not None:
            await self._session.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    data=self._image_bytes,
                                    mime_type="image/png",
                                )
                            ),
                        ],
                    )
                ],
                turn_complete=False,
            )
            logger.info(
                "[%s] Gemini Live image attached (%d bytes).",
                self._role,
                len(self._image_bytes),
            )

        # Kickoff for the initiator: VAD can't fire without incoming audio,
        # so one side must speak first. Empirical observation (not documented
        # Google behavior): in our testing with `google-genai==1.73.1` against
        # `gemini-3.1-flash-live-preview`, the equivalent `send_client_content`
        # text kickoff produced no server_content — only session_resumption_update
        # heartbeats — for the full silence-timeout window. Routing the same
        # kickoff through `send_realtime_input(text=...)` produced model_turn +
        # audio immediately, so we use that path here. The 2.5 native-audio
        # model accepted the `send_client_content` form. Root cause unconfirmed
        # (model change, SDK-version interaction, or a conflict with our
        # session config / prior image-attach turn).
        if self._initiate:
            logger.info("[%s] Initiating conversation...", self._role)
            await self._session.send_realtime_input(text="Please begin.")

    async def disconnect(self) -> None:
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                logger.debug(
                    "[%s] Error closing Gemini Live session",
                    self._role,
                    exc_info=True,
                )
        self._session = None
        self._session_cm = None
        self._client = None

    async def send_audio(self, pcm: np.ndarray) -> None:
        if self._session is None:
            return
        pcm_16k = _resample_24k_to_16k(pcm)
        pcm_int16 = pcm_float32_to_int16(pcm_16k)
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm_int16.tobytes(), mime_type="audio/pcm;rate=16000")
        )

    async def recv(self) -> AsyncGenerator[tuple[str, Any], None]:
        if self._session is None:
            return

        try:
            # The SDK's receive() iterator ends after each model turn
            # (on turn_complete). We loop so the session stays alive
            # across multiple turns for full-duplex conversation.
            #
            # Latched across responses: once `interrupted=True` arrives, drop
            # any further audio/text from the same turn. Gemini's server
            # often emits `interrupted` on a response that still carries an
            # audio chunk, and keeps delivering in-flight pre-generated
            # chunks in subsequent responses. Truncating the recording buffer
            # once isn't enough — new chunks would extend it again — so we
            # suppress them at the source until the turn boundary.
            suppress = False

            while self._session is not None:
                async for response in self._session.receive():
                    sc = response.server_content
                    if sc is None:
                        continue

                    # Check interrupted first so audio/text inside the same
                    # response are also suppressed.
                    if sc.interrupted:
                        logger.debug("[%s] Gemini: model was interrupted", self._role)
                        suppress = True
                        yield ("interrupt", None)

                    if sc.model_turn and sc.model_turn.parts and not suppress:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                raw_bytes = part.inline_data.data
                                yield ("audio", pcm_int16_bytes_to_float32(raw_bytes))
                                yield ("native_audio", raw_bytes)

                    if (
                        sc.output_transcription
                        and sc.output_transcription.text
                        and not suppress
                    ):
                        yield ("text", sc.output_transcription.text)

                    if sc.generation_complete or sc.turn_complete:
                        suppress = False

                # The SDK ends `receive()` on turn_complete. Reset here too
                # in case the turn ended without an explicit flag (e.g. after
                # an interruption that didn't carry turn_complete).
                suppress = False

                logger.debug(
                    "[%s] Gemini: turn complete, awaiting next turn", self._role
                )

        except Exception:
            logger.exception("[%s] Gemini Live recv loop error", self._role)

        yield ("closed", None)
