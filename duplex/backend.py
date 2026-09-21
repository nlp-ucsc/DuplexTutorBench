"""Abstract backend interface and PersonaPlex implementation for full-duplex models."""

import asyncio
import logging
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import numpy as np
import sphn

logger = logging.getLogger(__name__)

BACKEND_CHOICES = ["personaplex", "gpt-realtime", "gemini-live", "moshivis", "human"]
IMAGE_CAPABLE_BACKENDS = frozenset({"moshivis", "gpt-realtime", "gemini-live"})


def uses_image(backend_type: str, image_bytes: bytes | None) -> bool:
    return image_bytes is not None and backend_type in IMAGE_CAPABLE_BACKENDS


# Audio parameters — canonical format shared across all backends
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # samples per frame (80ms at 24kHz)


def pcm_float32_to_int16(pcm: np.ndarray) -> np.ndarray:
    """Convert float32 PCM [-1, 1] to int16 PCM."""
    return np.clip(pcm * 32767, -32768, 32767).astype(np.int16)


def pcm_int16_bytes_to_float32(raw: bytes) -> np.ndarray:
    """Convert int16 PCM bytes to float32 PCM in [-1, 1]."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class ModelBackend(ABC):
    """Interface for a full-duplex model connection.

    Each backend handles its own protocol (WebSocket, SDK, etc.) and codec
    (Opus, PCM16, etc.). The relay communicates with backends exclusively
    in float32 PCM at 24kHz.
    """

    def __init__(self, role: str):
        self._role = role

    @property
    def role(self) -> str:
        return self._role

    @property
    @abstractmethod
    def needs_continuous_input(self) -> bool:
        """Whether this model requires continuous audio input to step forward."""

    @property
    def is_realtime_source(self) -> bool:
        """Whether audio is produced at wall-clock pace (e.g. a mic).

        The relay caps the downstream cross-feed queue for such backends
        because any backlog is pure latency — there is nothing to pre-stream.
        """
        return False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection and complete any handshake."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection and release resources."""

    @abstractmethod
    async def send_audio(self, pcm: np.ndarray) -> None:
        """Send float32 PCM audio (24kHz) to the model.

        The backend is responsible for encoding to its native format.
        """

    @abstractmethod
    async def recv(self) -> AsyncGenerator[tuple[str, Any], None]:
        """Yield events from the model.

        Event types:
            ("audio", np.ndarray)    — float32 PCM at 24kHz
            ("native_audio", bytes)  — raw codec bytes for native-format recording
            ("text", str)            — a text token
            ("interrupt", None)      — the model's server detected the other
                                       agent speaking and cancelled its own
                                       in-flight response; the relay should
                                       drop any of this model's already-buffered
                                       output heading to the other agent
            ("closed", None)         — connection closed
        """
        yield  # pragma: no cover

    @abstractmethod
    def get_native_audio_ext(self) -> str:
        """File extension for native audio format (e.g. '.opus', '.raw')."""


# ---------------------------------------------------------------------------
# PersonaPlex backend
# ---------------------------------------------------------------------------

# PersonaPlex binary message types
_MSG_HANDSHAKE = 0x00
_MSG_AUDIO = 0x01
_MSG_TEXT = 0x02
_MSG_AUDIO_PREFIX = bytes([_MSG_AUDIO])


class PersonaPlexBackend(ModelBackend):
    """Backend for PersonaPlex (Moshi-based) servers.

    Uses a binary WebSocket protocol with Opus audio via the sphn library.
    """

    def __init__(
        self,
        host: str,
        port: int,
        text_prompt: str,
        voice_prompt: str,
        role: str,
        seed: int | None = None,
    ):
        super().__init__(role)
        self._host = host
        self._port = port
        self._text_prompt = text_prompt
        self._voice_prompt = voice_prompt
        self._seed = seed

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        # Encoder for sending audio TO the server
        self._opus_writer: sphn.OpusStreamWriter | None = None

    @property
    def needs_continuous_input(self) -> bool:
        return True

    def get_native_audio_ext(self) -> str:
        return ".opus"

    def _build_url(self) -> str:
        params = {
            "text_prompt": self._text_prompt,
            "voice_prompt": self._voice_prompt,
        }
        if self._seed is not None:
            params["seed"] = str(self._seed)
        query = urllib.parse.urlencode(params)
        return f"ws://{self._host}:{self._port}/api/chat?{query}"

    async def connect(self) -> None:
        url = self._build_url()
        logger.info("[%s] Connecting to PersonaPlex: %s", self._role, url)
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(url)
        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

        # Wait for handshake byte
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if len(msg.data) > 0 and msg.data[0] == _MSG_HANDSHAKE:
                    logger.info("[%s] PersonaPlex handshake received.", self._role)
                    return
            elif msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            ):
                raise ConnectionError(f"{self._role} WebSocket closed during handshake")
        raise ConnectionError(f"{self._role} WebSocket ended without handshake")

    async def disconnect(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self._ws = None
        self._session = None
        self._opus_writer = None

    async def send_audio(self, pcm: np.ndarray) -> None:
        if self._opus_writer is None or self._ws is None:
            return
        # Split into frame-sized chunks — Opus requires specific frame sizes
        offset = 0
        while offset < len(pcm):
            end = min(offset + FRAME_SIZE, len(pcm))
            chunk = pcm[offset:end]
            # Pad short final chunk to frame size
            if len(chunk) < FRAME_SIZE:
                chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
            self._opus_writer.append_pcm(chunk)
            offset = end
        opus_data = self._opus_writer.read_bytes()
        if len(opus_data) > 0:
            await self._ws.send_bytes(_MSG_AUDIO_PREFIX + opus_data)

    async def recv(self) -> AsyncGenerator[tuple[str, Any], None]:
        if self._ws is None:
            return

        # Decoder for converting received Opus to PCM
        opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                if len(data) == 0:
                    continue

                kind = data[0]
                payload = data[1:]

                if kind == _MSG_AUDIO:
                    opus_reader.append_bytes(payload)
                    pcm = opus_reader.read_pcm()
                    if pcm.shape[0] > 0:
                        yield ("audio", pcm)
                    # Also yield the raw Opus bytes for native-format recording
                    yield ("native_audio", payload)

                elif kind == _MSG_TEXT:
                    text = payload.decode("utf-8")
                    yield ("text", text)

            elif msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            ):
                break

        yield ("closed", None)
