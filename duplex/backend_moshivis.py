"""MoshiVis backend: image-aware Moshi, wire-compatible with PersonaPlex."""

import asyncio
import logging
import urllib.parse
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import numpy as np
import sphn

from duplex.backend import FRAME_SIZE, SAMPLE_RATE, ModelBackend

logger = logging.getLogger(__name__)

# Binary message types for the MoshiVis PT server (kyuteye_pt/kyuteye/server.py)
_MSG_HANDSHAKE = 0x00  # server -> client: ready
_MSG_AUDIO = 0x01  # bidirectional: Opus frame
_MSG_TEXT_PLAIN = 0x02  # server -> client (MLX backend only, kept for robustness)
_MSG_TEXT_COLORED = (
    0x07  # server -> client (PT backend): 1-byte gate-color prefix + utf8
)
_MSG_IMAGE = 0x08  # client -> server: raw PNG/JPEG bytes, sent once at session start

_MSG_AUDIO_PREFIX = bytes([_MSG_AUDIO])
_MSG_IMAGE_PREFIX = bytes([_MSG_IMAGE])


class MoshiVisBackend(ModelBackend):
    """Backend for Kyutai MoshiVis PyTorch servers.

    Wire-compatible with Moshi/PersonaPlex: same Opus@24kHz audio, same 1-byte-tagged
    binary frames on a /api/chat WebSocket. One difference: the client sends the
    image as the FIRST message (`0x08 + raw_png_bytes`) before the server emits
    its `0x00` ready handshake.
    """

    def __init__(
        self,
        host: str,
        port: int,
        image_bytes: bytes,
        role: str,
        seed: int | None = None,
        xa_start: int = 0,
        image_resolution: int | None = None,
    ):
        super().__init__(role)
        self._host = host
        self._port = port
        self._image_bytes = image_bytes
        self._seed = seed
        # Always passed as a URL param — the YAML config ships with xa_start="start"
        # (a str), which the PT server then tries to compare to an int offset and
        # crashes. Sending an int here forces the server's int(aux) override path.
        self._xa_start = xa_start
        self._image_resolution = image_resolution

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._opus_writer: sphn.OpusStreamWriter | None = None

    @property
    def needs_continuous_input(self) -> bool:
        return True

    def get_native_audio_ext(self) -> str:
        return ".opus"

    def _build_url(self) -> str:
        params: dict[str, str] = {"xa_start": str(self._xa_start)}
        if self._seed is not None:
            params["seed"] = str(self._seed)
        if self._image_resolution is not None:
            params["image_resolution"] = str(self._image_resolution)
        base = f"ws://{self._host}:{self._port}/api/chat"
        return f"{base}?{urllib.parse.urlencode(params)}"

    async def connect(self) -> None:
        url = self._build_url()
        logger.info(
            "[%s] Connecting to MoshiVis: %s (image=%d bytes)",
            self._role,
            url,
            len(self._image_bytes),
        )
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(url, max_msg_size=0)
        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

        # The server's extract_image() reads exactly one binary message with
        # kind=0x08 before anything else.
        await self._ws.send_bytes(_MSG_IMAGE_PREFIX + self._image_bytes)
        logger.info("[%s] MoshiVis image uploaded.", self._role)

        # Wait for handshake byte (server sends b"\x00" after vision encoding completes).
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if len(msg.data) > 0 and msg.data[0] == _MSG_HANDSHAKE:
                    logger.info("[%s] MoshiVis handshake received.", self._role)
                    return
            elif msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            ):
                raise ConnectionError(
                    f"{self._role} WebSocket closed during MoshiVis handshake"
                )
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
        offset = 0
        while offset < len(pcm):
            end = min(offset + FRAME_SIZE, len(pcm))
            chunk = pcm[offset:end]
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
                    yield ("native_audio", payload)

                elif kind == _MSG_TEXT_COLORED:
                    # Payload: 1-byte gate-color + utf8 text.
                    if len(payload) >= 1:
                        text = payload[1:].decode("utf-8", errors="replace")
                        if text:
                            yield ("text", text)

                elif kind == _MSG_TEXT_PLAIN:
                    text = payload.decode("utf-8", errors="replace")
                    if text:
                        yield ("text", text)

            elif msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            ):
                break

        yield ("closed", None)
