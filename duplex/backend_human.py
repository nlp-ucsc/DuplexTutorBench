"""Human-in-the-loop backend: sources audio from a browser microphone.

The browser captures mic audio via WebAudio, frames it into 1920-sample
float32 chunks at 24 kHz, and streams them as int16 binary frames on the
live WebSocket. ``DuplexWebApp._handle_ws`` decodes those frames and calls
``HumanBackend.push_mic_frame`` per frame; ``recv`` yields them to the
relay just like any model backend.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from duplex.backend import ModelBackend, pcm_float32_to_int16

logger = logging.getLogger(__name__)

# Small jitter buffer (~480 ms at 80 ms/frame) — absorbs network hiccups
# without letting the queue grow unboundedly if the relay stalls.
_QUEUE_MAX = 6


class HumanBackend(ModelBackend):
    """ModelBackend whose ``recv`` yields frames pushed from the web UI."""

    def __init__(self, role: str):
        super().__init__(role)
        self._queue: asyncio.Queue[np.ndarray | None] | None = None
        self._drop_warned = False

    @property
    def needs_continuous_input(self) -> bool:
        return False

    @property
    def is_realtime_source(self) -> bool:
        return True

    def get_native_audio_ext(self) -> str:
        return ".raw"

    async def connect(self) -> None:
        self._queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        logger.info("[%s] Human backend ready (awaiting mic frames).", self._role)

    async def disconnect(self) -> None:
        if self._queue is None:
            return
        # Drop any backlog and push a sentinel so recv() exits promptly.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def send_audio(self, pcm: np.ndarray) -> None:
        # AI audio destined for the human is broadcast to the browser by the
        # relay's on_audio callback — nothing to forward here.
        return

    async def recv(self) -> AsyncGenerator[tuple[str, Any], None]:
        assert self._queue is not None
        while True:
            pcm = await self._queue.get()
            if pcm is None:
                break
            yield "audio", pcm
            yield "native_audio", pcm_float32_to_int16(pcm).tobytes()
        yield "closed", None

    def push_mic_frame(self, pcm: np.ndarray) -> None:
        """Enqueue a float32 PCM frame from the browser mic.

        Drops the oldest frame on overflow so a stalled relay can't cause
        unbounded memory growth.
        """
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(pcm)
            except asyncio.QueueFull:
                pass
            if not self._drop_warned:
                logger.warning(
                    "[%s] mic jitter buffer full — dropping oldest frame.",
                    self._role,
                )
                self._drop_warned = True
