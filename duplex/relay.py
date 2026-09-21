"""Async relay that cross-connects two full-duplex model backends."""

import asyncio
import logging
import time

import numpy as np

from duplex.backend import FRAME_SIZE, SAMPLE_RATE, ModelBackend
from duplex.schemas import DuplexSegment

logger = logging.getLogger(__name__)

# Segmentation: flush text buffer after this many seconds of silence
SEGMENT_SILENCE_THRESHOLD = 1.0

END_TOKEN = "[END]"

_SILENCE_PCM = np.zeros(FRAME_SIZE, dtype=np.float32)


class DuplexRelay:
    """Cross-connects two model backends and captures their conversation.

    Tutor's audio output -> Student's audio input, and vice versa.
    Text tokens from both are captured for segmentation and live broadcast.

    For backends that need continuous input (e.g. PersonaPlex), the relay
    sends silence when no real audio is available from the other agent.
    """

    def __init__(
        self,
        tutor_backend: ModelBackend,
        student_backend: ModelBackend,
        max_duration: float = 300.0,
        silence_timeout: float = 10.0,
        on_token=None,
        on_segment=None,
        on_audio=None,
        on_interrupt=None,
        on_done=None,
    ):
        self._backends = {
            "tutor": tutor_backend,
            "student": student_backend,
        }
        self.max_duration = max_duration
        self.silence_timeout = silence_timeout

        # Callbacks for live updates (async callables)
        self._on_token = on_token
        self._on_segment = on_segment
        self._on_audio = on_audio  # on_audio(role, pcm_float32_bytes)
        self._on_interrupt = on_interrupt  # on_interrupt(role) — role was cut off
        self._on_done = on_done

        # State
        self._start_time: float = 0.0
        self._stop_event = asyncio.Event()
        self._segments: list[DuplexSegment] = []

        # PCM audio buffers for WAV recording
        self._audio_buffers: dict[str, bytearray] = {
            "tutor": bytearray(),
            "student": bytearray(),
        }
        # Native-format audio buffers (e.g. Opus bytes for PersonaPlex)
        self._native_audio_buffers: dict[str, bytearray] = {
            "tutor": bytearray(),
            "student": bytearray(),
        }

        # Per-agent audio queue: PCM from the OTHER agent to be sent as input
        self._audio_queues: dict[str, asyncio.Queue] = {
            "tutor": asyncio.Queue(),
            "student": asyncio.Queue(),
        }

        # Barge-in flush flags: set by _recv_from when a backend reports the
        # other agent spoke over it; consumed by _send_to to drop its local
        # metering buffer in addition to the queue.
        self._flush_send: dict[str, bool] = {"tutor": False, "student": False}

        # Per-agent text accumulator for segmentation
        self._text_buffers: dict[str, str] = {"tutor": "", "student": ""}
        self._text_first_time: dict[str, float] = {"tutor": 0.0, "student": 0.0}
        self._text_last_time: dict[str, float] = {"tutor": 0.0, "student": 0.0}

        # Track last activity for silence detection
        self._last_activity: float = 0.0

        # Track when audio recording started per role (relative to _start_time)
        self._audio_start: dict[str, float | None] = {
            "tutor": None,
            "student": None,
        }

    @property
    def elapsed(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    @property
    def segments(self) -> list[DuplexSegment]:
        return list(self._segments)

    @property
    def audio_buffers(self) -> dict[str, bytes]:
        """PCM float32 audio buffers for WAV saving."""
        return {k: bytes(v) for k, v in self._audio_buffers.items()}

    @property
    def native_audio_buffers(self) -> dict[str, bytes]:
        """Native-format audio buffers (e.g. Opus bytes)."""
        return {k: bytes(v) for k, v in self._native_audio_buffers.items()}

    @property
    def audio_start_offsets(self) -> dict[str, float]:
        """Seconds between conversation start and first audio frame per role."""
        return {k: (v if v is not None else 0.0) for k, v in self._audio_start.items()}

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self):
        """Signal the relay to stop."""
        self._stop_event.set()

    async def run(self) -> list[DuplexSegment]:
        """Connect both backends and relay audio until termination."""
        self._start_time = time.time()
        self._last_activity = self._start_time

        try:
            # Connect both backends concurrently
            logger.info("Connecting backends...")
            await asyncio.gather(
                self._backends["tutor"].connect(),
                self._backends["student"].connect(),
            )
            logger.info("Both backends connected. Starting relay.")

            # Run all tasks concurrently
            tasks = [
                asyncio.create_task(self._recv_from("tutor"), name="recv-tutor"),
                asyncio.create_task(self._recv_from("student"), name="recv-student"),
                asyncio.create_task(self._send_to("tutor"), name="send-tutor"),
                asyncio.create_task(self._send_to("student"), name="send-student"),
                asyncio.create_task(self._watchdog(), name="watchdog"),
            ]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            for task in done:
                exc = task.exception()
                if exc:
                    logger.error("Task %s failed: %s", task.get_name(), exc)

        finally:
            # Disconnect both backends
            for role in ("tutor", "student"):
                try:
                    await self._backends[role].disconnect()
                except Exception:
                    logger.exception("Error disconnecting %s backend", role)

        # Flush any remaining text buffers
        self._flush_segment("tutor")
        self._flush_segment("student")

        duration = time.time() - self._start_time
        logger.info(
            "Relay finished. Duration: %.1fs, Segments: %d",
            duration,
            len(self._segments),
        )

        if self._on_done:
            await self._on_done(duration)

        return self._segments

    def _truncate_recording_to_realtime(self, role: str) -> None:
        """Trim `role`'s recording buffers to their real-time length.

        Used on barge-in: cloud backends emit audio faster than real-time and
        append it into `_audio_buffers[role]` immediately, but only the bytes
        that have been metered through `_send_to(other)` by now have actually
        reached the listener. Any trailing audio represents cancelled output.
        """
        if self._audio_start[role] is None:
            return
        role_wall_start = self._start_time + self._audio_start[role]
        elapsed = time.time() - role_wall_start
        expected_samples = max(0, int(elapsed * SAMPLE_RATE))

        # WAV: float32, 4 bytes per sample.
        expected_wav_bytes = expected_samples * 4
        if len(self._audio_buffers[role]) > expected_wav_bytes:
            dropped = len(self._audio_buffers[role]) - expected_wav_bytes
            del self._audio_buffers[role][expected_wav_bytes:]
            logger.info(
                "[%s] barge-in: truncated WAV buffer by %d bytes (%.2fs)",
                role,
                dropped,
                dropped / 4 / SAMPLE_RATE,
            )

        # Native: only safe for raw PCM16 (GPT). Codec formats like Opus
        # can't be truncated mid-packet, so leave those alone.
        if self._backends[role].get_native_audio_ext() == ".raw":
            expected_native_bytes = expected_samples * 2  # int16
            if len(self._native_audio_buffers[role]) > expected_native_bytes:
                del self._native_audio_buffers[role][expected_native_bytes:]

    async def _recv_from(self, role: str):
        """Read events from a backend, queue audio for the other agent."""
        other = "student" if role == "tutor" else "tutor"
        backend = self._backends[role]

        async for event_type, data in backend.recv():
            if self._stop_event.is_set():
                break

            if event_type == "audio":
                pcm = data
                # Real-time producers (mic) can't pre-stream; any backlog is
                # pure delay on the listener, so drop stale frames.
                if backend.is_realtime_source:
                    while self._audio_queues[other].qsize() >= 2:
                        try:
                            self._audio_queues[other].get_nowait()
                        except asyncio.QueueEmpty:
                            break
                await self._audio_queues[other].put(pcm)
                now = time.time()
                if self._audio_start[role] is None:
                    self._audio_start[role] = now - self._start_time

                # Cloud backends emit response audio faster than real-time, so
                # pad this role's WAV buffer up to its real-time position
                # before appending. Otherwise the saved WAV has no inter-turn
                # gaps. Native buffer is left alone — raw codec bytes (e.g.
                # Opus) can't accept silence padding without corruption.
                role_wall_start = self._start_time + self._audio_start[role]
                expected_samples = int((now - role_wall_start) * SAMPLE_RATE)
                current_samples = len(self._audio_buffers[role]) // 4  # float32
                gap_samples = expected_samples - current_samples
                if gap_samples > SAMPLE_RATE // 10:  # ignore < 100 ms jitter
                    self._audio_buffers[role].extend(bytes(gap_samples * 4))

                pcm_bytes = pcm.tobytes()
                self._audio_buffers[role].extend(pcm_bytes)
                self._last_activity = now

                if self._on_audio is not None:
                    asyncio.ensure_future(self._on_audio(role, pcm_bytes))

            elif event_type == "native_audio":
                # Raw backend-native bytes for native-format recording
                self._native_audio_buffers[role].extend(data)

            elif event_type == "text":
                now = time.time() - self._start_time
                self._on_text_token(role, data, now)
                self._last_activity = time.time()

            elif event_type == "interrupt":
                # This backend's server cancelled its own response because the
                # other agent started talking. Drop any already-banked output
                # heading to the other agent so we don't keep playing over
                # them. Cloud backends generate faster than real-time, so this
                # buffer can be several seconds long.
                dropped_frames = 0
                while not self._audio_queues[other].empty():
                    try:
                        self._audio_queues[other].get_nowait()
                        dropped_frames += 1
                    except asyncio.QueueEmpty:
                        break
                self._flush_send[other] = True
                if dropped_frames:
                    logger.info(
                        "[%s] barge-in: dropped %d queued frames to %s",
                        role,
                        dropped_frames,
                        other,
                    )
                # Truncate this role's recording buffers to their real-time
                # position so interrupted/cancelled audio doesn't end up in
                # the saved WAV or native file. Without this the WAV plays
                # GPT's pre-cancellation overproduction that the listener
                # never actually heard.
                self._truncate_recording_to_realtime(role)
                # Notify listeners (e.g. the web UI) so they can flush any
                # audio for this role that is already scheduled client-side.
                if self._on_interrupt is not None:
                    asyncio.ensure_future(self._on_interrupt(role))

            elif event_type == "closed":
                logger.info("%s backend connection closed.", role)
                self._stop_event.set()
                break

    async def _send_to(self, role: str):
        """Send audio to a backend at real-time pace. Silence when queue is empty.

        Exactly FRAME_SIZE samples are sent every frame_interval seconds. Audio
        arriving from the other agent faster than real-time is held in a local
        buffer and metered out. This keeps the listener's server VAD and turn
        boundaries aligned with real-time conversation flow. ``next_tick`` is
        anchored to ``time.monotonic()`` so cadence stays wall-clock accurate
        regardless of per-iteration work time — drift against a real-time mic
        would otherwise accumulate in the downstream queue.
        """
        backend = self._backends[role]
        queue = self._audio_queues[role]
        frame_interval = FRAME_SIZE / SAMPLE_RATE
        buf = np.zeros(0, dtype=np.float32)
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            # Barge-in: the other side's backend signalled it cancelled its
            # response. _recv_from already drained the queue; here we drop
            # the local metering buffer too so no leftover frame goes out.
            if self._flush_send[role]:
                buf = np.zeros(0, dtype=np.float32)
                self._flush_send[role] = False

            # Drain queue into the local buffer without blocking.
            while not queue.empty():
                try:
                    buf = np.concatenate([buf, queue.get_nowait()])
                except asyncio.QueueEmpty:
                    break

            if len(buf) >= FRAME_SIZE:
                chunk = buf[:FRAME_SIZE]
                buf = buf[FRAME_SIZE:]
                await backend.send_audio(chunk)
                # Count drain as activity: cloud backends finish generating
                # well before the real-time playout completes, so without this
                # the silence watchdog fires mid-drain.
                self._last_activity = time.time()
            elif len(buf) > 0:
                # Partial leftover — pad with silence so we still send a full frame.
                chunk = np.concatenate(
                    [buf, np.zeros(FRAME_SIZE - len(buf), dtype=np.float32)]
                )
                buf = np.zeros(0, dtype=np.float32)
                await backend.send_audio(chunk)
                self._last_activity = time.time()
            else:
                await backend.send_audio(_SILENCE_PCM)

            next_tick += frame_interval
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            elif sleep_time < -2 * frame_interval:
                # Fell significantly behind (e.g. GC pause). Reset the anchor
                # to avoid bursting a backlog of frames to the backend.
                next_tick = time.monotonic()
            else:
                # Small lag: yield without sleeping, next iteration catches up.
                await asyncio.sleep(0)

    def _on_text_token(self, role: str, text: str, timestamp: float):
        """Handle an incoming text token — accumulate for segmentation."""
        # Check if we need to flush the previous segment (silence gap)
        if self._text_buffers[role] and (
            timestamp - self._text_last_time[role] > SEGMENT_SILENCE_THRESHOLD
        ):
            self._flush_segment(role)

        # Start new segment if buffer is empty
        if not self._text_buffers[role]:
            self._text_first_time[role] = timestamp

        self._text_buffers[role] += text
        self._text_last_time[role] = timestamp

        # Fire token callback
        if self._on_token:
            asyncio.ensure_future(self._on_token(role, text, timestamp))

        # Check for END token in the newly added text
        if END_TOKEN in text:
            logger.info("%s emitted [END] token.", role)
            self._stop_event.set()

    def _flush_segment(self, role: str):
        """Flush accumulated text buffer into a DuplexSegment."""
        text = self._text_buffers[role].strip()
        if not text:
            return

        segment = DuplexSegment(
            role=role,
            text=text,
            start_time=self._text_first_time[role],
            end_time=self._text_last_time[role],
        )
        self._segments.append(segment)
        self._text_buffers[role] = ""

        if self._on_segment:
            asyncio.ensure_future(self._on_segment(segment))

        logger.debug(
            "Segment [%s] %.1f-%.1fs: %s",
            role,
            segment.start_time,
            segment.end_time,
            text[:80],
        )

    async def _watchdog(self):
        """Monitor termination conditions."""
        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)
            now = time.time()
            elapsed = now - self._start_time

            # Time limit
            if elapsed >= self.max_duration:
                logger.info("Max duration reached (%.0fs).", self.max_duration)
                self._stop_event.set()
                break

            # Silence timeout
            if now - self._last_activity >= self.silence_timeout:
                logger.info("Silence timeout (%.0fs).", self.silence_timeout)
                self._stop_event.set()
                break

            # Periodically flush stale segments
            for role in ("tutor", "student"):
                if self._text_buffers[role] and (
                    elapsed - self._text_last_time[role] > SEGMENT_SILENCE_THRESHOLD
                ):
                    self._flush_segment(role)
