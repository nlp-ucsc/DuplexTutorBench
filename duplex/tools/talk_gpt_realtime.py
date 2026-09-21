"""Testing script: Talk to GPT Realtime with your microphone.

Full-duplex behavior:
  1. Server VAD with `interrupt_response: true` — the server cancels its own
     in-flight response as soon as it detects you start talking.
  2. On our side, we flush the local playback buffer on
     `input_audio_buffer.speech_started` so the model's voice cuts off in your
     ears immediately, not several hundred ms later after the already-buffered
     audio drains.

Run: uv run python duplex/tools/talk_gpt_realtime.py
Requires: pyaudio (uv add pyaudio), OPENAI_API_KEY in .env
Press Ctrl+C to quit.
"""

import asyncio
import base64
import os
import sys
import threading

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

SAMPLE_RATE = 24000
CHUNK_DURATION = 0.02  # 20ms @ 24kHz
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
BYTES_PER_SAMPLE = 2  # int16 mono

try:
    import pyaudio
except ImportError:
    print("Need pyaudio: uv add pyaudio (or pip install pyaudio)")
    sys.exit(1)


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY in .env")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    pa = pyaudio.PyAudio()

    # Shared playback buffer consumed by the output-stream callback. The
    # callback runs on a PortAudio thread, so guard the buffer with a lock.
    play_buf = bytearray()
    play_lock = threading.Lock()

    def spk_cb(in_data, frame_count, time_info, status):
        need = frame_count * BYTES_PER_SAMPLE
        with play_lock:
            chunk = bytes(play_buf[:need])
            del play_buf[: len(chunk)]
        if len(chunk) < need:
            chunk += b"\x00" * (need - len(chunk))
        return (chunk, pyaudio.paContinue)

    mic = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )
    spk = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SAMPLES,
        stream_callback=spk_cb,
    )

    print("Connecting to GPT Realtime...")
    async with client.realtime.connect(model="gpt-realtime") as conn:
        await conn.session.update(
            session={
                "type": "realtime",
                "instructions": (
                    "You are a chatty conversation partner. Give long, "
                    "multi-sentence answers so the user has room to interrupt."
                ),
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
        )
        # Wait for session.updated so config is in effect before audio flows.
        async for ev in conn:
            if ev.type == "session.updated":
                break

        print("Connected. Speak into your mic. Ctrl+C to quit.")
        print("Try interrupting the model mid-sentence to test full-duplex.\n")

        async def send_mic():
            """Read mic → send to model."""
            loop = asyncio.get_event_loop()
            while True:
                data = await loop.run_in_executor(None, mic.read, CHUNK_SAMPLES, False)
                b64 = base64.b64encode(data).decode("ascii")
                await conn.input_audio_buffer.append(audio=b64)

        async def recv_events():
            """Receive model events → buffer audio + print transcript."""
            printing = False
            async for ev in conn:
                t = ev.type

                if t == "response.output_audio.delta":
                    raw = base64.b64decode(ev.delta)
                    with play_lock:
                        play_buf.extend(raw)

                elif t == "response.output_audio_transcript.delta":
                    if not printing:
                        print("\nassistant: ", end="", flush=True)
                        printing = True
                    print(ev.delta, end="", flush=True)

                elif t == "response.output_audio_transcript.done":
                    print()
                    printing = False

                elif t == "conversation.item.input_audio_transcription.completed":
                    print(f"you: {getattr(ev, 'transcript', '').strip()}")

                elif t == "input_audio_buffer.speech_started":
                    # User just started talking. Drop any pending model audio
                    # so the voice cuts off in your ears right away.
                    with play_lock:
                        dropped = len(play_buf)
                        play_buf.clear()
                    if dropped:
                        print(f"\n[barge-in: flushed {dropped} bytes]")

                elif t in ("response.cancelled", "response.done"):
                    printing = False

                elif t == "error":
                    print(f"[error] {getattr(ev, 'error', ev)}", file=sys.stderr)

        try:
            await asyncio.gather(send_mic(), recv_events())
        finally:
            mic.stop_stream()
            mic.close()
            spk.stop_stream()
            spk.close()
            pa.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
