"""Save and load full-duplex conversations."""

import json
import logging
import wave
from pathlib import Path

import numpy as np

from duplex.backend import SAMPLE_RATE, pcm_float32_to_int16
from duplex.schemas import DuplexConversation

logger = logging.getLogger(__name__)


def save_conversation(conversation: DuplexConversation, output_path: Path) -> None:
    """Append a DuplexConversation as a JSON line to the output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(conversation.to_dict()) + "\n")
    logger.info("Saved conversation pid=%s to %s", conversation.pid, output_path)


def save_audio(audio_bytes: bytes, path: Path) -> None:
    """Save raw audio bytes to a file (native format, e.g. Opus or raw PCM16)."""
    if not audio_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_bytes)
    logger.info("Saved audio (%d bytes) to %s", len(audio_bytes), path)


def _pad_channel(
    pcm_float32_bytes: bytes,
    offset_s: float,
    total_duration_s: float,
    sample_rate: int,
) -> np.ndarray:
    """Place float32 PCM into a fixed-length silence buffer starting at offset_s.

    The output has exactly `round(total_duration_s * sample_rate)` samples. Audio
    is written starting at sample `round(offset_s * sample_rate)`; anything that
    would overrun the end is truncated. Empty input returns pure silence.
    """
    n_total = int(round(total_duration_s * sample_rate))
    out = np.zeros(n_total, dtype=np.float32)
    if not pcm_float32_bytes or n_total == 0:
        return out
    pcm = np.frombuffer(pcm_float32_bytes, dtype=np.float32)
    start = max(0, int(round(offset_s * sample_rate)))
    end = min(n_total, start + pcm.size)
    if end > start:
        out[start:end] = pcm[: end - start]
    return out


def save_audio_wav(
    pcm_float32_bytes: bytes,
    path: Path,
    total_duration_s: float,
    offset_s: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Save float32 PCM audio as a mono WAV aligned to conversation t=0.

    The output is exactly `total_duration_s` long; `offset_s` leading silence
    precedes the audio and any remaining tail is silence-padded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    channel = _pad_channel(pcm_float32_bytes, offset_s, total_duration_s, sample_rate)
    int16_data = pcm_float32_to_int16(channel).tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data)

    logger.info("Saved WAV audio (%d samples) to %s", channel.size, path)


def save_audio_wav_stereo(
    left_float32_bytes: bytes,
    right_float32_bytes: bytes,
    path: Path,
    total_duration_s: float,
    left_offset_s: float = 0.0,
    right_offset_s: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Save two float32 PCM streams as a single 2-channel WAV aligned to t=0.

    Both channels are placed into a fixed-length silence buffer of exactly
    `total_duration_s` so the stereo file matches the per-role mono files.
    If both inputs are empty, skip writing — matches the per-role savers'
    behavior of not producing pure-silence files for empty conversations.
    """
    if not left_float32_bytes and not right_float32_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    left = _pad_channel(
        left_float32_bytes, left_offset_s, total_duration_s, sample_rate
    )
    right = _pad_channel(
        right_float32_bytes, right_offset_s, total_duration_s, sample_rate
    )
    stereo = np.stack([left, right], axis=1)
    int16_data = pcm_float32_to_int16(stereo.reshape(-1)).tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data)

    logger.info("Saved stereo WAV audio (%d samples) to %s", left.size, path)


def load_conversations(jsonl_path: Path) -> list[dict]:
    """Load all conversations from a JSONL file."""
    if not jsonl_path.is_file():
        return []
    conversations = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", line_num, jsonl_path)
    return conversations


def load_done_set(jsonl_path: Path) -> set[tuple[str, int]]:
    """Return the set of (pid, attempt_index) pairs already in conversations.jsonl."""
    done: set[tuple[str, int]] = set()
    for conv in load_conversations(jsonl_path):
        pid = conv.get("pid")
        if pid is None:
            continue
        attempt = int(conv.get("attempt_index", 0))
        done.add((str(pid), attempt))
    return done


def next_attempt_index(jsonl_path: Path, pid: str) -> int:
    """Return the next free attempt_index for a given pid."""
    highest = -1
    for conv in load_conversations(jsonl_path):
        if str(conv.get("pid")) != str(pid):
            continue
        highest = max(highest, int(conv.get("attempt_index", 0)))
    return highest + 1
