"""Audio loading + duration helpers."""

from __future__ import annotations

import wave
from pathlib import Path


def wav_duration_seconds(wav_path: Path) -> float:
    """Return duration in seconds, reading WAV header only (no decode)."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())
