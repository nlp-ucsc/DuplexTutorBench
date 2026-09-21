"""Dataclasses for full-duplex conversation transcripts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DuplexSegment:
    """A continuous speech segment from one agent (silence-delimited)."""

    role: str  # "tutor" or "student"
    text: str
    start_time: float  # Seconds since conversation start
    end_time: float
    audio_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "role": self.role,
            "text": self.text,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
        }
        if self.audio_path:
            d["audio_path"] = self.audio_path
        return d


@dataclass
class DuplexConversation:
    """A full-duplex conversation between tutor and student."""

    pid: str
    question: str
    answer: str
    tutor_voice: str
    student_voice: str
    duration: float
    segments: list[DuplexSegment] = field(default_factory=list)
    image_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tutor_backend: str = ""
    student_backend: str = ""
    tutor_uses_image: bool = False
    student_uses_image: bool = False
    attempt_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pid": self.pid,
            "attempt_index": self.attempt_index,
            "question": self.question,
            "answer": self.answer,
            "tutor_voice": self.tutor_voice,
            "student_voice": self.student_voice,
            "tutor_backend": self.tutor_backend,
            "student_backend": self.student_backend,
            "tutor_uses_image": self.tutor_uses_image,
            "student_uses_image": self.student_uses_image,
            "duration": round(self.duration, 3),
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata,
        }
        if self.image_path:
            d["image_path"] = self.image_path
        return d
