"""Dataclasses for MathVista questions and conversation transcripts."""

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class MathVistaQuestion:
    """A single question from the MathVista dataset."""

    pid: str
    question: str
    answer: str
    question_type: str
    answer_type: str
    choices: list[str]
    image: Image.Image | None
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Utterance:
    """A single utterance in a conversation."""

    role: str  # "student" or "tutor"
    content: str
    turn: int
    audio_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {"role": self.role, "content": self.content, "turn": self.turn}
        if self.audio_path:
            d["audio_path"] = self.audio_path
        return d


@dataclass
class Conversation:
    """A complete simulated conversation about a MathVista question."""

    pid: str
    question: str
    answer: str
    tutor_model: str = ""
    student_model: str = ""
    image_path: str = ""
    utterances: list[Utterance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.utterances)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pid": self.pid,
            "question": self.question,
            "answer": self.answer,
            "tutor_model": self.tutor_model,
            "student_model": self.student_model,
            "utterances": [u.to_dict() for u in self.utterances],
            "num_turns": self.num_turns,
            "metadata": self.metadata,
        }
        if self.image_path:
            d["image_path"] = self.image_path
        return d
