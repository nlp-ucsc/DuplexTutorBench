"""Evaluator ABC and result dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class Evaluator(ABC):
    """Compute one set of metrics for a single duplex conversation.

    Each concrete evaluator owns its own metric schema. The runner stores
    each evaluator's output dict under its `name` key so adding a new axis
    is a one-file change.
    """

    name: str = ""
    requires_audio: bool = False

    @abstractmethod
    def evaluate(self, conv: dict, audio_dir: Path) -> dict[str, Any]:
        """Return metrics for `conv` (raw JSONL dict, as loaded by duplex.storage.load_conversations).

        `audio_dir` is the per-conversation directory holding `tutor_full.wav`,
        `student_full.wav`, `combined.wav`, etc. Audio-free evaluators may ignore it.
        """


@dataclass
class EvaluationResult:
    """One row in `eval_output/{run_name}/scores.jsonl`."""

    pid: str
    conv_index: int
    timestamp: str
    evaluator_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
