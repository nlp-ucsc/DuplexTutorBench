"""Conversation-level summary stats. No external deps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.base import Evaluator


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


class ConversationStats(Evaluator):
    name = "stats"

    def evaluate(self, conv: dict, audio_dir: Path) -> dict[str, Any]:
        segments = conv.get("segments", [])
        duration = float(conv.get("duration", 0.0)) or 0.0

        per_role: dict[str, dict[str, float]] = {
            "tutor": {"segments": 0, "words": 0, "speaking_time_s": 0.0},
            "student": {"segments": 0, "words": 0, "speaking_time_s": 0.0},
        }
        role_switches = 0
        prev_role: str | None = None
        for seg in segments:
            role = seg.get("role", "")
            if role not in per_role:
                continue
            per_role[role]["segments"] += 1
            per_role[role]["words"] += _word_count(seg.get("text", ""))
            seg_dur = max(
                0.0, float(seg.get("end_time", 0.0)) - float(seg.get("start_time", 0.0))
            )
            per_role[role]["speaking_time_s"] += seg_dur
            if prev_role is not None and role != prev_role:
                role_switches += 1
            prev_role = role

        out: dict[str, Any] = {
            "duration_s": round(duration, 3),
            "num_segments": len(segments),
            "role_switches": role_switches,
        }
        for role, m in per_role.items():
            wpm = (
                (m["words"] / (m["speaking_time_s"] / 60.0))
                if m["speaking_time_s"] > 0
                else 0.0
            )
            out[f"{role}_segments"] = int(m["segments"])
            out[f"{role}_words"] = int(m["words"])
            out[f"{role}_speaking_time_s"] = round(m["speaking_time_s"], 3)
            out[f"{role}_words_per_min"] = round(wpm, 1)
            out[f"{role}_talk_ratio"] = (
                round(m["speaking_time_s"] / duration, 3) if duration > 0 else 0.0
            )
        return out
