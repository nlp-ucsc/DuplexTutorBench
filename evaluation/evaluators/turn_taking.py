"""Turn-taking metrics computed from segment timestamps.

Inspired by Full-Duplex-Bench (Lin et al., 2025) — simplified proxies of
its dimensions, but computable purely from the segment list we already
store in conversations.jsonl.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from evaluation.base import Evaluator

# A "backchannel" is short and from the role that isn't currently holding the floor.
_BACKCHANNEL_MAX_SEC = 1.5
_BACKCHANNEL_MAX_WORDS = 4


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _total_overlap(
    intervals_a: list[tuple[float, float]], intervals_b: list[tuple[float, float]]
) -> float:
    """Sum of overlap durations between two sets of intervals."""
    total = 0.0
    for a0, a1 in intervals_a:
        for b0, b1 in intervals_b:
            o = min(a1, b1) - max(a0, b0)
            if o > 0:
                total += o
    return total


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    s = sorted(intervals)
    out = [s[0]]
    for a, b in s[1:]:
        last_a, last_b = out[-1]
        if a <= last_b:
            out[-1] = (last_a, max(last_b, b))
        else:
            out.append((a, b))
    return out


class TurnTaking(Evaluator):
    name = "turn_taking"

    def evaluate(self, conv: dict, audio_dir: Path) -> dict[str, Any]:
        segments = list(conv.get("segments", []))
        duration = float(conv.get("duration", 0.0)) or 0.0
        # Sort by start time for sequential analysis
        segments.sort(key=lambda s: float(s.get("start_time", 0.0)))

        tutor_iv = [
            (float(s["start_time"]), float(s["end_time"]))
            for s in segments
            if s.get("role") == "tutor"
        ]
        student_iv = [
            (float(s["start_time"]), float(s["end_time"]))
            for s in segments
            if s.get("role") == "student"
        ]

        # Response latencies: gap between the end of one role's segment and the
        # start of the *next* segment from the *other* role. Negative gaps
        # (overlap) are excluded; those are reported separately as overlap.
        latencies: list[float] = []
        for a, b in zip(segments, segments[1:]):
            if a.get("role") == b.get("role"):
                continue
            gap = float(b["start_time"]) - float(a["end_time"])
            if gap >= 0:
                latencies.append(gap)

        # Overlap between the two roles in seconds (fraction of total duration).
        overlap_s = _total_overlap(tutor_iv, student_iv)

        # Silence: fraction of duration where no role is speaking.
        merged_all = _merge_intervals(tutor_iv + student_iv)
        speaking_s = sum(b - a for a, b in merged_all)
        silence_s = max(0.0, duration - speaking_s)

        # Backchannels: short, low-word-count segments from a role while the
        # *other* role's segment is still in progress (i.e., overlapping start).
        backchannels = 0
        for s in segments:
            seg_dur = float(s.get("end_time", 0.0)) - float(s.get("start_time", 0.0))
            if seg_dur > _BACKCHANNEL_MAX_SEC:
                continue
            if _word_count(s.get("text", "")) > _BACKCHANNEL_MAX_WORDS:
                continue
            other_iv = student_iv if s.get("role") == "tutor" else tutor_iv
            seg_start = float(s.get("start_time", 0.0))
            seg_end = float(s.get("end_time", 0.0))
            if any(a < seg_end and b > seg_start for a, b in other_iv):
                backchannels += 1

        n_segments = len(segments) or 1
        return {
            "response_latency_s": {
                "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
                "median": round(statistics.median(latencies), 3) if latencies else 0.0,
                "p90": round(_percentile(latencies, 0.9), 3),
                "count": len(latencies),
            },
            "overlap_s": round(overlap_s, 3),
            "overlap_ratio": round(overlap_s / duration, 3) if duration > 0 else 0.0,
            "silence_s": round(silence_s, 3),
            "silence_ratio": round(silence_s / duration, 3) if duration > 0 else 0.0,
            "backchannel_count": backchannels,
            "backchannel_ratio": round(backchannels / n_segments, 3),
        }
