"""Build segment dicts from word-level timestamps.

Each pipeline produces a list of words per pre-bounded segment (Whisper VAD
in whisper_mms, original duplex segments in text_mms). We further split
each list whenever the gap between consecutive words exceeds
`SEGMENT_SILENCE_THRESHOLD` — same value the live relay uses to close out
a segment.
"""

from __future__ import annotations

SEGMENT_SILENCE_THRESHOLD = 1.0  # seconds; same as duplex/relay.py


def words_to_segments(role: str, words: list[dict]) -> list[dict]:
    """Convert a word list into one or more output segments.

    Splits whenever the gap between consecutive words exceeds
    SEGMENT_SILENCE_THRESHOLD. Returns an empty list if `words` is empty.
    """
    if not words:
        return []
    groups: list[list[dict]] = [[words[0]]]
    for prev, cur in zip(words, words[1:]):
        if cur["start"] - prev["end"] > SEGMENT_SILENCE_THRESHOLD:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return [
        _make_segment(role, g, start=g[0]["start"], end=max(w["end"] for w in g))
        for g in groups
    ]


def _make_segment(role: str, words: list[dict], start: float, end: float) -> dict:
    text = " ".join(w["word"] for w in words).strip()
    return {
        "role": role,
        "text": text,
        "start_time": round(start, 3),
        "end_time": round(end, 3),
        "words": [
            {
                "word": w["word"],
                "start_time": round(w["start"], 3),
                "end_time": round(w["end"], 3),
                "confidence": round(float(w.get("score", 0.0)), 3),
            }
            for w in words
        ],
    }
