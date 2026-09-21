"""Approach B: per-original-segment forced alignment with <star> skip tokens.

We trust the duplex relay's silence-based segment boundaries (start_time /
end_time per role) but suspect the *text* in each segment is overlong
because barge-in cut the audio mid-utterance. So for each original segment,
we slice the matching audio range, feed the segment's text to
ctc-forced-aligner with `<star>` between every word, and keep only the
words the aligner actually matched to the audio.

Single pass, no Whisper. Faster than `whisper_mms`, but assumes the saved
text deltas are otherwise faithful (modulo trailing truncation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from pipeline_whisper_mms import (
    SR_ALIGNER,
    _load_aligner,
    _load_audio_16k_mono,
    _torch_dtype,
)

logger = logging.getLogger(__name__)


def align_role_segments(
    wav_path: Path,
    role_segments: list[dict],
    *,
    aligner_model: str,
    device: str,
    compute_dtype: str = "float16",
    iso_language: str = "eng",
) -> tuple[str, list[list[dict]]]:
    """Per-segment alignment with `<star>` between words.

    `role_segments` arrive on the conversation timeline; the per-role WAV is
    conversation-aligned (starts at t=0, length = conv.duration), so segment
    times index directly into the audio. The original segment `end_time` is
    text-event time (often very short for cloud backends that stream all text
    before the audio plays), so we extend each segment's audio slice to "until
    the next same-role segment starts" (or end of file). With `<star>` between
    every word, the aligner drops words that weren't actually spoken in that
    window.

    Returns (kept_text, segments_words[]) — segments_words[i] aligned to
    role_segments[i] on the conversation timeline.
    """
    if not role_segments:
        return "", []

    audio_np = _load_audio_16k_mono(wav_path)
    dtype = _torch_dtype(compute_dtype)
    model, tokenizer = _load_aligner(aligner_model, device, dtype)

    out_per_seg: list[list[dict]] = []
    kept_chunks: list[str] = []
    audio_dur = len(audio_np) / SR_ALIGNER

    for i, seg in enumerate(role_segments):
        text = (seg.get("text") or "").strip()
        if not text:
            out_per_seg.append([])
            continue
        seg_start = max(0.0, float(seg["start_time"]))
        seg_end = (
            max(0.0, float(role_segments[i + 1]["start_time"]))
            if i + 1 < len(role_segments)
            else audio_dur
        )
        seg_end = max(seg_start, seg_end)
        words = _align_segment(
            audio_np,
            seg_start,
            seg_end,
            text,
            model,
            tokenizer,
            iso_language,
        )
        out_per_seg.append(words)
        if words:
            kept_chunks.append(" ".join(w["word"] for w in words))

    return " ".join(kept_chunks).strip(), out_per_seg


def _align_segment(
    audio_np: np.ndarray,
    start_s: float,
    end_s: float,
    text: str,
    model,
    tokenizer,
    iso_language: str,
) -> list[dict]:
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        postprocess_results,
        preprocess_text,
    )

    if end_s <= start_s:
        return []
    s_idx = max(0, int(start_s * SR_ALIGNER))
    e_idx = min(len(audio_np), int(end_s * SR_ALIGNER))
    slice_np = audio_np[s_idx:e_idx]
    if slice_np.size < SR_ALIGNER // 10:
        return []

    audio_t = torch.from_numpy(slice_np).to(model.dtype).to(model.device)
    emissions, stride = generate_emissions(
        model, audio_t, window_length=30, context_length=2, batch_size=4
    )
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=iso_language,
        split_size="word",
        star_frequency="segment",  # <star> between every word -> skip un-spoken
    )
    try:
        segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, tokenizer
        )
    except Exception as e:
        logger.warning("get_alignments failed (%s) — skipping segment", e)
        return []
    spans = get_spans(tokens_starred, segments, blank_token)
    raw_words = postprocess_results(text_starred, spans, stride, scores)

    return [
        {
            "word": w["text"],
            "start": float(w["start"]) + start_s,
            "end": float(w["end"]) + start_s,
            "score": float(w.get("score", 0.0)),
        }
        for w in raw_words
        if w.get("text") and not w["text"].startswith("<star>")
    ]
