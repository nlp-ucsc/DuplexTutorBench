"""Approach A: free ASR (faster-whisper) -> per-segment forced alignment.

We use faster-whisper to:
  1. produce the ground-truth transcript (the saved deltas may be longer than
     what was actually played due to barge-in),
  2. give us VAD-aware segment boundaries that encode true silences,

then use MahmoudAshraf/mms-300m-1130-forced-aligner per Whisper segment to
get crisper word timestamps. Whisper's intended-text bias (`initial_prompt`)
is fed in for proper-noun / math accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

SR_ALIGNER = 16000  # MMS aligner native rate

_FW_MODEL = None
_ALIGN_MODEL = None
_ALIGN_TOKENIZER = None


def _load_whisper(model_name: str, device: str, compute_type: str):
    global _FW_MODEL
    if _FW_MODEL is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper %s on %s (%s)", model_name, device, compute_type
        )
        _FW_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _FW_MODEL


def _load_aligner(model_name: str, device: str, dtype):
    global _ALIGN_MODEL, _ALIGN_TOKENIZER
    if _ALIGN_MODEL is None:
        from ctc_forced_aligner import load_alignment_model

        logger.info("Loading aligner %s on %s", model_name, device)
        _ALIGN_MODEL, _ALIGN_TOKENIZER = load_alignment_model(
            device, model_name, attn_implementation=None, dtype=dtype
        )
    return _ALIGN_MODEL, _ALIGN_TOKENIZER


def align_role(
    wav_path: Path,
    intended_text: str,
    *,
    whisper_model: str,
    aligner_model: str,
    device: str,
    compute_dtype: str = "float16",
    language: str = "en",
    iso_language: str = "eng",
    use_intended_prompt: bool = True,
) -> tuple[str, list[list[dict]]]:
    """Run ASR + per-segment forced alignment.

    When ``use_intended_prompt=True`` (default), the role's saved
    (overlong) text is fed to Whisper as ``initial_prompt`` for soft
    bias on math/proper nouns. When ``False``, no prompt is passed and
    Whisper runs purely from the audio — useful for measuring how much
    of the transcript is being shaped by the prompt.

    Returns (asr_text, segments_words[]) where each entry in
    segments_words is a list of {word, start, end, score} dicts on the
    conversation timeline (per-role WAV is conversation-aligned);
    segments are bounded by Whisper's VAD output.
    """
    fw = _load_whisper(whisper_model, device, compute_dtype)

    initial_prompt = _trim_prompt(intended_text) if use_intended_prompt else ""
    seg_iter, _info = fw.transcribe(
        str(wav_path),
        language=language,
        initial_prompt=initial_prompt or None,
        beam_size=5,
        vad_filter=True,
        # Tighten Silero VAD vs. faster-whisper defaults (pad=400ms,
        # min_silence=2000ms). The default 400ms speech pad gives the
        # forced aligner ~400ms of slack past the actual speech end —
        # on barge-in cuts that slack lets the last word's end_time
        # drift into the silent tail.
        vad_parameters={"speech_pad_ms": 50, "min_silence_duration_ms": 500},
        word_timestamps=False,
        condition_on_previous_text=False,
    )
    whisper_segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in seg_iter
        if s.text and s.text.strip()
    ]

    if not whisper_segments:
        return "", []

    asr_text = " ".join(s["text"] for s in whisper_segments).strip()

    # Load full audio once at 16kHz mono for the aligner.
    audio_np = _load_audio_16k_mono(wav_path)
    dtype = _torch_dtype(compute_dtype)
    model, tokenizer = _load_aligner(aligner_model, device, dtype)

    segments_words: list[list[dict]] = []
    for ws in whisper_segments:
        words = _align_segment(
            audio_np,
            ws["start"],
            ws["end"],
            ws["text"],
            model,
            tokenizer,
            iso_language,
        )
        segments_words.append(words)

    return asr_text, segments_words


def _align_segment(
    audio_np: np.ndarray,
    start_s: float,
    end_s: float,
    text: str,
    model,
    tokenizer,
    iso_language: str,
) -> list[dict]:
    """Forced-align `text` against audio[start_s:end_s], return word dicts in
    the original audio timeline."""
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        postprocess_results,
        preprocess_text,
    )

    if not text.strip() or end_s <= start_s:
        return []

    s_idx = max(0, int(start_s * SR_ALIGNER))
    e_idx = min(len(audio_np), int(end_s * SR_ALIGNER))
    slice_np = audio_np[s_idx:e_idx]
    # Skip sub-500ms segments. Below this, the MMS forced_align_cpp
    # extension corrupts glibc's heap on near-degenerate inputs (path
    # collapses a real token into <star>), which then aborts the
    # process at the next malloc with "corrupted size vs. prev_size".
    # The earlier T <= L + N_repeat guard catches the strict-segfault
    # mode but not this softer corruption mode — observed on a 0.44s
    # "You're welcome." segment.
    if slice_np.size < SR_ALIGNER // 2:
        logger.warning(
            "Skipping segment %.2fs-%.2fs: shorter than 0.5s (text=%r)",
            start_s,
            end_s,
            text[:60],
        )
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
        star_frequency="segment",
    )
    # forced_align_cpp segfaults (not a Python exception — kills the
    # process) when the emissions time dim T is not strictly greater
    # than the expanded label length L + N_repeat. The docstring states
    # T >= L + N_repeat but empirically the equality case also crashes
    # (e.g. T=L=8 on a 0.44s "Is that..." segment). Skip cleanly.
    char_tokens = " ".join(tokens_starred).split(" ")
    n_repeat = sum(1 for a, b in zip(char_tokens, char_tokens[1:]) if a == b)
    if emissions.shape[0] <= len(char_tokens) + n_repeat:
        logger.warning(
            "Skipping segment %.2fs-%.2fs: emissions too short for "
            "forced alignment (T=%d, L=%d, repeats=%d, text=%r)",
            start_s,
            end_s,
            emissions.shape[0],
            len(char_tokens),
            n_repeat,
            text[:60],
        )
        return []
    # ctc_forced_aligner can fail in two distinct ways past this point:
    #   1. get_alignments() can raise from preprocess/tokenizer issues.
    #   2. forced_align_cpp can return a path that doesn't visit every
    #      target token (collapsing some into <star>), and then
    #      get_spans() trips its `seg.label == ltr` assertion. The
    #      length pre-check above prevents the segfault mode but not
    #      this one — empirically it still hits on a few segments.
    # Skip the segment cleanly in either case.
    try:
        segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, tokenizer
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        raw_words = postprocess_results(text_starred, spans, stride, scores)
    except Exception as e:  # includes AssertionError from get_spans
        logger.warning(
            "alignment failed for segment %.2fs-%.2fs (%s: %s) — skipping",
            start_s,
            end_s,
            type(e).__name__,
            e,
        )
        return []

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


def _load_audio_16k_mono(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR_ALIGNER:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_ALIGNER)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _trim_prompt(text: str, max_chars: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _torch_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
