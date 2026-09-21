"""Approach C: free ASR (faster-whisper) -> per-segment MFA forced alignment.

Like `pipeline_whisper_mms`, we use faster-whisper to produce the transcript
plus VAD-aware segment boundaries. Then, instead of feeding each segment
to the MMS-CTC aligner (which folds silence into edge-word boundaries),
we materialize each Whisper segment as a `seg_<i>.wav` + `seg_<i>.lab`
pair under a per-call corpus dir and invoke Montreal Forced Aligner once
for the whole role. MFA's Kaldi HMM-GMM model with explicit silence
phones produces tighter word boundaries (silence is not absorbed into
adjacent words).

MFA itself lives in a separate conda env at `~/miniconda3/envs/mfa-env/`
and is invoked as a subprocess. The calling Python (this file's env) only
needs `praatio` for parsing the TextGrid output back into our schema.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pipeline_whisper_mms import (
    SR_ALIGNER,
    _load_audio_16k_mono,
    _load_whisper,
    _trim_prompt,
)

logger = logging.getLogger(__name__)

MFA_BIN = Path.home() / "miniconda3" / "envs" / "mfa-env" / "bin" / "mfa"
DEFAULT_MFA_ROOT = Path.home() / "Documents" / "MFA"
DEFAULT_ACOUSTIC = "english_mfa"
DEFAULT_DICTIONARY = "english_mfa"

# MFA's `words` tier emits these labels for non-speech intervals.
_MFA_SILENCE_LABELS = {"", "sil", "sp", "spn"}

# faster-whisper with `vad_filter=True` compacts speech internally then
# re-projects segment timestamps back to the original timeline. The remap
# means a single emitted "segment" can span a long silent gap on the
# original timeline (whenever Whisper didn't insert a segment boundary
# inside the compacted speech). MFA per-utterance alignment can't handle
# such spans — the audio slice contains long silence with text that
# describes only the speech, and MFA fails to align. We use Whisper's
# per-word timestamps to detect intra-segment gaps and split before
# handing to MFA.
MAX_INTRA_SEGMENT_GAP = 1.5  # seconds; split a Whisper seg on word gaps wider than this


def align_role(
    wav_path: Path,
    intended_text: str,
    *,
    whisper_model: str,
    aligner_model: str,  # accepted for signature parity; unused by MFA
    device: str,  # MFA is CPU-only; only Whisper uses this
    compute_dtype: str = "float16",
    language: str = "en",
    iso_language: str = "eng",  # accepted for parity; MFA uses its own model
    use_intended_prompt: bool = True,
    mfa_acoustic_model: str = DEFAULT_ACOUSTIC,
    mfa_dictionary: str = DEFAULT_DICTIONARY,
    num_jobs: int = 12,
) -> tuple[str, list[list[dict]]]:
    """Run faster-whisper -> per-segment MFA forced alignment.

    Returns (asr_text, segments_words[]) on the conversation timeline,
    with the same {word, start, end, score} schema as
    `pipeline_whisper_mms.align_role`. `score` is a constant 1.0 because
    MFA's TextGrid output does not carry per-word confidence; can be
    revisited later via `--output_format json` + phone-posterior
    aggregation.
    """
    fw = _load_whisper(whisper_model, device, compute_dtype)

    initial_prompt = _trim_prompt(intended_text) if use_intended_prompt else ""
    seg_iter, _info = fw.transcribe(
        str(wav_path),
        language=language,
        initial_prompt=initial_prompt or None,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"speech_pad_ms": 50, "min_silence_duration_ms": 500},
        word_timestamps=True,  # need per-word times to re-split on intra-segment silences
        condition_on_previous_text=False,
    )
    raw_segments = [
        {
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip(),
            "words": [
                {"word": w.word, "start": float(w.start), "end": float(w.end)}
                for w in (s.words or [])
            ],
        }
        for s in seg_iter
        if s.text and s.text.strip()
    ]

    if not raw_segments:
        return "", []

    asr_text = " ".join(s["text"] for s in raw_segments).strip()

    # Split each Whisper segment wherever a word-to-word gap exceeds
    # MAX_INTRA_SEGMENT_GAP. Inside-utterance silences are kept; cross-
    # utterance silences become segment boundaries. Each entry below
    # corresponds to one MFA utterance.
    sub_segments = _split_on_word_gaps(raw_segments, MAX_INTRA_SEGMENT_GAP)
    if not sub_segments:
        return asr_text, []

    audio_np = _load_audio_16k_mono(wav_path)
    audio_dur = len(audio_np) / SR_ALIGNER

    with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as scratch_root:
        corpus_dir = Path(scratch_root) / "corpus"
        out_dir = Path(scratch_root) / "out"
        corpus_dir.mkdir(parents=True)

        seg_meta: list[Optional[dict]] = []
        for i, ss in enumerate(sub_segments):
            seg_meta.append(
                _write_segment(
                    audio_np,
                    ss["start"],
                    ss["end"],
                    ss["text"],
                    corpus_dir,
                    i,
                    audio_dur,
                )
            )

        if not any(m is not None for m in seg_meta):
            return asr_text, [[] for _ in sub_segments]

        rc = _run_whisper_mfa(
            corpus_dir,
            out_dir,
            acoustic=mfa_acoustic_model,
            dictionary=mfa_dictionary,
            num_jobs=num_jobs,
        )
        if rc != 0:
            logger.warning(
                "mfa align exited non-zero (rc=%d) for %s — returning empty alignments",
                rc,
                wav_path,
            )
            return asr_text, [[] for _ in sub_segments]

        segments_words: list[list[dict]] = []
        for i, meta in enumerate(seg_meta):
            if meta is None:
                segments_words.append([])
                continue
            tg_path = out_dir / f"seg_{i:04d}.TextGrid"
            if not tg_path.is_file():
                logger.warning(
                    "MFA produced no TextGrid for seg %d (%.2fs-%.2fs, text=%r)",
                    i,
                    meta["start"],
                    meta["end"],
                    meta["text"][:60],
                )
                segments_words.append([])
                continue
            try:
                words = _parse_textgrid(tg_path, offset=meta["start"])
            except Exception as e:
                logger.warning(
                    "TextGrid parse failed for seg %d (%s: %s) — skipping",
                    i,
                    type(e).__name__,
                    e,
                )
                words = []
            segments_words.append(words)

    return asr_text, segments_words


def _split_on_word_gaps(raw_segments: list[dict], max_gap_s: float) -> list[dict]:
    """Re-split Whisper segments wherever consecutive words are separated
    by more than `max_gap_s` seconds.

    Returns a flat list of sub-segments with {start, end, text} on the
    original audio timeline. If a Whisper segment has no per-word
    timestamps (e.g. faster-whisper returned None for `.words`), it is
    kept as-is — we don't have a basis for splitting it.
    """
    out: list[dict] = []
    for seg in raw_segments:
        words = seg.get("words") or []
        if not words:
            out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            continue
        groups: list[list[dict]] = [[words[0]]]
        for prev, cur in zip(words, words[1:]):
            if cur["start"] - prev["end"] > max_gap_s:
                groups.append([cur])
            else:
                groups[-1].append(cur)
        for g in groups:
            text = "".join(w["word"] for w in g).strip()
            if not text:
                continue
            out.append(
                {
                    "start": float(g[0]["start"]),
                    "end": float(g[-1]["end"]),
                    "text": text,
                }
            )
    return out


def _write_segment(
    audio_np: np.ndarray,
    start_s: float,
    end_s: float,
    text: str,
    corpus_dir: Path,
    i: int,
    audio_dur: float,
) -> Optional[dict]:
    if not text.strip():
        return None
    start_s = max(0.0, start_s)
    end_s = min(audio_dur, end_s)
    # MFA's feature extraction needs at least ~10 frames (frame_shift=10ms).
    if end_s - start_s < 0.1:
        logger.warning(
            "Skipping seg %d: too short (%.3fs-%.3fs, text=%r)",
            i,
            start_s,
            end_s,
            text[:60],
        )
        return None
    s_idx = max(0, int(start_s * SR_ALIGNER))
    e_idx = min(len(audio_np), int(end_s * SR_ALIGNER))
    slice_np = audio_np[s_idx:e_idx]
    if slice_np.size < SR_ALIGNER // 10:
        return None

    stem = f"seg_{i:04d}"
    sf.write(str(corpus_dir / f"{stem}.wav"), slice_np, SR_ALIGNER, subtype="PCM_16")
    (corpus_dir / f"{stem}.lab").write_text(text.strip(), encoding="utf-8")
    return {"start": start_s, "end": end_s, "text": text.strip()}


def _run_whisper_mfa(
    corpus_dir: Path,
    out_dir: Path,
    *,
    acoustic: str,
    dictionary: str,
    num_jobs: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # NB: `--beam` / `--retry_beam` are NOT exposed as CLI flags in MFA 3.x
    # (they moved to `--config_path` YAML). Default beam/retry_beam are 10/40.
    # Per-role failure (NoAlignmentsError when none of a role's utterances
    # align under those defaults) is isolated by the caller — the other role
    # still produces output.
    cmd = [
        str(MFA_BIN),
        "align",
        "--clean",
        "--single_speaker",
        "--num_jobs",
        str(num_jobs),
        "--output_format",
        "long_textgrid",
        "--quiet",
        str(corpus_dir),
        dictionary,
        acoustic,
        str(out_dir),
    ]
    logger.info("$ %s", " ".join(cmd))
    # MFA shells out to fstcompile/openfst/kaldi binaries by name (no absolute
    # path), so the conda env's bin/ must be first on PATH for those lookups.
    env = os.environ.copy()
    env["PATH"] = f"{MFA_BIN.parent}{os.pathsep}{env.get('PATH', '')}"

    # Give this subprocess its own MFA root dir. The default
    # `~/Documents/MFA/` contains `command_history.yaml` which MFA loads
    # and rewrites in an atexit hook — concurrent runs (parallel tutor +
    # student) race on that file and one of them crashes with a YAML
    # ScannerError. We symlink `pretrained_models/` so the english_mfa
    # acoustic + dictionary are still found; everything else (joblib
    # cache, extracted models, command history) lives privately under
    # the temp root.
    with tempfile.TemporaryDirectory(prefix="mfa_root_") as mfa_root:
        mfa_root_p = Path(mfa_root)
        models_src = DEFAULT_MFA_ROOT / "pretrained_models"
        if models_src.is_dir():
            (mfa_root_p / "pretrained_models").symlink_to(models_src)
        env["MFA_ROOT_DIR"] = str(mfa_root_p)
        proc = subprocess.run(cmd, check=False, env=env)
        return proc.returncode


def _parse_textgrid(tg_path: Path, *, offset: float) -> list[dict]:
    """Parse MFA TextGrid -> [{word, start, end, score}] on conv timeline.

    MFA emits 'words' and 'phones' tiers. We read 'words', drop silence
    labels, and shift each interval by `offset` so times are absolute on
    the conversation timeline.
    """
    from praatio import textgrid

    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    if "words" not in tg.tierNames:
        return []
    tier = tg.getTier("words")
    out: list[dict] = []
    for interval in tier.entries:
        label = (interval.label or "").strip()
        if not label or label.lower() in _MFA_SILENCE_LABELS:
            continue
        out.append(
            {
                "word": label,
                "start": float(interval.start) + offset,
                "end": float(interval.end) + offset,
                "score": 1.0,
            }
        )
    return out
