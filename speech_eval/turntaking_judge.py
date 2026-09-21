"""ESPnet `Turn_taking_prediction_SWBD` judge over duplex conversations.

Loads the released HF checkpoint via `espnet_model_zoo` and slides a
fixed-size context window across the mono mixdown of each conversation.
Each call returns one row of 5-class probabilities per ~40 ms encoder frame;
we record those plus the greedy label per frame.

Writes ``speech_eval_output/{run_name}/turntaking_judge.jsonl``.

Implementation notes
--------------------

* The ESPnet config used at training time has ``ctc_weight: 0.0`` and no
  decoder beyond the SLU classifier head, so the default ``Speech2Understand``
  construction crashes inside the beam-search CTC scorer. We construct with
  ``ctc_weight=0.0`` and ``beam_size=1`` to match the recipe's
  ``conf/decode_asr_chunk.yaml`` (with ``ctc_weight`` clamped down because the
  model has no CTC head).
* The newer ``openai-whisper`` package (>= 20250625) removed the public
  ``N_MELS`` constant that ESPnet's ``whisper_encoder.py`` imports. We
  monkey-patch ``whisper.audio.N_MELS = 80`` before any ESPnet import. 80 is
  the correct value for the Whisper-medium encoder this checkpoint uses.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# Per-frame label order in the model's output. The ESPnet token list (with
# special tokens stripped) is ['C', 'NA', 'I', 'BC', 'T'].
LABELS = ("C", "NA", "I", "BC", "T")

JUDGE_MODEL_TAG = "espnet/Turn_taking_prediction_SWBD"
JUDGE_SR = 16_000
JUDGE_CONTEXT_S = 30.0  # paper's W; also the model's encoder receptive field
FRAME_S = 0.04  # one prediction per 40 ms (matches paper)
# ESPnet `Speech2Understand.__call__` (run_chunk=True) iterates
# `range((speech.size(1) - start_chunk) // sim_chunk_length)` with defaults
# `start_chunk=3200` and `sim_chunk_length=640`. A chunk smaller than the sum
# skips the loop body entirely and crashes on the post-loop
# `Hypothesis(yseq=torch.tensor(token_int_corr))` with `UnboundLocalError`.
# Skip such tail windows — < 240 ms is well under one 40 ms output frame's
# context anyway.
SLU_MIN_SAMPLES = 3840


def _patch_whisper_n_mels() -> None:
    """Restore the ``N_MELS`` constant that newer openai-whisper drops."""
    import whisper.audio

    if not hasattr(whisper.audio, "N_MELS"):
        whisper.audio.N_MELS = 80


@dataclass
class JudgeFrames:
    conv_index: int
    window_start_s: float
    window_end_s: float
    # length-T list of 5-floats (probabilities) in LABELS order
    frame_probs: list[list[float]]
    # length-T list of argmax labels
    frame_labels: list[str]


def _resolve_device(device: str) -> str:
    """Resolve "auto" to a concrete device string.

    Auto priority is cuda -> mps -> cpu. CUDA wins on Linux/GPU boxes;
    MPS is the fallback on Apple Silicon, empirically verified
    label-for-label equal to CPU on this checkpoint (probs within 1e-4)
    for a ~3.7x speedup over CPU.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError(
            f"device must be 'auto', 'cpu', 'mps', or 'cuda'; got {device!r}"
        )
    return device


def _build_predictor(device: str):
    """Construct a Speech2Understand instance configured for this checkpoint.

    ``device`` is one of "auto", "cpu", "mps", or "cuda". For "mps" we set
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so ops without MPS kernels (e.g.
    ``aten::repeat_interleave``) fall back to CPU rather than crashing.
    For "cuda" we don't touch that env var.

    Returns ``(predictor, kwargs, resolved_device)`` where
    ``resolved_device`` is the concrete device string actually used.
    """
    resolved = _resolve_device(device)
    if resolved == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS not available on this machine")
    elif resolved == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on this machine")

    _patch_whisper_n_mels()
    from espnet2.bin.slu_inference import Speech2Understand
    from espnet_model_zoo.downloader import ModelDownloader

    d = ModelDownloader()
    info = d.download_and_unpack(JUDGE_MODEL_TAG)
    # `download_and_unpack` returns asr_* keys; the SLU constructor takes slu_*.
    kwargs = {k.replace("asr_", "slu_"): v for k, v in info.items()}

    predictor = Speech2Understand(
        device=resolved,
        beam_size=1,
        ctc_weight=0.0,
        lm_weight=0.0,
        penalty=0.0,
        maxlenratio=0.0,
        minlenratio=0.0,
        run_chunk=True,
        **kwargs,
    )
    return predictor, kwargs, resolved


def _parse_result(result_tuple) -> tuple[list[list[float]], list[str]]:
    """Parse a single Speech2Understand result tuple into per-frame outputs.

    The token field is a string of space-separated frames, each frame being
    comma-separated 5-float probabilities in LABELS order.
    """
    # result_tuple is (text, token, token_int, hyp). The first element is
    # the joined whitespace string we saw in our latency test.
    text = result_tuple[0]
    frames_str = [f for f in text.split(" ") if f.strip()]
    frame_probs: list[list[float]] = []
    frame_labels: list[str] = []
    for fr in frames_str:
        try:
            probs = [float(x) for x in fr.split(",")]
        except ValueError:
            continue
        if len(probs) != len(LABELS):
            continue
        frame_probs.append(probs)
        frame_labels.append(LABELS[max(range(len(LABELS)), key=lambda i: probs[i])])
    return frame_probs, frame_labels


def _load_mono_16k(path: Path) -> torch.Tensor:
    import torchaudio.functional as F

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    waveform = torch.from_numpy(data).unsqueeze(0)
    if sr != JUDGE_SR:
        waveform = F.resample(waveform, sr, JUDGE_SR)
    return waveform.squeeze(0).contiguous()


def _audio_path(duplex_root: Path, run_name: str, conv_index: int) -> Path:
    return duplex_root / run_name / "audio" / str(conv_index) / "combined.wav"


def score_run(
    run_name: str,
    *,
    duplex_root: Path,
    out_dir: Path,
    conv_indices: list[int] | None = None,
    overwrite: bool = False,
    device: str = "auto",
    max_windows: int | None = None,
) -> tuple[Path, str]:
    """Run the judge on every conversation under ``duplex_root/run_name``.

    Parameters
    ----------
    max_windows
        Cap on non-overlapping 30 s windows per conversation. ``None``
        (default) processes the full conversation. A positive integer
        limits how many windows are processed (e.g. ``1`` = first 30 s
        only, useful for fast smoke tests on CPU). CPU inference is
        ~7 min per window, MPS ~2 min.
    device
        "auto" (default) picks mps if available else cpu; "cpu" / "mps" force
        a specific device.

    Returns
    -------
    (output_path, resolved_device)
    """
    out_path = out_dir / "turntaking_judge.jsonl"
    existing: dict[int, dict] = {}
    if out_path.is_file() and not overwrite:
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                existing[int(row["conv_index"])] = row

    audio_root = duplex_root / run_name / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"No audio dir at {audio_root}")
    all_indices = sorted(
        int(p.name) for p in audio_root.iterdir() if p.is_dir() and p.name.isdigit()
    )
    if conv_indices is not None:
        all_indices = [i for i in all_indices if i in set(conv_indices)]

    todo = [i for i in all_indices if overwrite or i not in existing]
    if not todo:
        # Even with nothing to do, we still want to surface what device a
        # rerun would have used.
        return out_path, _resolve_device(device)

    logger.info(
        "TurnTakingJudge: %d conversations to score (device=%s, max_windows=%s)",
        len(todo),
        device,
        max_windows,
    )

    predictor, kwargs_used, resolved_device = _build_predictor(device)
    logger.info(
        "TurnTakingJudge: model on device=%s (requested %s), weights=%s",
        resolved_device,
        device,
        kwargs_used["slu_model_file"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = dict(existing)
    window_samples = int(JUDGE_CONTEXT_S * JUDGE_SR)
    for ci in todo:
        audio_path = _audio_path(duplex_root, run_name, ci)
        if not audio_path.is_file():
            logger.warning(
                "TurnTakingJudge: conv %d missing %s, skipping", ci, audio_path
            )
            continue
        waveform = _load_mono_16k(audio_path)
        total = waveform.shape[0]
        n_windows = (total + window_samples - 1) // window_samples
        if max_windows is not None:
            n_windows = min(n_windows, max_windows)

        windows: list[dict] = []
        for w in range(n_windows):
            start = w * window_samples
            end = min(start + window_samples, total)
            if end - start < SLU_MIN_SAMPLES:
                logger.info(
                    "TurnTakingJudge conv %d window %d: tail %d samples "
                    "< %d, skipping (espnet slu_inference run_chunk loop "
                    "would no-op and raise UnboundLocalError)",
                    ci,
                    w,
                    end - start,
                    SLU_MIN_SAMPLES,
                )
                continue
            chunk = waveform[start:end].numpy()
            import time

            t0 = time.time()
            result_list = predictor(chunk)
            elapsed = time.time() - t0
            if not result_list:
                logger.warning("TurnTakingJudge conv %d window %d: empty result", ci, w)
                continue
            frame_probs, frame_labels = _parse_result(result_list[0])
            label_counts = Counter(frame_labels)
            logger.info(
                "TurnTakingJudge conv %d window %d (%.1f-%.1fs): %d frames in %.1fs -- %s",
                ci,
                w,
                start / JUDGE_SR,
                end / JUDGE_SR,
                len(frame_probs),
                elapsed,
                dict(label_counts),
            )
            windows.append(
                {
                    "window_index": w,
                    "window_start_s": start / JUDGE_SR,
                    "window_end_s": end / JUDGE_SR,
                    "n_frames": len(frame_probs),
                    "frame_s": FRAME_S,
                    "label_counts": dict(label_counts),
                    "frame_labels": frame_labels,
                    "frame_probs": frame_probs,
                    "inference_seconds": elapsed,
                }
            )

        rows[ci] = {
            "conv_index": ci,
            "labels_order": list(LABELS),
            "device": resolved_device,
            "n_windows_processed": len(windows),
            "max_windows": max_windows,
            "windows": windows,
        }

        # Stream-write after each conversation so a partial run isn't lost.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w") as f:
            for k in sorted(rows):
                f.write(json.dumps(rows[k]) + "\n")
        tmp.replace(out_path)

    logger.info("TurnTakingJudge: wrote %s (%d rows)", out_path, len(rows))
    return out_path, resolved_device
