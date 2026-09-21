"""Audiobox Aesthetics scoring for duplex conversations.

Runs the released `audiobox-aesthetics` predictor on three channels per
conversation -- the tutor mono, the student mono, and a mono mixdown of the
stereo ``combined.wav`` -- and writes one JSONL row per conversation to
``speech_eval_output/{run_name}/audiobox.jsonl``.

Each row has 4 axes (PQ / PC / CE / CU) for each of the 3 channels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

logger = logging.getLogger(__name__)

CHANNELS = ("tutor", "student", "mixed")
AXES = ("PQ", "PC", "CE", "CU")


@dataclass
class AudioboxResult:
    conv_index: int
    pid: str
    # Nested dict: channel -> axis -> score
    scores: dict[str, dict[str, float]]
    device: str


def _load_mono(path: Path) -> tuple[torch.Tensor, int]:
    """Load a wav as a (1, samples) float32 tensor and the sample rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); mix to mono if needed.
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    tensor = torch.from_numpy(data.T).contiguous()  # (1, samples)
    return tensor, sr


def _audio_dir(duplex_root: Path, run_name: str, conv_index: int) -> Path:
    return duplex_root / run_name / "audio" / str(conv_index)


def score_run(
    run_name: str,
    *,
    duplex_root: Path,
    out_dir: Path,
    conv_indices: list[int] | None = None,
    overwrite: bool = False,
) -> tuple[Path, str]:
    """Score every conversation under ``duplex_root/run_name``.

    The audiobox-aesthetics library auto-detects its own device (cuda → mps →
    cpu) inside ``AesPredictor.setup_model``; we surface the resolved device
    back to the caller alongside the output path.

    Returns
    -------
    (output_path, resolved_device)
    """
    from audiobox_aesthetics.infer import initialize_predictor

    out_path = out_dir / "audiobox.jsonl"
    existing: dict[int, dict] = {}
    if out_path.is_file() and not overwrite:
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                existing[int(row["conv_index"])] = row

    # Enumerate conversations from the audio/ folder so we don't need the
    # JSONL just for indexing.
    audio_root = duplex_root / run_name / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"No audio dir at {audio_root}")
    all_indices = sorted(
        int(p.name) for p in audio_root.iterdir() if p.is_dir() and p.name.isdigit()
    )
    if conv_indices is not None:
        all_indices = [i for i in all_indices if i in set(conv_indices)]

    logger.info("Audiobox: %d conversations to score", len(all_indices))

    predictor = initialize_predictor()
    device = str(predictor.device)
    logger.info("Audiobox: model on device=%s", device)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = dict(existing)
    for ci in all_indices:
        if not overwrite and ci in existing:
            logger.info("Audiobox: conv %d already scored, skipping", ci)
            continue
        adir = _audio_dir(duplex_root, run_name, ci)
        try:
            tutor, sr_t = _load_mono(adir / "tutor_full.wav")
            student, sr_s = _load_mono(adir / "student_full.wav")
            mixed, sr_m = _load_mono(adir / "combined.wav")
        except FileNotFoundError as e:
            logger.warning("Audiobox: conv %d missing audio (%s), skipping", ci, e)
            continue

        # Predictor accepts a list of {path: tensor, sample_rate}.
        batch = [
            {"path": tutor, "sample_rate": sr_t},
            {"path": student, "sample_rate": sr_s},
            {"path": mixed, "sample_rate": sr_m},
        ]
        outs = predictor.forward(batch)
        # outs is a list of dicts in the same order, with keys CE/CU/PC/PQ.
        scores = {
            ch: {ax: float(o[ax]) for ax in AXES} for ch, o in zip(CHANNELS, outs)
        }

        row = {
            "conv_index": ci,
            "channels": scores,
            "device": device,
            "audio_sample_rate": sr_t,
        }
        rows[ci] = row
        logger.info(
            "Audiobox conv %d: tutor PQ=%.2f CE=%.2f / student PQ=%.2f CE=%.2f / mixed PQ=%.2f CE=%.2f",
            ci,
            scores["tutor"]["PQ"],
            scores["tutor"]["CE"],
            scores["student"]["PQ"],
            scores["student"]["CE"],
            scores["mixed"]["PQ"],
            scores["mixed"]["CE"],
        )

        # Stream-write after each conversation so a Ctrl-C mid-run doesn't
        # lose work. Matches the judge's pattern.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w") as f:
            for k in sorted(rows):
                f.write(json.dumps(rows[k]) + "\n")
        tmp.replace(out_path)

    logger.info("Audiobox: wrote %s (%d rows)", out_path, len(rows))
    return out_path, device
