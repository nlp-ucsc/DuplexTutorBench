"""VAP turn-taking scorer over duplex conversations.

Local orchestrator with the same ``score_run`` contract as the other
speech_eval scorers. It runs the VAP forward pass on the remote GPU box
(``vap_remote``) and does all metric logic locally (``vap_events``), so a
re-score never touches the GPU.

Pipeline per run:

1. Read the forced-alignment timeline ``aligned/{variant}.jsonl`` (NOT the raw
   ``conversations.jsonl`` — those timestamps are text-delta arrival times, not
   audio times). Compute each conversation's first speech onset for the
   leading-silence trim.
2. For conversations that need the forward pass, rsync ``combined.wav`` +
   jobs.json to the remote, run ``_vap_infer.py``, pull ``frames/{conv}.npz``
   back into ``speech_eval_output/{run}/vap_frames/``.
3. Join frames + aligned timeline through ``vap_events.detect_and_score`` and
   stream-write ``speech_eval_output/{run}/vap.jsonl`` (one row per conversation).

See ``speech_eval/docs/architecture.md`` for the three readouts and
``speech_eval/docs/outputs.md`` for the row schema.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from speech_eval import vap_events
from speech_eval.vap_remote import (
    DEFAULT_VAP_HOST,
    REMOTE_VAP_CHECKPOINT,
    cleanup_vap,
    pull_vap,
    push_vap,
    run_vap,
)

logger = logging.getLogger(__name__)

DEFAULT_ALIGNED_VARIANT = "whisper_mfa"


def _aligned_path(duplex_root: Path, run_name: str, variant: str) -> Path:
    return duplex_root / run_name / "aligned" / f"{variant}.jsonl"


def _load_npz(path: Path) -> tuple[dict, dict]:
    """Return (frames-dict-of-arrays, header-dict) from a per-conv npz."""
    with np.load(path) as data:
        frames = {
            k: data[k]
            for k in (
                "p_now0",
                "p_now1",
                "p_future0",
                "p_future1",
                "vad0",
                "vad1",
                "frame_times_original",
            )
        }
        header = json.loads(str(data["header"]))
    return frames, header


def score_run(
    run_name: str,
    *,
    duplex_root: Path,
    out_dir: Path,
    conv_indices: list[int] | None = None,
    overwrite: bool = False,
    host: str = DEFAULT_VAP_HOST,
    vap_gpu: int | None = None,
    keep_remote_tmp: bool = False,
    aligned_variant: str = DEFAULT_ALIGNED_VARIANT,
) -> tuple[Path, str]:
    """Score every conversation under ``duplex_root/run_name`` with VAP.

    Returns ``(output_path, device_string)`` where the device string is e.g.
    ``"ucsc_lab_sv11:cuda:0"`` (the forward pass always runs remotely on CUDA).
    """
    device_str = f"{host}:cuda" + (f":{vap_gpu}" if vap_gpu is not None else "")
    out_path = out_dir / "vap.jsonl"
    frames_dir = out_dir / "vap_frames"

    # Resume: keep existing rows unless overwriting.
    existing: dict[int, dict] = {}
    if out_path.is_file() and not overwrite:
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
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
        return out_path, device_str

    aligned_path = _aligned_path(duplex_root, run_name, aligned_variant)
    if not aligned_path.is_file():
        raise FileNotFoundError(
            f"missing aligned timeline {aligned_path}. VAP event detection needs "
            f"forced-alignment timestamps (run `python -m alignment --run-name "
            f"{run_name} --mode {aligned_variant}`); the raw conversations.jsonl "
            f"timestamps are text-delta arrival times and unusable for timing."
        )
    aligned = vap_events.load_aligned_conversations(aligned_path)

    # Build the jobs spec; skip convs lacking aligned data or a stereo wav.
    jobs: list[dict] = []
    scorable: list[int] = []
    for ci in todo:
        row = aligned.get(ci)
        if row is None:
            logger.warning("VAP: conv %d not in %s, skipping", ci, aligned_path.name)
            continue
        onset = vap_events.first_onset(row["segments"])
        if onset is None:
            logger.warning("VAP: conv %d has no aligned words, skipping", ci)
            continue
        if not (audio_root / str(ci) / "combined.wav").is_file():
            logger.warning("VAP: conv %d missing combined.wav, skipping", ci)
            continue
        scorable.append(ci)
        jobs.append(
            {
                "conv_index": ci,
                "wav": f"audio/{ci}/combined.wav",
                "first_onset_s": float(onset),
            }
        )

    if not scorable:
        logger.warning("VAP: nothing scorable for %s", run_name)
        return out_path, device_str

    # Only run the forward pass for convs whose frames aren't already cached
    # locally — even under --overwrite, which re-derives the metrics (event
    # detection, distributions) from cached frames without touching the GPU.
    # That is what makes threshold sweeps and metric additions cheap; to force a
    # fresh forward pass, delete the conversation's vap_frames/<idx>.npz.
    need_infer = [
        j for j in jobs if not (frames_dir / f"{j['conv_index']}.npz").is_file()
    ]
    if need_infer:
        logger.info(
            "VAP: %d/%d conversations need the remote forward pass (host=%s gpu=%s)",
            len(need_infer),
            len(scorable),
            host,
            vap_gpu,
        )
        infer_script = Path(__file__).resolve().parent / "_vap_infer.py"
        fd, jobs_tmp = tempfile.mkstemp(suffix="_vap_jobs.json")
        os.close(fd)
        jobs_path = Path(jobs_tmp)
        try:
            jobs_path.write_text(json.dumps({"jobs": need_infer}))
            push_vap(
                duplex_root / run_name,
                run_name,
                need_infer,
                jobs_path,
                infer_script,
                host=host,
            )
            run_vap(run_name, host=host, gpu=vap_gpu, force=overwrite)
            pull_vap(frames_dir, run_name, host=host)
        finally:
            jobs_path.unlink(missing_ok=True)
            cleanup_vap(run_name, host=host, keep=keep_remote_tmp)
    else:
        logger.info("VAP: all %d frame files cached, re-scoring only", len(scorable))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = dict(existing)
    checkpoint_name = os.path.basename(REMOTE_VAP_CHECKPOINT)
    for ci in scorable:
        npz_path = frames_dir / f"{ci}.npz"
        if not npz_path.is_file():
            logger.warning("VAP: conv %d frames missing after pull, skipping", ci)
            continue
        frames, header = _load_npz(npz_path)
        segments = aligned[ci]["segments"]
        scored = vap_events.detect_and_score(segments, frames)
        rows[ci] = {
            "conv_index": ci,
            "channel_order": ["tutor", "student"],
            "frame_hz": header["frame_hz"],
            "sample_rate": header["sample_rate"],
            "device": device_str,
            "checkpoint": checkpoint_name,
            "aligned_variant": aligned_variant,
            "first_onset_s": round(float(header["first_onset_s"]), 3),
            "preroll_s": header["preroll_s"],
            "trim_start_s": round(float(header["trim_start_s"]), 3),
            "n_frames": int(header["n_frames"]),
            "thresholds": vap_events.thresholds_dict(),
            "distributions": vap_events.compute_distributions(segments),
            **scored,
        }

        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w") as f:
            for k in sorted(rows):
                f.write(json.dumps(rows[k]) + "\n")
        tmp.replace(out_path)

    logger.info("VAP: wrote %s (%d rows)", out_path, len(rows))
    return out_path, device_str
