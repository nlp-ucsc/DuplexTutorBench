"""Push a duplex run to ucsc_lab_sv11, run align.py, pull results back.

The Mac stores audio + conversations.jsonl; the GPU lives on the remote.
This module is a thin rsync + ssh wrapper, no SDK.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HOST = "ucsc_lab_sv11"
REMOTE_PROJECT = "~/repo/asr_postproc"
REMOTE_SCRATCH_BASE = "/tmp/asr_postproc_runs"


def _check_tools() -> None:
    for tool in ("rsync", "ssh"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool!r} is required but not on PATH")


def _run(cmd: list[str], *, check: bool = True) -> int:
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def push_run(
    run_dir: Path,
    run_name: str,
    *,
    variant: str,
    host: str = DEFAULT_HOST,
) -> str:
    """rsync conversations.jsonl + audio/ to remote scratch, return remote path.

    Also seeds the remote per-conv cache with any prior local
    aligned/<variant>.jsonl, so re-runs without --force actually skip
    completed conversations (the remote scratch dir gets wiped on cleanup, so
    without this the cache always looks empty and every conv gets re-aligned).
    """
    _check_tools()
    remote_dir = f"{REMOTE_SCRATCH_BASE}/{run_name}"
    _run(["ssh", host, f"mkdir -p {remote_dir}/audio"])
    src_jsonl = run_dir / "conversations.jsonl"
    if not src_jsonl.is_file():
        raise FileNotFoundError(src_jsonl)
    _run(["rsync", "-a", str(src_jsonl), f"{host}:{remote_dir}/"])
    src_audio = run_dir / "audio"
    if src_audio.is_dir():
        # Only ship the per-role mono WAVs the aligner actually reads; skip
        # combined.wav (mixed stereo, 2x size) and *.raw (redundant with the
        # WAVs). Cuts upload size by ~55% on cloud-backend runs.
        _run(
            [
                "rsync",
                "-a",
                "--include=*/",
                "--include=tutor_full.wav",
                "--include=student_full.wav",
                "--exclude=*",
                f"{str(src_audio)}/",
                f"{host}:{remote_dir}/audio/",
            ]
        )
    else:
        logger.warning("no audio/ dir under %s — alignment will be empty", run_dir)

    local_aligned = run_dir / "aligned" / f"{variant}.jsonl"
    if local_aligned.is_file():
        _run(["ssh", host, f"mkdir -p {remote_dir}/aligned"])
        _run(
            [
                "rsync",
                "-a",
                str(local_aligned),
                f"{host}:{remote_dir}/aligned/{variant}.jsonl",
            ]
        )
    return remote_dir


def run_align(
    run_name: str,
    *,
    mode: str,
    host: str = DEFAULT_HOST,
    conv_index: int | None = None,
    force: bool = False,
    whisper_model: str = "large-v3-turbo",
    aligner_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    device: str = "cuda",
    compute_dtype: str = "float16",
    no_bias: bool = False,
    mfa_num_jobs: int = 12,
    no_role_parallel: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    """Invoke align.py on the remote host."""
    remote_dir = f"{REMOTE_SCRATCH_BASE}/{run_name}"
    cmd_parts = [
        "cd",
        REMOTE_PROJECT,
        "&&",
        "~/.local/bin/uv",
        "run",
        "python",
        "align.py",
        "--input-dir",
        remote_dir,
        "--mode",
        mode,
        "--whisper-model",
        whisper_model,
        "--aligner-model",
        aligner_model,
        "--device",
        device,
        "--compute-dtype",
        compute_dtype,
    ]
    if conv_index is not None:
        cmd_parts += ["--conv-index", str(conv_index)]
    if force:
        cmd_parts.append("--force")
    if no_bias:
        cmd_parts.append("--no-bias")
    if mode == "whisper_mfa":
        cmd_parts += ["--mfa-num-jobs", str(mfa_num_jobs)]
    if no_role_parallel:
        cmd_parts.append("--no-role-parallel")
    if extra_args:
        cmd_parts += extra_args
    _run(["ssh", host, " ".join(cmd_parts)])


def pull_aligned(
    run_dir: Path,
    run_name: str,
    *,
    variant: str,
    host: str = DEFAULT_HOST,
) -> Path:
    """rsync the per-variant aligned file back into <run_dir>/aligned/."""
    remote_path = f"{REMOTE_SCRATCH_BASE}/{run_name}/aligned/{variant}.jsonl"
    out_path = run_dir / "aligned" / f"{variant}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["rsync", "-a", f"{host}:{remote_path}", str(out_path)])
    return out_path


def cleanup_remote(run_name: str, *, host: str = DEFAULT_HOST) -> None:
    remote_dir = f"{REMOTE_SCRATCH_BASE}/{run_name}"
    _run(["ssh", host, f"rm -rf {remote_dir}"], check=False)
