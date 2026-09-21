"""Push audio to ucsc_lab_sv11, run the VAP forward pass, pull frames back.

Mirrors ``alignment/remote_runner.py``: the Mac orchestrates, the GPU box does
the heavy compute. Thin rsync + ssh wrapper, no SDK. The VAP venv + checkpoint
live at ``~/repo/VoiceActivityProjection`` on the remote (one-time setup, see
``speech_eval/docs/quickstart.md``).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_VAP_HOST = "ucsc_lab_sv11"
REMOTE_VAP_PROJECT = "~/repo/VoiceActivityProjection"
REMOTE_VAP_SCRATCH = "/tmp/vap_runs"
REMOTE_VAP_CHECKPOINT = "example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"


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


def push_vap(
    run_dir: Path,
    run_name: str,
    jobs: list[dict],
    jobs_path: Path,
    infer_script: Path,
    *,
    host: str = DEFAULT_VAP_HOST,
) -> str:
    """rsync combined.wav (todo convs only) + jobs.json + _vap_infer.py to remote."""
    _check_tools()
    remote_dir = f"{REMOTE_VAP_SCRATCH}/{run_name}"
    _run(["ssh", host, f"mkdir -p {remote_dir}/audio"])

    for j in jobs:
        ci = int(j["conv_index"])
        src = run_dir / "audio" / str(ci) / "combined.wav"
        if not src.is_file():
            raise FileNotFoundError(src)
        # Trailing slash on the dest makes rsync create the per-conv dir.
        _run(["ssh", host, f"mkdir -p {remote_dir}/audio/{ci}"])
        _run(["rsync", "-a", str(src), f"{host}:{remote_dir}/audio/{ci}/combined.wav"])

    _run(["rsync", "-a", str(jobs_path), f"{host}:{remote_dir}/jobs.json"])
    # The repo is the source of truth for the inference script; ship it next to
    # run.py so `import vap` + the relative checkpoint path resolve.
    _run(
        ["rsync", "-a", str(infer_script), f"{host}:{REMOTE_VAP_PROJECT}/_vap_infer.py"]
    )
    return remote_dir


def run_vap(
    run_name: str,
    *,
    host: str = DEFAULT_VAP_HOST,
    gpu: int | None = None,
    conv_index: int | None = None,
    force: bool = False,
) -> None:
    """Invoke _vap_infer.py on the remote with the VAP venv."""
    remote_dir = f"{REMOTE_VAP_SCRATCH}/{run_name}"
    parts = ["cd", REMOTE_VAP_PROJECT, "&&"]
    if gpu is not None:
        parts.append(f"CUDA_VISIBLE_DEVICES={gpu}")
    parts += [
        ".venv/bin/python",
        "_vap_infer.py",
        "--input-dir",
        remote_dir,
        "--checkpoint",
        REMOTE_VAP_CHECKPOINT,
    ]
    if conv_index is not None:
        parts += ["--conv-index", str(conv_index)]
    if force:
        parts.append("--force")
    _run(["ssh", host, " ".join(parts)])


def pull_vap(
    frames_dir: Path,
    run_name: str,
    *,
    host: str = DEFAULT_VAP_HOST,
) -> Path:
    """rsync the per-conv frame npzs back into ``frames_dir``."""
    remote_dir = f"{REMOTE_VAP_SCRATCH}/{run_name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run(["rsync", "-a", f"{host}:{remote_dir}/frames/", f"{frames_dir}/"])
    return frames_dir


def cleanup_vap(
    run_name: str, *, host: str = DEFAULT_VAP_HOST, keep: bool = False
) -> None:
    if keep:
        return
    remote_dir = f"{REMOTE_VAP_SCRATCH}/{run_name}"
    _run(["ssh", host, f"rm -rf {remote_dir}"], check=False)
