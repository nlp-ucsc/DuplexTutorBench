"""Top-level orchestration for the speech_eval module.

Dispatches to ``audiobox_scorer`` and ``turntaking_judge``, writes a single
``manifest.json`` describing what was run, where the weights live, and which
conversations were touched.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ALL_COMPONENTS = ("audiobox", "judge", "vap")


def _git_sha(repo: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _resolve_weight_paths(components: list[str]) -> dict[str, str | None]:
    """Best-effort report of where each backend's weights are cached.

    Triggers a small import each so that the paths reflect what we'd actually
    use at run time. Returns ``None`` for any path that cannot be resolved.
    """
    paths: dict[str, str | None] = {
        "audiobox_hf_cache_root": str(Path.home() / ".cache" / "huggingface" / "hub"),
        "audiobox_dir": None,
        "espnet_judge_file": None,
        "espnet_judge_dir": None,
        "whisper_medium_pt": None,
    }
    # Audiobox
    abx = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--facebook--audiobox-aesthetics"
    )
    if abx.is_dir():
        paths["audiobox_dir"] = str(abx)
    # ESPnet judge — known to live inside the venv via espnet_model_zoo.
    # Skipped when the judge component isn't requested, since
    # ``download_and_unpack`` will fetch ~70 MB if the weights aren't cached.
    if "judge" in components:
        try:
            from speech_eval.turntaking_judge import _patch_whisper_n_mels

            _patch_whisper_n_mels()
            from espnet_model_zoo.downloader import ModelDownloader

            info = ModelDownloader().download_and_unpack(
                "espnet/Turn_taking_prediction_SWBD"
            )
            paths["espnet_judge_file"] = info.get("asr_model_file")
            if paths["espnet_judge_file"]:
                paths["espnet_judge_dir"] = str(
                    Path(paths["espnet_judge_file"]).parent.parent.parent
                )
        except Exception as e:
            logger.warning("resolve weights: espnet path lookup failed: %s", e)
    # Whisper-medium
    wm = Path.home() / ".cache" / "whisper" / "medium.pt"
    if wm.is_file():
        paths["whisper_medium_pt"] = str(wm)
    # VAP — runs remotely; record static pointers (no local import).
    if "vap" in components:
        from speech_eval.vap_remote import (
            DEFAULT_VAP_HOST,
            REMOTE_VAP_CHECKPOINT,
            REMOTE_VAP_PROJECT,
        )

        paths["vap_remote_host"] = DEFAULT_VAP_HOST
        paths["vap_remote_project"] = REMOTE_VAP_PROJECT
        paths["vap_checkpoint"] = REMOTE_VAP_CHECKPOINT
    return paths


def run(
    *,
    run_name: str,
    components: list[str],
    duplex_root: Path,
    out_root: Path,
    device: str,
    conv_indices: list[int] | None,
    overwrite: bool,
    max_windows: int | None,
    host: str = "ucsc_lab_sv11",
    vap_gpu: int | None = None,
    keep_remote_tmp: bool = False,
    aligned_variant: str = "whisper_mfa",
) -> Path:
    """Run the requested components and write a manifest.

    Returns the output directory.
    """
    bad = [c for c in components if c not in ALL_COMPONENTS]
    if bad:
        raise ValueError(
            f"Unknown component(s): {bad}. Known: {', '.join(ALL_COMPONENTS)}"
        )

    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)

    artifacts: dict[str, str] = {}
    devices_used: dict[str, str] = {}
    if "audiobox" in components:
        from speech_eval.audiobox_scorer import score_run as audiobox_run

        path, audiobox_device = audiobox_run(
            run_name,
            duplex_root=duplex_root,
            out_dir=out_dir,
            conv_indices=conv_indices,
            overwrite=overwrite,
        )
        artifacts["audiobox"] = str(path.relative_to(out_dir))
        devices_used["audiobox"] = audiobox_device
    if "judge" in components:
        from speech_eval.turntaking_judge import score_run as judge_run

        path, judge_device = judge_run(
            run_name,
            duplex_root=duplex_root,
            out_dir=out_dir,
            conv_indices=conv_indices,
            overwrite=overwrite,
            device=device,
            max_windows=max_windows,
        )
        artifacts["turntaking_judge"] = str(path.relative_to(out_dir))
        devices_used["judge"] = judge_device
    if "vap" in components:
        from speech_eval.vap_scorer import score_run as vap_run

        path, vap_device = vap_run(
            run_name,
            duplex_root=duplex_root,
            out_dir=out_dir,
            conv_indices=conv_indices,
            overwrite=overwrite,
            host=host,
            vap_gpu=vap_gpu,
            keep_remote_tmp=keep_remote_tmp,
            aligned_variant=aligned_variant,
        )
        artifacts["vap"] = str(path.relative_to(out_dir))
        devices_used["vap"] = vap_device

    end = datetime.now(timezone.utc)
    manifest = {
        "run_name": run_name,
        "components": components,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": end.isoformat(),
        "elapsed_seconds": (end - start).total_seconds(),
        "device_requested": device,
        "devices_used": devices_used,
        "conv_indices": conv_indices,
        "max_windows": max_windows,
        "overwrite": overwrite,
        "duplex_root": str(duplex_root),
        "out_root": str(out_root),
        "argv": sys.argv,
        "git_sha": _git_sha(Path(__file__).resolve().parent.parent),
        "artifacts": artifacts,
        "weight_paths": _resolve_weight_paths(components),
        "vap": (
            {
                "frame_hz": 50,
                "sample_rate": 16000,
                "channel_order": ["tutor", "student"],
                "aligned_variant": aligned_variant,
                "host": host,
                "gpu": vap_gpu,
            }
            if "vap" in components
            else None
        ),
        "env": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK"
            ),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("speech_eval: wrote %s", manifest_path)
    return out_dir
