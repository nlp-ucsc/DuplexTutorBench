"""Per-run manifest: snapshot of args + prompts + targeted pids for replay."""

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

# Args that influence what the dataset contains. Mismatches on these block resume
# unless --force is passed. Operational flags (--web, --port, --concurrency,
# --force, --replay, hosts/ports for the model servers) are deliberately excluded.
OUTPUT_AFFECTING_ARGS: tuple[str, ...] = (
    "tutor_backend",
    "student_backend",
    "tutor_voice",
    "student_voice",
    "gpt_model",
    "tutor_gpt_voice",
    "student_gpt_voice",
    "gemini_model",
    "tutor_gemini_voice",
    "student_gemini_voice",
    "max_duration",
    "split",
    "attempts",
)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return ""


def manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_FILENAME


def build_manifest(
    args: argparse.Namespace,
    tutor_prompt_template: str,
    student_prompt_template: str,
    target_pids: list[str],
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": list(sys.argv),
        "git_sha": _git_sha(),
        "args": vars(args),
        "prompts": {
            "tutor": tutor_prompt_template,
            "student": student_prompt_template,
        },
        "target_pids": list(target_pids),
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)
    logger.info("Wrote manifest to %s", path)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def args_from_manifest(
    manifest: dict[str, Any], run_name: str
) -> tuple[argparse.Namespace, str, str]:
    """Reconstruct args + inlined prompt templates for a replay.

    `run_name` overrides the manifest's saved run_name so replay can write
    to a fresh directory.
    """
    raw = dict(manifest["args"])
    raw["run_name"] = run_name
    # Replay flags themselves should not propagate.
    raw["replay"] = None
    raw["force"] = False
    args = argparse.Namespace(**raw)
    prompts = manifest["prompts"]
    return args, prompts["tutor"], prompts["student"]


def validate_against_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    tutor_prompt_template: str,
    student_prompt_template: str,
    target_pids: list[str],
) -> list[str]:
    """Return a list of human-readable mismatch descriptions (empty if all match)."""
    mismatches: list[str] = []
    saved_args = manifest.get("args", {})
    current_args = vars(args)
    for key in OUTPUT_AFFECTING_ARGS:
        old = saved_args.get(key)
        new = current_args.get(key)
        if old != new:
            mismatches.append(f"  {key}: manifest={old!r} vs current={new!r}")

    saved_prompts = manifest.get("prompts", {})
    if saved_prompts.get("tutor", "") != tutor_prompt_template:
        mismatches.append("  tutor prompt content differs from manifest")
    if saved_prompts.get("student", "") != student_prompt_template:
        mismatches.append("  student prompt content differs from manifest")

    saved_pids = list(manifest.get("target_pids", []))
    if saved_pids != list(target_pids):
        mismatches.append(
            f"  target_pids differ: manifest has {len(saved_pids)} pids, "
            f"current has {len(target_pids)}"
        )

    return mismatches
