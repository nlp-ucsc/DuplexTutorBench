"""Orchestrate one alignment run end-to-end."""

from __future__ import annotations

import logging
from pathlib import Path

from alignment.remote_runner import (
    DEFAULT_HOST,
    cleanup_remote,
    pull_aligned,
    push_run,
    run_align,
)
from alignment.schemas import variant_key

logger = logging.getLogger(__name__)


def align_run(
    run_name: str,
    *,
    duplex_root: Path = Path("duplex_output"),
    mode: str = "whisper_mms",
    host: str = DEFAULT_HOST,
    conv_index: int | None = None,
    force: bool = False,
    whisper_model: str = "large-v3-turbo",
    aligner_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    device: str = "cuda",
    compute_dtype: str = "float16",
    keep_remote_tmp: bool = False,
    no_bias: bool = False,
    mfa_num_jobs: int = 12,
    no_role_parallel: bool = False,
) -> Path:
    """Push a run to remote, align it, pull the result back.

    Returns the local path to the written aligned/<variant>.jsonl.
    """
    run_dir = duplex_root / run_name
    if not (run_dir / "conversations.jsonl").is_file():
        raise FileNotFoundError(f"No conversations.jsonl under {run_dir}")

    variant = variant_key(mode, no_bias)

    logger.info("Pushing %s to %s ...", run_dir, host)
    push_run(run_dir, run_name, variant=variant, host=host)

    logger.info(
        "Running alignment on remote (mode=%s, prompt_bias=%s, variant=%s) ...",
        mode,
        "off" if no_bias else "on",
        variant,
    )
    run_align(
        run_name,
        mode=mode,
        host=host,
        conv_index=conv_index,
        force=force,
        whisper_model=whisper_model,
        aligner_model=aligner_model,
        device=device,
        compute_dtype=compute_dtype,
        no_bias=no_bias,
        mfa_num_jobs=mfa_num_jobs,
        no_role_parallel=no_role_parallel,
    )

    logger.info("Pulling aligned/%s.jsonl back ...", variant)
    out_path = pull_aligned(run_dir, run_name, variant=variant, host=host)
    logger.info("Wrote %s", out_path)

    if not keep_remote_tmp:
        logger.info("Cleaning up remote scratch dir ...")
        cleanup_remote(run_name, host=host)

    return out_path
