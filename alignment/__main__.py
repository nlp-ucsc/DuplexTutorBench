"""CLI for running ASR + forced-alignment post-processing.

Usage:
    uv run python -m alignment --run-name <name>
    uv run python -m alignment --run-name <name> --mode text_mms
    uv run python -m alignment --run-name <name> --conv-index 0 --force
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alignment.remote_runner import DEFAULT_HOST
from alignment.runner import align_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("alignment")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Align duplex_output/{run}/ via remote ASR."
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--duplex-root", default="duplex_output")
    p.add_argument(
        "--mode",
        choices=["whisper_mms", "text_mms", "whisper_mfa"],
        default="whisper_mms",
    )
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument(
        "--conv-index",
        type=int,
        default=None,
        help="Process only this conversation index.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if already aligned with the same mode.",
    )
    p.add_argument("--whisper-model", default="large-v3-turbo")
    p.add_argument(
        "--aligner-model", default="MahmoudAshraf/mms-300m-1130-forced-aligner"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument(
        "--keep-remote-tmp",
        action="store_true",
        help="Don't delete the remote /tmp/asr_postproc_runs/<run> dir on exit.",
    )
    p.add_argument(
        "--no-bias",
        action="store_true",
        help=(
            "whisper_mms mode only — don't pass the saved overlong text as "
            "Whisper's initial_prompt. Useful as a baseline."
        ),
    )
    p.add_argument(
        "--mfa-num-jobs",
        type=int,
        default=12,
        help=(
            "whisper_mfa mode only — number of parallel MFA jobs per role. "
            "MFA is CPU-only; remote box has 32 cores."
        ),
    )
    p.add_argument(
        "--no-role-parallel",
        action="store_true",
        help=(
            "Run tutor and student alignment sequentially. Default is "
            "parallel so MFA / CTC work overlaps."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = align_run(
        args.run_name,
        duplex_root=Path(args.duplex_root),
        mode=args.mode,
        host=args.host,
        conv_index=args.conv_index,
        force=args.force,
        whisper_model=args.whisper_model,
        aligner_model=args.aligner_model,
        device=args.device,
        compute_dtype=args.compute_dtype,
        keep_remote_tmp=args.keep_remote_tmp,
        no_bias=args.no_bias,
        mfa_num_jobs=args.mfa_num_jobs,
        no_role_parallel=args.no_role_parallel,
    )
    logger.info("Done. Aligned data at: %s", out)


if __name__ == "__main__":
    main()
