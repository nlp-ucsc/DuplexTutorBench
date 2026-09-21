"""CLI for the speech_eval module.

Usage:
    uv run python -m speech_eval --run-name <duplex_run> [...]

Scores the audio of a duplex run on two axes:

* Audiobox Aesthetics (4 quality axes, per channel)
* ESPnet turn-taking judge (5-class per-frame predictions)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from speech_eval.runner import ALL_COMPONENTS, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m speech_eval",
        description=(
            "Score the audio of a duplex run with Audiobox Aesthetics and the "
            "ESPnet turn-taking judge."
        ),
    )
    p.add_argument(
        "--run-name", required=True, help="Duplex run name under duplex_output/"
    )
    p.add_argument(
        "--component",
        default="all",
        help=(
            "Which component(s) to run. 'all' (default), or a comma-separated "
            f"subset of: {', '.join(ALL_COMPONENTS)}"
        ),
    )
    p.add_argument(
        "--duplex-root", default="duplex_output", help="Default: duplex_output"
    )
    p.add_argument(
        "--output-dir",
        default="speech_eval_output",
        help="Default: speech_eval_output",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help=(
            "Device for the ESPnet judge. 'auto' (default) prefers cuda, "
            "then mps, then cpu. Audiobox always auto-detects regardless "
            "of this flag. The vap component ignores this flag (it always "
            "runs remotely on CUDA; see --vap-host / --vap-gpu). On a "
            "multi-GPU host, pick a specific card with CUDA_VISIBLE_DEVICES=N. "
            "The actually-resolved device is recorded in manifest.json."
        ),
    )
    p.add_argument(
        "--vap-host",
        default="ucsc_lab_sv11",
        help="SSH host for the remote VAP forward pass. Default: ucsc_lab_sv11",
    )
    p.add_argument(
        "--vap-gpu",
        type=int,
        default=None,
        help=(
            "Remote GPU index for the vap component (sets CUDA_VISIBLE_DEVICES "
            "on the remote). Default: None (remote default)."
        ),
    )
    p.add_argument(
        "--keep-remote-tmp",
        action="store_true",
        help="Don't delete the remote /tmp/vap_runs scratch dir after the vap run.",
    )
    p.add_argument(
        "--vap-aligned-variant",
        default="whisper_mfa",
        help=(
            "Which aligned/<variant>.jsonl timeline the vap component reads for "
            "event detection. Default: whisper_mfa. Use a different name for a "
            "run whose timings are gold (e.g. 'maptask_gold' for the Map Task "
            "human reference, whose NXT word times need no forced alignment)."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N conversations (sorted by index).",
    )
    p.add_argument(
        "--conv-index",
        type=int,
        default=None,
        help="Score only this single conversation index.",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help=(
            "For the judge: cap on non-overlapping 30 s windows per "
            "conversation. 0 (default) = full conversation. A positive "
            "integer limits how many windows are processed (e.g. 1 = first "
            "30 s only, useful for fast smoke tests on CPU)."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if outputs already exist for a conversation.",
    )
    return p.parse_args()


def _resolve_conv_indices(args: argparse.Namespace) -> list[int] | None:
    if args.conv_index is not None and args.limit is not None:
        raise SystemExit("--conv-index and --limit are mutually exclusive")
    if args.conv_index is not None:
        return [args.conv_index]
    if args.limit is not None:
        audio_root = Path(args.duplex_root) / args.run_name / "audio"
        all_idx = sorted(
            int(p.name) for p in audio_root.iterdir() if p.is_dir() and p.name.isdigit()
        )
        return all_idx[: args.limit]
    return None


def main() -> None:
    args = _parse_args()
    if args.component == "all":
        components = list(ALL_COMPONENTS)
    else:
        components = [c.strip() for c in args.component.split(",") if c.strip()]

    if args.max_windows < 0:
        raise SystemExit(
            f"--max-windows must be >= 0 (0 = full conversation); got {args.max_windows}"
        )
    max_windows = args.max_windows or None

    out_dir = run(
        run_name=args.run_name,
        components=components,
        duplex_root=Path(args.duplex_root),
        out_root=Path(args.output_dir),
        device=args.device,
        conv_indices=_resolve_conv_indices(args),
        overwrite=args.overwrite,
        max_windows=max_windows,
        host=args.vap_host,
        vap_gpu=args.vap_gpu,
        keep_remote_tmp=args.keep_remote_tmp,
        aligned_variant=args.vap_aligned_variant,
    )
    print(f"speech_eval output: {out_dir}")


if __name__ == "__main__":
    main()
