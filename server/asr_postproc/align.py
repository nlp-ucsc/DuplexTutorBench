"""ASR + forced-alignment post-processing for duplex conversations.

Reads `<input_dir>/conversations.jsonl` and `<input_dir>/audio/{idx}/{role}_full.wav`,
writes `<input_dir>/aligned/<variant_key>.jsonl` (atomic via .tmp rename),
where the variant key encodes `(mode, no_bias)` so biased and un-biased
runs of the same mode can coexist:
  whisper_mms, whisper_mms_no_bias,
  whisper_mfa, whisper_mfa_no_bias,
  text_mms.

Usage (on remote):
  uv run python align.py --input-dir /tmp/asr_postproc_runs/<run> \\
      --mode whisper_mms --device cuda --whisper-model large-v3-turbo \\
      [--conv-index N] [--force] [--no-bias]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from segment_words import words_to_segments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("align")


ROLES = ("tutor", "student")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument(
        "--mode",
        choices=["whisper_mms", "text_mms", "whisper_mfa"],
        default="whisper_mms",
    )
    p.add_argument("--whisper-model", default="large-v3-turbo")
    p.add_argument(
        "--aligner-model", default="MahmoudAshraf/mms-300m-1130-forced-aligner"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument("--language", default="en", help="Whisper language code")
    p.add_argument("--iso-language", default="eng", help="ISO 639-3 for aligner")
    p.add_argument(
        "--conv-index",
        type=int,
        default=None,
        help="Process only this conversation index.",
    )
    p.add_argument(
        "--roles", default="tutor,student", help="Comma-separated roles to align."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if already in aligned/<variant>.jsonl.",
    )
    p.add_argument(
        "--no-bias",
        action="store_true",
        help=(
            "Don't pass the saved overlong transcript as Whisper's "
            "initial_prompt. whisper_mms mode only — let Whisper transcribe "
            "purely from audio. Useful as a baseline against the bias-on run."
        ),
    )
    p.add_argument(
        "--mfa-num-jobs",
        type=int,
        default=12,
        help=(
            "whisper_mfa mode only — passed to `mfa align --num_jobs`. MFA is "
            "CPU-only; the remote box has 32 cores so 12 is a sensible default. "
            "Combined with per-conv tutor+student parallelism (2 concurrent "
            "mfa subprocesses), this uses ~2*num_jobs cores at peak."
        ),
    )
    p.add_argument(
        "--no-role-parallel",
        action="store_true",
        help=(
            "Disable running tutor and student alignment in parallel. By "
            "default they run concurrently via a ThreadPoolExecutor so their "
            "MFA / CTC work overlaps."
        ),
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_all(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in sorted(records, key=lambda r: r.get("conv_index", 0)):
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def _concat_role_text(conv: dict, role: str) -> str:
    parts = [s["text"] for s in conv.get("segments", []) if s.get("role") == role]
    return " ".join(p.strip() for p in parts if p and p.strip())


def _role_segments(conv: dict, role: str) -> list[dict]:
    return [s for s in conv.get("segments", []) if s.get("role") == role]


def _run_whisper_mms(args, wav_path, intended_text):
    """Returns (text, segments_words[]) via the whisper_mms pipeline."""
    from pipeline_whisper_mms import align_role as a_align

    return a_align(
        wav_path,
        intended_text,
        whisper_model=args.whisper_model,
        aligner_model=args.aligner_model,
        device=args.device,
        compute_dtype=args.compute_dtype,
        language=args.language,
        iso_language=args.iso_language,
        use_intended_prompt=(not args.no_bias) and bool(intended_text),
    )


def _run_whisper_mfa(args, wav_path, intended_text):
    """Returns (text, segments_words[]) via the whisper_mfa pipeline."""
    from pipeline_whisper_mfa import align_role as m_align

    return m_align(
        wav_path,
        intended_text,
        whisper_model=args.whisper_model,
        aligner_model=args.aligner_model,  # ignored by MFA, passed for parity
        device=args.device,
        compute_dtype=args.compute_dtype,
        language=args.language,
        iso_language=args.iso_language,
        use_intended_prompt=(not args.no_bias) and bool(intended_text),
        num_jobs=args.mfa_num_jobs,
    )


def _process_role(args, wav_path, intended_text, role_segs):
    """Returns (text, segments_words[]) — only used for text_mms."""
    from pipeline_text_mms import align_role_segments as f_align

    return f_align(
        wav_path,
        role_segs,
        aligner_model=args.aligner_model,
        device=args.device,
        compute_dtype=args.compute_dtype,
        iso_language=args.iso_language,
    )


def _align_one_role(args, run_dir, conv, conv_index, role) -> dict:
    """Run the alignment pipeline for one role; return a dict the caller
    aggregates. Pure with respect to shared state (writes nothing).

    Safe to call concurrently for tutor + student: faster-whisper's
    WhisperModel.transcribe serializes internally on the singleton, and
    each role's MFA subprocess is isolated; the overlap win comes from
    one role's MFA running while the other's Whisper/MFA runs.
    """
    audio_dir = run_dir / "audio" / str(conv_index)
    wav = audio_dir / f"{role}_full.wav"
    role_segs = _role_segments(conv, role)

    if not wav.is_file():
        logger.warning("[conv=%d %s] missing %s, skipping role", conv_index, role, wav)
        return {
            "role": role,
            "source_count": len(role_segs),
            "aligned_count": 0,
            "text": "",
            "aligned_segments": [],
        }

    # Per-role WAVs are conversation-aligned (start at t=0, length =
    # conv.duration), so segment times are already in the audio's frame
    # of reference — no offset shift needed.
    if args.mode == "text_mms":
        if not role_segs:
            # Nothing to align (e.g. human role — no text deltas were ever
            # emitted). text_mms requires text; switch this role to
            # whisper_mms as a fallback so we still produce something useful.
            logger.info(
                "[conv=%d %s] no original segments — falling back to "
                "whisper_mms (no prompt bias) for this role",
                conv_index,
                role,
            )
            local_segs = []
            intended = ""
        else:
            local_segs = role_segs
            intended = ""
    else:
        local_segs = []
        intended = _concat_role_text(conv, role)
        # Empty `intended` is normal for a human role — no bias is passed,
        # but we still want the ASR transcript.

    t0 = time.time()
    effective_mode = (
        "whisper_mms" if args.mode == "text_mms" and not role_segs else args.mode
    )
    if effective_mode == "whisper_mms":
        text, segments_words = _run_whisper_mms(args, wav, intended)
    elif effective_mode == "whisper_mfa":
        text, segments_words = _run_whisper_mfa(args, wav, intended)
    else:
        text, segments_words = _process_role(args, wav, intended, local_segs)
    dt = time.time() - t0

    n_words = sum(len(sw) for sw in segments_words)
    if n_words == 0:
        logger.info(
            "[conv=%d %s] %.1fs -> 0 words (mode=%s)",
            conv_index,
            role,
            dt,
            args.mode,
        )
        return {
            "role": role,
            "source_count": len(role_segs),
            "aligned_count": 0,
            "text": "",
            "aligned_segments": [],
        }

    aligned_segments: list[dict] = []
    n_role_segments = 0
    for seg_words in segments_words:
        if not seg_words:
            continue
        new_segs = words_to_segments(role, seg_words)
        aligned_segments.extend(new_segs)
        n_role_segments += len(new_segs)

    logger.info(
        "[conv=%d %s] %.1fs -> %d words / %d segments (mode=%s)",
        conv_index,
        role,
        dt,
        n_words,
        n_role_segments,
        args.mode,
    )
    return {
        "role": role,
        "source_count": len(role_segs),
        "aligned_count": n_role_segments,
        "text": text,
        "aligned_segments": aligned_segments,
    }


def _process_conv(args, run_dir, conv, conv_index) -> dict:
    pid = str(conv.get("pid", ""))
    aligned_segments: list[dict] = []
    diag = {
        "source_segments_count": {},
        "aligned_segments_count": {},
        "asr_text": {},
    }

    roles = [r.strip() for r in args.roles.split(",") if r.strip() in ROLES]

    if len(roles) <= 1 or args.no_role_parallel:
        results = [_align_one_role(args, run_dir, conv, conv_index, r) for r in roles]
    else:
        # Tutor and student share no per-call state. whisper_mfa: their
        # subprocesses run in parallel on CPU. whisper_mms/text_mms:
        # both submit to the same GPU singleton aligner — calls serialize
        # internally but the orchestration overhead drops to zero.
        with ThreadPoolExecutor(max_workers=len(roles)) as ex:
            results = list(
                ex.map(
                    lambda r: _align_one_role(args, run_dir, conv, conv_index, r),
                    roles,
                )
            )

    for res in results:
        role = res["role"]
        diag["source_segments_count"][role] = res["source_count"]
        diag["aligned_segments_count"][role] = res["aligned_count"]
        if res["text"]:
            diag["asr_text"][role] = res["text"]
        aligned_segments.extend(res["aligned_segments"])

    aligned_segments.sort(key=lambda s: s["start_time"])

    return {
        "pid": pid,
        "attempt_index": int(conv.get("attempt_index", 0)),
        "conv_index": conv_index,
        "duration": float(conv.get("duration", 0.0)),
        "tutor_voice": conv.get("tutor_voice", ""),
        "student_voice": conv.get("student_voice", ""),
        "tutor_backend": conv.get("tutor_backend", ""),
        "student_backend": conv.get("student_backend", ""),
        "alignment": {
            "mode": args.mode,
            "asr_model": (
                args.whisper_model
                if args.mode in ("whisper_mms", "whisper_mfa")
                else None
            ),
            "aligner": (
                "mfa:english_mfa" if args.mode == "whisper_mfa" else args.aligner_model
            ),
            "language": args.language,
            "prompt_bias": (
                args.mode in ("whisper_mms", "whisper_mfa") and not args.no_bias
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_segments_count": diag["source_segments_count"],
            "aligned_segments_count": diag["aligned_segments_count"],
        },
        "segments": aligned_segments,
    }


def _variant_key(mode: str, no_bias: bool) -> str:
    # Duplicated from `alignment.schemas.variant_key` on the Mac — this
    # script runs on the remote box and isn't part of the local package.
    if mode == "text_mms":
        return "text_mms"
    return f"{mode}_no_bias" if no_bias else mode


def main() -> int:
    args = parse_args()
    run_dir: Path = args.input_dir
    src_path = run_dir / "conversations.jsonl"
    variant = _variant_key(args.mode, args.no_bias)
    out_path = run_dir / "aligned" / f"{variant}.jsonl"

    if not src_path.is_file():
        logger.error("Source not found: %s", src_path)
        return 2

    convs = _load_jsonl(src_path)
    logger.info(
        "Loaded %d source conversations from %s (variant=%s)",
        len(convs),
        src_path,
        variant,
    )

    # Each variant lives in its own file, so "already aligned" reduces to
    # "this conv_index is already in this file". No need to re-check mode/
    # bias — the filename encodes them.
    existing = {r.get("conv_index", -1): r for r in _load_jsonl(out_path)}

    for idx, conv in enumerate(convs):
        if args.conv_index is not None and idx != args.conv_index:
            continue
        if not args.force and idx in existing:
            logger.info("[conv=%d] cached (variant=%s), skipping", idx, variant)
            continue
        record = _process_conv(args, run_dir, conv, idx)
        existing[idx] = record
        _write_all(out_path, list(existing.values()))  # checkpoint after each conv

    logger.info("Wrote %d aligned conversations to %s", len(existing), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
