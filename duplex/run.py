"""CLI entry point for full-duplex conversations.

Usage:
    # Batch mode: run a single question (PersonaPlex)
    uv run python -m duplex --run-name test --tutor-backend personaplex --pid 42

    # Batch mode: GPT Realtime
    uv run python -m duplex --run-name test --tutor-backend gpt-realtime --student-backend gpt-realtime --pid 42

    # Mixed backends
    uv run python -m duplex --run-name test --tutor-backend personaplex --student-backend gpt-realtime --pid 42

    # Web UI mode
    uv run python -m duplex --run-name test --web --port 5002
"""

import argparse
import asyncio
import io
import logging
import os
import random
import shutil
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from duplex.backend import (
    BACKEND_CHOICES,
    ModelBackend,
    PersonaPlexBackend,
    uses_image,
)
from duplex.manifest import (
    args_from_manifest,
    build_manifest,
    load_manifest,
    manifest_path,
    validate_against_manifest,
    write_manifest,
)
from duplex.relay import DuplexRelay
from duplex.schemas import DuplexConversation
from duplex.storage import (
    load_done_set,
    save_audio,
    save_audio_wav,
    save_audio_wav_stereo,
    save_conversation,
)
from duplex.web import DuplexWebApp
from src.conversation import _format_prompt
from src.data import load_mathvista
from src.schemas import MathVistaQuestion

CLOUD_BACKENDS = {"gpt-realtime", "gemini-live"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _pil_to_png_bytes(image) -> bytes:
    """Encode a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-duplex student-tutor conversations."
    )

    # Backend selection (per-role for mixed-backend support)
    parser.add_argument(
        "--tutor-backend",
        choices=BACKEND_CHOICES,
        default="personaplex",
        help="Backend for the tutor role",
    )
    parser.add_argument(
        "--student-backend",
        choices=BACKEND_CHOICES,
        default="personaplex",
        help="Backend for the student role",
    )

    # PersonaPlex server connection
    pp = parser.add_argument_group("PersonaPlex options")
    pp.add_argument("--tutor-host", default="100.116.140.1", help="Tutor server host")
    pp.add_argument("--tutor-port", type=int, default=8998, help="Tutor server port")
    pp.add_argument(
        "--student-host", default="100.116.140.1", help="Student server host"
    )
    pp.add_argument(
        "--student-port", type=int, default=8999, help="Student server port"
    )
    pp.add_argument("--tutor-voice", default="NATM1.pt", help="Tutor voice prompt file")
    pp.add_argument(
        "--student-voice", default="NATF2.pt", help="Student voice prompt file"
    )

    # GPT Realtime options
    gpt = parser.add_argument_group("GPT Realtime options")
    gpt.add_argument(
        "--gpt-model",
        default="gpt-realtime",
        help="GPT Realtime model name",
    )
    gpt.add_argument("--tutor-gpt-voice", default="ash", help="GPT voice for tutor")
    gpt.add_argument(
        "--student-gpt-voice", default="shimmer", help="GPT voice for student"
    )

    # MoshiVis options
    mv = parser.add_argument_group("MoshiVis options")
    mv.add_argument(
        "--moshivis-host", default="100.116.140.1", help="MoshiVis server host"
    )
    mv.add_argument(
        "--moshivis-tutor-port",
        type=int,
        default=8088,
        help="MoshiVis tutor server port",
    )
    mv.add_argument(
        "--moshivis-student-port",
        type=int,
        default=8089,
        help="MoshiVis student server port",
    )

    # Gemini Live options
    gem = parser.add_argument_group("Gemini Live options")
    gem.add_argument(
        "--gemini-model",
        default="gemini-3.1-flash-live-preview",
        help="Gemini Live model name",
    )
    gem.add_argument(
        "--tutor-gemini-voice", default="Charon", help="Gemini voice for tutor"
    )
    gem.add_argument(
        "--student-gemini-voice", default="Leda", help="Gemini voice for student"
    )

    # Prompts
    parser.add_argument(
        "--tutor-prompt",
        default="prompts/duplex_tutor_system.txt",
        help="Path to tutor prompt template",
    )
    parser.add_argument(
        "--student-prompt",
        default="prompts/duplex_student_system.txt",
        help="Path to student prompt template",
    )

    # Conversation control
    parser.add_argument(
        "--max-duration",
        type=int,
        default=300,
        help="Max conversation duration in seconds",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Name for this run (duplex_output/{run_name}/). Required for "
            "batch mode; optional for --web (the dropdown can pick or "
            "create one after launch)."
        ),
    )
    parser.add_argument("--split", default="testmini", help="MathVista split")
    parser.add_argument(
        "--pid",
        type=str,
        nargs="+",
        help="One or more question PIDs to run (e.g. --pid 1 2 4)",
    )
    parser.add_argument(
        "--n", type=int, default=None, help="First N questions in dataset order"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Random N questions (mutually exclusive with --n / --pid)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed used for --sample"
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Independent generations per question (default 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel conversations (cloud-API backends only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override manifest mismatch on resume",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Replay from a manifest.json (overrides most other batch flags)",
    )

    # Web UI mode
    parser.add_argument(
        "--web", action="store_true", help="Launch web UI instead of CLI batch"
    )
    parser.add_argument("--port", type=int, default=5002, help="Web UI port")

    return parser.parse_args()


def create_backend(
    backend_type: str,
    role: str,
    prompt: str,
    args: argparse.Namespace,
    image_bytes: bytes | None = None,
) -> ModelBackend:
    """Create a ModelBackend instance for the given role."""
    if backend_type == "moshivis":
        from duplex.backend_moshivis import MoshiVisBackend

        if image_bytes is None:
            raise ValueError(
                f"MoshiVis backend ({role}) requires an image — "
                "this question has no image."
            )
        port = (
            args.moshivis_tutor_port if role == "tutor" else args.moshivis_student_port
        )
        return MoshiVisBackend(
            host=args.moshivis_host,
            port=port,
            image_bytes=image_bytes,
            role=role,
        )
    if backend_type == "personaplex":
        host = args.tutor_host if role == "tutor" else args.student_host
        port = args.tutor_port if role == "tutor" else args.student_port
        voice = args.tutor_voice if role == "tutor" else args.student_voice
        return PersonaPlexBackend(
            host=host,
            port=port,
            text_prompt=prompt,
            voice_prompt=voice,
            role=role,
        )
    elif backend_type == "gpt-realtime":
        from duplex.backend_gpt import GPTRealtimeBackend

        voice = args.tutor_gpt_voice if role == "tutor" else args.student_gpt_voice
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for gpt-realtime backend"
            )
        return GPTRealtimeBackend(
            instructions=prompt,
            role=role,
            voice=voice,
            model=args.gpt_model,
            api_key=api_key,
            initiate=(role == "student"),
            image_bytes=image_bytes,
        )
    elif backend_type == "gemini-live":
        from duplex.backend_gemini import GeminiLiveBackend

        voice = (
            args.tutor_gemini_voice if role == "tutor" else args.student_gemini_voice
        )
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for gemini-live backend"
            )
        return GeminiLiveBackend(
            instructions=prompt,
            role=role,
            voice=voice,
            model=args.gemini_model,
            api_key=api_key,
            initiate=(role == "student"),
            image_bytes=image_bytes,
        )
    elif backend_type == "human":
        from duplex.backend_human import HumanBackend

        return HumanBackend(role=role)
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


def get_voice_label(backend_type: str, role: str, args: argparse.Namespace) -> str:
    """Get the voice label for a given role and backend."""
    if backend_type == "personaplex":
        return args.tutor_voice if role == "tutor" else args.student_voice
    elif backend_type == "gpt-realtime":
        voice = args.tutor_gpt_voice if role == "tutor" else args.student_gpt_voice
        return voice or "default"
    elif backend_type == "gemini-live":
        voice = (
            args.tutor_gemini_voice if role == "tutor" else args.student_gemini_voice
        )
        return voice or "Kore"
    elif backend_type == "moshivis":
        return "moshika"
    elif backend_type == "human":
        return "human"
    return ""


def _question_image_bytes(question: MathVistaQuestion) -> bytes | None:
    """Return PNG bytes for a question's image, or None if it has none."""
    if question.image is None:
        return None
    return _pil_to_png_bytes(question.image)


def _resolve_questions(
    args: argparse.Namespace,
) -> list[MathVistaQuestion]:
    """Load and filter MathVista questions per --pid / --n / --sample / --seed."""
    if args.sample is not None and args.n is not None:
        raise SystemExit("--sample and --n are mutually exclusive.")
    if args.sample is not None and args.pid:
        raise SystemExit("--sample and --pid are mutually exclusive.")

    if args.sample is not None:
        questions = load_mathvista(split=args.split, n=None)
        rng = random.Random(args.seed)
        questions = rng.sample(questions, k=min(args.sample, len(questions)))
    else:
        questions = load_mathvista(split=args.split, n=args.n)

    if args.pid:
        wanted = list(dict.fromkeys(args.pid))  # de-dupe, preserve order
        by_pid = {q.pid: q for q in questions}
        missing = [pid for pid in wanted if pid not in by_pid]
        if missing:
            raise SystemExit(f"PID(s) not found in {args.split}: {', '.join(missing)}")
        questions = [by_pid[pid] for pid in wanted]
    return questions


def _build_work_plan(
    questions: list[MathVistaQuestion],
    attempts: int,
    done: set[tuple[str, int]],
) -> list[tuple[MathVistaQuestion, int]]:
    """Cartesian (question x attempt), pid-outer / attempt-inner, minus done pairs."""
    plan: list[tuple[MathVistaQuestion, int]] = []
    for q in questions:
        for attempt in range(attempts):
            if (str(q.pid), attempt) in done:
                continue
            plan.append((q, attempt))
    return plan


def _next_jsonl_idx(output_path: Path) -> int:
    if not output_path.is_file():
        return 0
    with open(output_path) as f:
        return sum(1 for line in f if line.strip())


async def _run_one_conversation(
    args: argparse.Namespace,
    question: MathVistaQuestion,
    attempt_index: int,
    tutor_prompt_template: str,
    student_prompt_template: str,
    run_dir: Path,
    output_path: Path,
    write_lock: asyncio.Lock,
    progress: dict,
) -> None:
    """Run one conversation, save it under an atomically-allocated idx."""
    moshivis_in_use = "moshivis" in (args.tutor_backend, args.student_backend)
    image_bytes = _question_image_bytes(question)
    if moshivis_in_use and image_bytes is None:
        logger.warning(
            "pid=%s attempt=%d -- skipped: MoshiVis requires an image.",
            question.pid,
            attempt_index,
        )
        return

    tutor_prompt = _format_prompt(tutor_prompt_template, question)
    student_prompt = _format_prompt(student_prompt_template, question)

    tutor_backend = create_backend(
        args.tutor_backend, "tutor", tutor_prompt, args, image_bytes=image_bytes
    )
    student_backend = create_backend(
        args.student_backend,
        "student",
        student_prompt,
        args,
        image_bytes=image_bytes,
    )

    relay = DuplexRelay(
        tutor_backend=tutor_backend,
        student_backend=student_backend,
        max_duration=args.max_duration,
    )

    segments = await relay.run()
    duration = relay.elapsed
    pcm_bufs = relay.audio_buffers
    native_bufs = relay.native_audio_buffers
    audio_offsets = relay.audio_start_offsets

    # Cloud backends (notably gemini-live) occasionally connect, exchange
    # heartbeats, and disconnect on silence_timeout without ever producing
    # audio or text. Skip the save so this (pid, attempt) pair is left
    # un-done and the next resume retries it instead of permanently
    # recording it as an empty row.
    if not segments and not pcm_bufs.get("tutor") and not pcm_bufs.get("student"):
        logger.warning(
            "pid=%s attempt=%d -- empty conversation (no audio, no segments) "
            "after %.1fs; skipping save (will retry on next run).",
            question.pid,
            attempt_index,
            duration,
        )
        return

    metadata = dict(question.metadata)
    conversation = DuplexConversation(
        pid=question.pid,
        question=question.question,
        answer=question.answer,
        tutor_voice=get_voice_label(args.tutor_backend, "tutor", args),
        student_voice=get_voice_label(args.student_backend, "student", args),
        duration=duration,
        segments=segments,
        metadata=metadata,
        tutor_backend=args.tutor_backend,
        student_backend=args.student_backend,
        tutor_uses_image=uses_image(args.tutor_backend, image_bytes),
        student_uses_image=uses_image(args.student_backend, image_bytes),
        attempt_index=attempt_index,
    )

    # Allocate idx + write JSONL + audio under the lock so concurrent
    # conversations don't race for the same idx.
    async with write_lock:
        idx = _next_jsonl_idx(output_path)
        audio_dir = run_dir / "audio" / str(idx)
        if audio_dir.exists():
            # Orphaned dir from a previous interrupted run at this idx.
            shutil.rmtree(audio_dir)

        for role in ("tutor", "student"):
            backend_obj = tutor_backend if role == "tutor" else student_backend
            ext = backend_obj.get_native_audio_ext()
            if pcm_bufs.get(role):
                save_audio_wav(
                    pcm_bufs[role],
                    audio_dir / f"{role}_full.wav",
                    total_duration_s=duration,
                    offset_s=audio_offsets[role],
                )
            if native_bufs.get(role):
                save_audio(native_bufs[role], audio_dir / f"{role}_full{ext}")

        save_audio_wav_stereo(
            pcm_bufs.get("tutor", b""),
            pcm_bufs.get("student", b""),
            audio_dir / "combined.wav",
            total_duration_s=duration,
            left_offset_s=audio_offsets["tutor"],
            right_offset_s=audio_offsets["student"],
        )
        save_conversation(conversation, output_path)
        progress["done"] += 1
        pbar = progress.get("pbar")
        msg = (
            f"pid={question.pid} attempt={attempt_index} -- "
            f"{duration:.1f}s, {len(segments)} segments (idx={idx})"
        )
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"pid={question.pid} a{attempt_index}")
            tqdm.write(msg)
        else:
            logger.info(msg)


async def run_batch(
    args: argparse.Namespace,
    tutor_prompt_template: str | None = None,
    student_prompt_template: str | None = None,
) -> None:
    """Run conversations in batch mode (CLI).

    If `tutor_prompt_template` / `student_prompt_template` are provided (replay
    case), they are used directly instead of being read from disk.
    """
    if tutor_prompt_template is None:
        tutor_prompt_template = Path(args.tutor_prompt).read_text()
    if student_prompt_template is None:
        student_prompt_template = Path(args.student_prompt).read_text()

    run_dir = Path("duplex_output") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "conversations.jsonl"

    # Concurrency is only safe for cloud-API backends.
    if args.concurrency > 1:
        if (
            args.tutor_backend not in CLOUD_BACKENDS
            or args.student_backend not in CLOUD_BACKENDS
        ):
            raise SystemExit(
                f"--concurrency > 1 requires both backends to be one of "
                f"{sorted(CLOUD_BACKENDS)}; got "
                f"tutor={args.tutor_backend}, student={args.student_backend}."
            )

    questions = _resolve_questions(args)
    target_pids = [q.pid for q in questions]

    # Manifest: write on first run, validate on subsequent runs.
    mpath = manifest_path(run_dir)
    manifest = build_manifest(
        args=args,
        tutor_prompt_template=tutor_prompt_template,
        student_prompt_template=student_prompt_template,
        target_pids=target_pids,
    )
    if mpath.is_file():
        existing = load_manifest(mpath)
        mismatches = validate_against_manifest(
            existing,
            args,
            tutor_prompt_template,
            student_prompt_template,
            target_pids,
        )
        if mismatches:
            msg = (
                f"Manifest mismatch in {mpath}:\n"
                + "\n".join(mismatches)
                + "\nPass --force to override (will not rewrite the manifest)."
            )
            if not args.force:
                raise SystemExit(msg)
            logger.warning("%s", msg)
        # Manifest already exists and either matches or --force was passed.
        # Don't rewrite — the on-disk manifest stays as the canonical record.
    else:
        write_manifest(run_dir, manifest)

    done = load_done_set(output_path)
    work_plan = _build_work_plan(questions, args.attempts, done)
    total_target = len(questions) * args.attempts
    skipped = total_target - len(work_plan)

    logger.info(
        "Batch: %d questions x %d attempts = %d target rows; %d already done; "
        "%d to run -> %s",
        len(questions),
        args.attempts,
        total_target,
        skipped,
        len(work_plan),
        output_path,
    )
    logger.info(
        "Backends: tutor=%s, student=%s, concurrency=%d",
        args.tutor_backend,
        args.student_backend,
        args.concurrency,
    )

    if not work_plan:
        logger.info("Nothing to do.")
        return

    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(work_plan), desc="duplex", unit="conv", dynamic_ncols=True)
    progress = {"done": 0, "total": len(work_plan), "pbar": pbar}

    async def _guarded(question: MathVistaQuestion, attempt: int) -> None:
        try:
            await _run_one_conversation(
                args,
                question,
                attempt,
                tutor_prompt_template,
                student_prompt_template,
                run_dir,
                output_path,
                write_lock,
                progress,
            )
        except Exception:
            logger.exception(
                "pid=%s attempt=%d -- failed, skipping.",
                question.pid,
                attempt,
            )

    try:
        if args.concurrency <= 1:
            for q, attempt in work_plan:
                await _guarded(q, attempt)
        else:
            sem = asyncio.Semaphore(args.concurrency)

            async def _bounded(q: MathVistaQuestion, attempt: int) -> None:
                async with sem:
                    await _guarded(q, attempt)

            await asyncio.gather(*(_bounded(q, a) for q, a in work_plan))
    finally:
        pbar.close()

    logger.info("Done. Output written to %s", output_path)


def run_web(args: argparse.Namespace) -> None:
    """Launch the web UI for live monitoring."""
    tutor_prompt_template = Path(args.tutor_prompt).read_text()
    student_prompt_template = Path(args.student_prompt).read_text()

    duplex_root = Path("duplex_output")
    duplex_root.mkdir(parents=True, exist_ok=True)
    # `--run-name` is optional in web mode; the dropdown can pick or create
    # a run after launch. If provided, pre-create + select it so the user
    # lands directly in that run. Otherwise, default to the most recently
    # modified existing run so the dropdown's visible selection matches
    # self.run_dir on first load (without this, the browser visually selects
    # the first option but no change event fires, leaving run_dir=None and
    # History/Eval/Align panels empty until the user toggles the dropdown).
    if args.run_name:
        run_dir: Path | None = duplex_root / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        existing = sorted(
            (
                p
                for p in duplex_root.iterdir()
                if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        run_dir = existing[0] if existing else None

    def backend_factory(
        role: str,
        prompt: str,
        backend_type: str,
        image_bytes: bytes | None = None,
    ) -> ModelBackend:
        return create_backend(backend_type, role, prompt, args, image_bytes=image_bytes)

    def voice_factory(backend_type: str, role: str) -> str:
        return get_voice_label(backend_type, role, args)

    web_app = DuplexWebApp(
        backend_factory=backend_factory,
        voice_factory=voice_factory,
        tutor_backend_type=args.tutor_backend,
        student_backend_type=args.student_backend,
        tutor_prompt_template=tutor_prompt_template,
        student_prompt_template=student_prompt_template,
        run_dir=run_dir,
        duplex_root=duplex_root,
        max_duration=args.max_duration,
        split=args.split,
        n_questions=args.n,
    )

    app = web_app.create_app()
    logger.info("Starting web UI at http://localhost:%d", args.port)
    web.run_app(app, port=args.port, print=None)


def main() -> None:
    args = parse_args()

    if args.replay:
        if args.web:
            raise SystemExit("--replay is batch-only; cannot combine with --web.")
        if not args.run_name:
            raise SystemExit("--run-name is required for --replay.")
        manifest = load_manifest(Path(args.replay))
        replay_args, tutor_tpl, student_tpl = args_from_manifest(
            manifest, run_name=args.run_name
        )
        # Allow the user to override --concurrency on replay (operational only).
        if args.concurrency and args.concurrency != 1:
            replay_args.concurrency = args.concurrency
        logger.info(
            "Replaying manifest %s into run_name=%s", args.replay, args.run_name
        )
        asyncio.run(
            run_batch(
                replay_args,
                tutor_prompt_template=tutor_tpl,
                student_prompt_template=student_tpl,
            )
        )
        return

    if args.tutor_backend == "human" and args.student_backend == "human":
        raise SystemExit("At most one role can be 'human' (not both).")
    if not args.web and "human" in (args.tutor_backend, args.student_backend):
        raise SystemExit(
            "The 'human' backend requires --web (microphone input is browser-driven)."
        )
    if not args.web and not args.run_name:
        raise SystemExit("--run-name is required in batch mode.")

    if args.web:
        ignored = []
        if args.attempts != 1:
            ignored.append(f"--attempts={args.attempts}")
        if args.sample is not None:
            ignored.append(f"--sample={args.sample}")
        if args.concurrency != 1:
            ignored.append(f"--concurrency={args.concurrency}")
        if args.force:
            ignored.append("--force")
        if ignored:
            logger.warning(
                "Ignoring batch-only flags in --web mode: %s", " ".join(ignored)
            )
        run_web(args)
    else:
        asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
