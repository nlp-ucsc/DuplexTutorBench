"""Convert MathTutorBench dialogues into our duplex conversation schema.

MathTutorBench (Macina et al., EMNLP 2025; https://github.com/eth-lre/mathtutorbench)
ships its dialogue data as a JSON file `datasets/mathdial_bridge.json` in the repo.
Each example has:
  - `problem`            : the math word problem
  - `reference_solution` : ground-truth worked solution
  - `topic`              : problem category
  - `dialog_history`     : list of {"text": str, "user": "Teacher"|"Student"}

We map Teacher->tutor and Student->student and emit a
`duplex_output/{run_name}/conversations.jsonl` so the existing evaluation
pipeline (e.g. `uv run python -m evaluation run --run-name mathtutorbench
--evaluators llm_judge_bea`) scores it unchanged.

The source is text-only (no audio, no timestamps), so segments get synthetic
1-second-spaced timestamps purely to preserve turn order for the judge's
transcript formatter. Audio-dependent evaluators (stats/turn_taking/naturalness)
are not meaningful here — run only the LLM judges.

Usage:
    uv run python -m evaluation import-mtb                       # 30 sampled dialogues
    uv run python -m evaluation import-mtb --limit 100 --seed 0
    uv run python -m evaluation import-mtb --all                 # all 1150
    uv run python -m evaluation import-mtb --hard                # the _hard split
    uv run python -m evaluation import-mtb --source /path/to/local.json
"""

from __future__ import annotations

import json
import logging
import random
import urllib.request
from pathlib import Path

from duplex.schemas import DuplexConversation, DuplexSegment

logger = logging.getLogger(__name__)

_RAW_BASE = "https://raw.githubusercontent.com/eth-lre/mathtutorbench/main/datasets"
SOURCE_URLS = {
    "main": f"{_RAW_BASE}/mathdial_bridge.json",
    "hard": f"{_RAW_BASE}/mathdial_bridge_hard.json",
}

_ROLE_MAP = {"Teacher": "tutor", "Student": "student"}


def _load_source(source: str) -> list[dict]:
    """Load the raw MathTutorBench JSON from a local path or a URL."""
    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Downloading %s", source)
        with urllib.request.urlopen(source) as r:  # noqa: S310 (trusted GitHub raw)
            return json.loads(r.read().decode())
    return json.loads(Path(source).read_text())


def _to_conversation(idx: int, ex: dict) -> DuplexConversation:
    segments: list[DuplexSegment] = []
    t = 0.0
    for turn in ex.get("dialog_history", []):
        role = _ROLE_MAP.get(turn.get("user", ""), "student")
        text = (turn.get("text", "") or "").strip()
        segments.append(
            DuplexSegment(role=role, text=text, start_time=t, end_time=t + 1.0)
        )
        t += 1.0
    return DuplexConversation(
        pid=str(idx),
        question=ex.get("problem", ""),
        answer=ex.get("reference_solution", ""),
        tutor_voice="mathtutorbench",
        student_voice="mathtutorbench",
        duration=t,
        segments=segments,
        tutor_backend="mathtutorbench",
        student_backend="mathtutorbench",
        metadata={"topic": ex.get("topic", "")},
    )


def convert(
    source: str,
    out_path: Path,
    *,
    limit: int | None = 30,
    seed: int = 0,
) -> int:
    raw = _load_source(source)
    logger.info("Loaded %d MathTutorBench examples", len(raw))

    indexed = list(enumerate(raw))
    if limit is not None and limit < len(indexed):
        random.Random(seed).shuffle(indexed)
        indexed = sorted(indexed[:limit], key=lambda p: p[0])
        logger.info("Sampled %d (seed=%d)", len(indexed), seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for idx, ex in indexed:
            conv = _to_conversation(idx, ex)
            f.write(json.dumps(conv.to_dict()) + "\n")
    logger.info("Wrote %d conversations to %s", len(indexed), out_path)
    return len(indexed)


def add_subparser(subparsers) -> None:
    """Register the `import-mtb` subcommand on the unified CLI."""
    p = subparsers.add_parser(
        "import-mtb",
        help="Convert MathTutorBench dialogues to a duplex conversations.jsonl.",
        description=(
            "Convert MathTutorBench (Macina et al., EMNLP 2025) dialogues into "
            "the duplex conversation schema so existing evaluators (llm_judge / "
            "llm_judge_bea) can score them unchanged."
        ),
    )
    p.add_argument("--run-name", default="mathtutorbench")
    p.add_argument("--duplex-root", default="duplex_output")
    p.add_argument(
        "--source",
        default=None,
        help="Local path or URL to the MathTutorBench JSON (default: GitHub raw).",
    )
    p.add_argument("--hard", action="store_true", help="Use the _hard split.")
    p.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Sample this many dialogues (default 30; controls judge cost).",
    )
    p.add_argument("--all", action="store_true", help="Convert every dialogue.")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(_cmd_fn=_run)


def _run(args) -> None:
    source = args.source or SOURCE_URLS["hard" if args.hard else "main"]
    out_path = Path(args.duplex_root) / args.run_name / "conversations.jsonl"
    limit = None if args.all else args.limit
    n = convert(source, out_path, limit=limit, seed=args.seed)
    print(f"Wrote {n} conversations to {out_path}")
    print(
        "Now score with:\n"
        f"  uv run python -m evaluation run --run-name {args.run_name} "
        "--evaluators llm_judge_bea"
    )
