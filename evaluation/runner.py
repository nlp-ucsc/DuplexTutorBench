"""Load duplex conversations, dispatch to evaluators, write scores.jsonl."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from duplex.storage import load_conversations
from evaluation.base import EvaluationResult, Evaluator

logger = logging.getLogger(__name__)


def _judge_kwargs(kwargs: dict) -> dict:
    """Collect the LLM-judge ctor kwargs (model + provider) from runner kwargs."""
    jk: dict = {}
    if kwargs.get("judge_model"):
        jk["model"] = kwargs["judge_model"]
    if kwargs.get("judge_provider"):
        jk["provider_name"] = kwargs["judge_provider"]
    return jk


def build_evaluators(names: Iterable[str], **kwargs) -> list[Evaluator]:
    """Instantiate the named evaluators.

    Construction is lazy-imported so that, for example, asking for `stats`
    alone doesn't trigger the OpenAI client init that `llm_judge` performs.
    """
    out: list[Evaluator] = []
    for n in names:
        if n == "stats":
            from evaluation.evaluators.stats import ConversationStats

            out.append(ConversationStats())
        elif n == "turn_taking":
            from evaluation.evaluators.turn_taking import TurnTaking

            out.append(TurnTaking())
        elif n == "naturalness":
            from evaluation.evaluators.naturalness import SpeechNaturalness

            out.append(SpeechNaturalness())
        elif n == "llm_judge":
            from evaluation.evaluators.llm_judge import LLMJudge

            out.append(LLMJudge(**_judge_kwargs(kwargs)))
        elif n == "llm_judge_bea":
            from evaluation.evaluators.llm_judge_bea import LLMJudgeBEA

            out.append(LLMJudgeBEA(**_judge_kwargs(kwargs)))
        else:
            raise ValueError(
                f"Unknown evaluator: {n!r} (known: stats, turn_taking, naturalness, llm_judge, llm_judge_bea)"
            )
    return out


def _load_existing(scores_path: Path) -> dict[int, EvaluationResult]:
    """Read previous results so we can merge per-evaluator updates."""
    by_idx: dict[int, EvaluationResult] = {}
    if not scores_path.is_file():
        return by_idx
    with scores_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            er = EvaluationResult(
                pid=str(d.get("pid", "")),
                conv_index=int(d.get("conv_index", -1)),
                timestamp=str(d.get("timestamp", "")),
                evaluator_results=dict(d.get("evaluator_results", {})),
            )
            by_idx[er.conv_index] = er
    return by_idx


def _write_all(scores_path: Path, results: list[EvaluationResult]) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: full rewrite each run — file is small (~one short
    # JSON line per conversation, ~hundreds per dataset).
    tmp = scores_path.with_suffix(scores_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for er in sorted(results, key=lambda r: r.conv_index):
            f.write(json.dumps(er.to_dict()) + "\n")
    tmp.replace(scores_path)


def run_evaluation(
    run_name: str,
    evaluator_names: Iterable[str],
    *,
    duplex_root: Path = Path("duplex_output"),
    eval_root: Path = Path("eval_output"),
    conv_index: int | None = None,
    force: bool = False,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    align_variant: str | None = None,
    eval_subdir: str | None = None,
) -> Path:
    """Evaluate one duplex run, writing/updating scores.jsonl.

    When `align_variant` is set (e.g. ``"whisper_mms"`` or
    ``"whisper_mms_no_bias"``), source from
    ``<run_dir>/aligned/<variant>.jsonl`` (produced by the `alignment`
    module) and write scores to ``eval_output/{run_name}__{variant}/`` so
    the original and each aligned variant accumulate side-by-side without
    overwriting each other.

    Returns the path to the written scores.jsonl.
    """
    run_dir = duplex_root / run_name
    if align_variant:
        jsonl_path = run_dir / "aligned" / f"{align_variant}.jsonl"
    else:
        jsonl_path = run_dir / "conversations.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"No conversations file at {jsonl_path}")

    if eval_subdir is None:
        eval_subdir = f"{run_name}__{align_variant}" if align_variant else run_name
    eval_dir = eval_root / eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    scores_path = eval_dir / "scores.jsonl"

    conversations = load_conversations(jsonl_path)
    logger.info("Loaded %d conversations from %s", len(conversations), jsonl_path)

    kwargs: dict = {}
    if judge_model:
        kwargs["judge_model"] = judge_model
    if judge_provider:
        kwargs["judge_provider"] = judge_provider
    evaluators = build_evaluators(list(evaluator_names), **kwargs)
    logger.info("Running evaluators: %s", [e.name for e in evaluators])

    existing = _load_existing(scores_path)
    now = datetime.now(timezone.utc).isoformat()

    for pos, conv in enumerate(conversations):
        # Aligned files are sparse (one record per processed conv) and carry
        # an explicit `conv_index`; original files are dense, so the list
        # position is the index by convention.
        idx = int(conv.get("conv_index", pos))
        if conv_index is not None and idx != conv_index:
            continue
        pid = str(conv.get("pid", ""))
        audio_dir = run_dir / "audio" / str(idx)
        prev = existing.get(idx) or EvaluationResult(
            pid=pid, conv_index=idx, timestamp=now
        )
        prev.pid = pid  # in case the JSONL was regenerated
        prev.conv_index = idx
        for ev in evaluators:
            cached = prev.evaluator_results.get(ev.name)
            # Re-run cached failures automatically; only "ok"/"skipped" are
            # treated as done (transient API errors shouldn't stick forever).
            cached_done = cached is not None and cached.get("status") in (
                None,
                "ok",
                "skipped",
            )
            if not force and cached_done:
                logger.info("[%d] pid=%s :: %s -- cached, skipping", idx, pid, ev.name)
                continue
            if ev.requires_audio and not audio_dir.is_dir():
                logger.warning(
                    "[%d] pid=%s :: %s -- audio dir missing (%s), skipping",
                    idx,
                    pid,
                    ev.name,
                    audio_dir,
                )
                prev.evaluator_results[ev.name] = {
                    "status": "skipped",
                    "reason": f"audio dir missing: {audio_dir}",
                }
                continue
            logger.info("[%d] pid=%s :: %s -- evaluating", idx, pid, ev.name)
            try:
                prev.evaluator_results[ev.name] = ev.evaluate(conv, audio_dir)
            except Exception as e:
                logger.exception(
                    "[%d] pid=%s :: %s -- evaluator raised", idx, pid, ev.name
                )
                prev.evaluator_results[ev.name] = {"status": "error", "error": str(e)}
        prev.timestamp = datetime.now(timezone.utc).isoformat()
        existing[idx] = prev

    _write_all(scores_path, list(existing.values()))
    logger.info("Wrote %d results to %s", len(existing), scores_path)
    return scores_path
