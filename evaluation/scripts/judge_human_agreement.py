"""Measure judge–human agreement via MathTutorBench preference pairs.

For each pair (teacher_response_positive vs teacher_response_negative) from
dmacjam/pedagogical-rewardmodel-data we ask each judge to score two short
conversations that share the same dialogue prefix but differ only in the final
tutor turn.  The judge "agrees with humans" when it scores the positive turn
higher.

Metric: pairwise accuracy = (#correct + 0.5·#ties) / N  (with Wilson 95% CI)

Usage:
    uv run python evaluation/scripts/judge_human_agreement.py \\
        --split mrbench_train --limit 50 \\
        --judge-provider openai --rubrics llm_judge_bea

    # full sweep (1200 calls, ~$1-2)
    uv run python evaluation/scripts/judge_human_agreement.py \\
        --split mrbench_train --limit 100 \\
        --judge-provider openai gemini claude

    # re-run skips cached ok rows automatically
    # use --force to recompute everything
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_OUT_ROOT = _REPO_ROOT / "eval_output" / "human_agreement"

# Map HF dataset role labels to our internal roles.
_ROLE_MAP = {"Teacher": "tutor", "Student": "student"}

# Comparison keys extracted from each rubric to score preference agreement.
_COMPARE_KEYS = {
    "llm_judge": ["scaffolding_quality", "overall", "mean_component_score"],
    "llm_judge_bea": ["guidance_quality", "actionability", "mean_component_score"],
}


# ---------------------------------------------------------------------------
# Wilson 95% binomial CI (pure stdlib)
# ---------------------------------------------------------------------------


def _wilson_ci(k: float, n: int) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (k may be non-integer for ties)."""
    if n == 0:
        return 0.0, 1.0
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Build DuplexConversation-compatible dicts from a preference row
# ---------------------------------------------------------------------------


def _build_conv(row: dict, side: str) -> dict:
    """Return a conv dict (same schema as conversations.jsonl) for one side of the pair.

    side='pos'  → appends teacher_response_positive as the final tutor turn
    side='neg'  → appends teacher_response_negative
    """
    from duplex.schemas import DuplexConversation, DuplexSegment

    turns = list(row.get("dialog_history") or [])
    response = (
        row["teacher_response_positive"]
        if side == "pos"
        else row["teacher_response_negative"]
    )

    segments: list[DuplexSegment] = []
    t = 0.0
    for turn in turns:
        role = _ROLE_MAP.get(turn.get("user", ""), "student")
        text = (turn.get("text", "") or "").strip()
        segments.append(
            DuplexSegment(role=role, text=text, start_time=t, end_time=t + 1.0)
        )
        t += 1.0

    # Append the candidate tutor response as the final turn.
    segments.append(
        DuplexSegment(
            role="tutor", text=(response or "").strip(), start_time=t, end_time=t + 1.0
        )
    )

    conv = DuplexConversation(
        pid=str(row.get("_idx", 0)),
        question=row.get("problem", ""),
        answer=row.get("reference_solution", ""),
        tutor_voice="preference_pair",
        student_voice="preference_pair",
        duration=t + 1.0,
        segments=segments,
        tutor_backend="preference_pair",
        student_backend="preference_pair",
    )
    return conv.to_dict()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_cache(cache_path: Path) -> dict[tuple, dict]:
    cache: dict[tuple, dict] = {}
    if not cache_path.exists():
        return cache
    for line in cache_path.read_text().splitlines():
        try:
            rec = json.loads(line)
            key = (
                rec["split"],
                rec["row_idx"],
                rec["side"],
                rec["rubric"],
                rec["judge_model"],
            )
            cache[key] = rec
        except Exception:
            pass
    return cache


def _save_cache_entry(cache_path: Path, rec: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _rebuild_cache(cache_path: Path, entries: dict[tuple, dict]) -> None:
    """Rewrite cache atomically (used with --force or on startup to deduplicate)."""
    tmp = cache_path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w") as f:
        for rec in entries.values():
            f.write(json.dumps(rec) + "\n")
    tmp.replace(cache_path)


# ---------------------------------------------------------------------------
# Core scoring loop
# ---------------------------------------------------------------------------


def _get_score(result: dict, key: str) -> float | None:
    if result.get("status") != "ok":
        return None
    if key == "mean_component_score":
        v = result.get("mean_component_score")
        return float(v) if v is not None else None
    scores = result.get("scores", {})
    entry = scores.get(key)
    if isinstance(entry, dict) and "score" in entry:
        return float(entry["score"])
    return None


def run(
    split: str,
    limit: int,
    seed: int,
    judge_providers: list[str],
    rubrics: list[str],
    force: bool,
) -> None:
    from datasets import load_dataset

    from evaluation.evaluators.llm_judge import LLMJudge
    from evaluation.evaluators.llm_judge_bea import LLMJudgeBEA
    from evaluation.evaluators.providers import DEFAULT_MODELS

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = _OUT_ROOT / "cache.jsonl"
    cache = {} if force else _load_cache(cache_path)
    if force and cache_path.exists():
        cache_path.unlink()
        cache = {}

    logger.info("Loading dmacjam/pedagogical-rewardmodel-data split=%s", split)
    ds = load_dataset("dmacjam/pedagogical-rewardmodel-data", split=split)
    rows = list(ds)

    if limit and limit < len(rows):
        rng = random.Random(seed)
        indices = rng.sample(range(len(rows)), limit)
        rows = [rows[i] for i in sorted(indices)]

    # Attach stable row index for cache keys (position in the sampled list).
    for i, row in enumerate(rows):
        row["_idx"] = i

    logger.info("Using %d preference pairs from split '%s'", len(rows), split)

    # Build judge instances per (provider, rubric) combo.
    judges: list[tuple[str, str, object]] = []
    for provider in judge_providers:
        model = DEFAULT_MODELS.get(provider, "gpt-5-mini")
        for rubric in rubrics:
            if rubric == "llm_judge":
                j = LLMJudge(provider_name=provider, model=model)
            else:
                j = LLMJudgeBEA(provider_name=provider, model=model)
            judges.append((model, rubric, j))
            logger.info("Judge: model=%s rubric=%s", model, rubric)

    # Score every (row, side, judge) triple; use cache where possible.
    total = len(rows) * 2 * len(judges)
    done = 0
    for row in rows:
        for side in ("pos", "neg"):
            for model, rubric, judge in judges:
                key = (split, row["_idx"], side, rubric, model)
                if key in cache and cache[key].get("status") == "ok":
                    done += 1
                    continue
                conv = _build_conv(row, side)
                result = judge.evaluate(conv, audio_dir=_OUT_ROOT)
                rec = {
                    "split": split,
                    "row_idx": row["_idx"],
                    "side": side,
                    "rubric": rubric,
                    "judge_model": model,
                    **result,
                }
                cache[key] = rec
                _save_cache_entry(cache_path, rec)
                done += 1
                if done % 20 == 0:
                    logger.info("Progress: %d/%d", done, total)

    # Rebuild cache to remove duplicates from incremental appends.
    _rebuild_cache(cache_path, cache)

    # Aggregate results per (model, rubric, comparison_key).
    summary_rows: list[dict] = []
    pairs_by_judge: dict[tuple, list[dict]] = {}

    for model, rubric, _ in judges:
        compare_keys = _COMPARE_KEYS.get(rubric, ["mean_component_score"])
        for ckey in compare_keys:
            records = []
            correct = 0.0
            n = 0
            ties = 0
            margins = []

            for row in rows:
                pos_key = (split, row["_idx"], "pos", rubric, model)
                neg_key = (split, row["_idx"], "neg", rubric, model)
                pos_res = cache.get(pos_key, {})
                neg_res = cache.get(neg_key, {})
                s_pos = _get_score(pos_res, ckey)
                s_neg = _get_score(neg_res, ckey)
                if s_pos is None or s_neg is None:
                    continue
                margin = s_pos - s_neg
                margins.append(margin)
                n += 1
                if s_pos > s_neg:
                    verdict = "correct"
                    correct += 1.0
                elif s_pos == s_neg:
                    verdict = "tie"
                    correct += 0.5
                    ties += 1
                else:
                    verdict = "wrong"
                records.append(
                    {
                        "split": split,
                        "row_idx": row["_idx"],
                        "rubric": rubric,
                        "judge_model": model,
                        "comparison_key": ckey,
                        "score_pos": s_pos,
                        "score_neg": s_neg,
                        "margin": margin,
                        "verdict": verdict,
                    }
                )

            pairs_by_judge[(model, rubric, ckey)] = records

            if n == 0:
                continue
            accuracy = correct / n
            ci_lo, ci_hi = _wilson_ci(correct, n)
            tie_rate = ties / n
            mean_margin = sum(margins) / len(margins) if margins else 0.0
            summary_rows.append(
                {
                    "split": split,
                    "n": n,
                    "judge_model": model,
                    "rubric": rubric,
                    "comparison_key": ckey,
                    "accuracy": round(accuracy, 4),
                    "ci_low": round(ci_lo, 4),
                    "ci_high": round(ci_hi, 4),
                    "tie_rate": round(tie_rate, 4),
                    "mean_margin": round(mean_margin, 4),
                }
            )

    # Write per-judge pairs files.
    for (model, rubric, ckey), records in pairs_by_judge.items():
        safe = model.replace("/", "_").replace(".", "-")
        out = _OUT_ROOT / f"pairs_{safe}_{rubric}_{ckey}.jsonl"
        with out.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    # Write summary CSV.
    summary_csv = _REPO_ROOT / "eval_output" / "human_agreement_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)

    # Print summary table.
    print(
        f"\n{'judge_model':<22} {'rubric':<18} {'comparison_key':<26} {'n':>5} "
        f"{'accuracy':>9} {'95% CI':>16} {'tie%':>6} {'margin':>8}"
    )
    print("-" * 118)
    for r in summary_rows:
        ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        print(
            f"{r['judge_model']:<22} {r['rubric']:<18} {r['comparison_key']:<26} "
            f"{r['n']:>5} {r['accuracy']:>9.3f} {ci:>16} "
            f"{r['tie_rate']:>6.3f} {r['mean_margin']:>8.3f}"
        )
    print()
    if summary_rows:
        print(f"Summary written to {summary_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure LLM judge agreement with MathTutorBench human preference pairs."
    )
    p.add_argument(
        "--split",
        default="mrbench_train",
        help="HF split to use. Use mrbench_train (human) or test. "
        "Avoid gsm8k_inpainted_train (synthetic) for human-agreement claims.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of pairs to sample (default 100). Use 0 for all.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--judge-provider",
        nargs="+",
        choices=["openai", "gemini", "claude"],
        default=["openai"],
        dest="judge_providers",
    )
    p.add_argument(
        "--rubrics",
        nargs="+",
        choices=["llm_judge", "llm_judge_bea"],
        default=["llm_judge", "llm_judge_bea"],
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute all judgements, ignoring the cache.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    limit = args.limit if args.limit > 0 else None
    run(
        split=args.split,
        limit=limit,
        seed=args.seed,
        judge_providers=args.judge_providers,
        rubrics=args.rubrics,
        force=args.force,
    )


if __name__ == "__main__":
    main()
