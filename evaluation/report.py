"""Generate human-readable per-dialog text reports from evaluation scores.

Usage (programmatic):
    from evaluation.report import write_reports
    write_reports(scores_path, conversations_jsonl_path, reports_dir)

The --report flag in __main__.py calls this after run_evaluation completes.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from evaluation.evaluators.llm_judge import RUBRIC_KEYS
from evaluation.evaluators.llm_judge_bea import _AXIS_GROUPS_BEA
from evaluation.evaluators.llm_judge_bea import RUBRIC_KEYS_BEA as _RUBRIC_KEYS_BEA

logger = logging.getLogger(__name__)

_RUBRIC_LABELS = {
    "answer_correctness": "Answer correctness",
    "scaffolding_quality": "Scaffolding quality",
    "student_realism": "Student realism",
    "phrasing_naturalness": "Phrasing naturalness",
    "overall": "Overall",
}

# Workshop axis groupings (display only — no code depends on these).
_AXIS_GROUPS = {
    "Naturalness": ("phrasing_naturalness", "student_realism"),
    "Tutoring effectiveness": ("answer_correctness", "scaffolding_quality"),
    "Engagement / holistic": ("overall",),
}

_RUBRIC_LABELS_BEA = {
    "confusion_identification": "Confusion ID",
    "confusion_location": "Confusion location",
    "answer_withheld": "Answer withheld",
    "guidance_quality": "Guidance quality",
    "actionability": "Actionability",
    "coherence": "Coherence",
    "tutor_tone": "Tutor tone",
    "human_likeness": "Human-likeness",
}

_SEP = "-" * 72
_THICK = "=" * 72


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _format_transcript(segments: list[dict]) -> str:
    rows = []
    for s in sorted(segments, key=lambda x: float(x.get("start_time", 0.0))):
        t = float(s.get("start_time", 0.0))
        role = s.get("role", "?").upper()
        text = (s.get("text", "") or "").strip()
        rows.append(f"  [{t:7.2f}s] {role}: {text}")
    return "\n".join(rows) if rows else "  (no segments)"


def _format_rubric(scores: dict, axis_groups: dict, labels: dict) -> str:
    lines = []
    for axis_name, keys in axis_groups.items():
        lines.append(f"  {axis_name}")
        for k in keys:
            entry = scores.get(k)
            if entry is None:
                lines.append(f"    {labels.get(k, k):<26} --  (missing)")
                continue
            score = entry.get("score", "?")
            rationale = entry.get("rationale", "").strip()
            lines.append(f"    {labels.get(k, k):<26} {score}/5")
            if rationale:
                lines.append(f"      {rationale}")
    return "\n".join(lines)


def _format_stats(ev_results: dict) -> str:
    stats = ev_results.get("stats", {})
    if not stats or stats.get("status") == "error":
        return ""
    lines = []
    for k, v in stats.items():
        if k == "status":
            continue
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _format_turn_taking(ev_results: dict) -> str:
    tt = ev_results.get("turn_taking", {})
    if not tt or tt.get("status") == "error":
        return ""
    lines = []
    for k, v in tt.items():
        if k == "status":
            continue
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.3f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _write_conv_report(
    conv: dict,
    score_record: dict | None,
    idx: int,
    out_path: Path,
) -> None:
    pid = conv.get("pid", "?")
    attempt = conv.get("attempt_index", 0)
    question = conv.get("question", "").strip()
    answer = str(conv.get("answer", "")).strip()
    duration = conv.get("duration", None)
    t_backend = conv.get("tutor_backend", "?")
    s_backend = conv.get("student_backend", "?")
    t_voice = conv.get("tutor_voice", "?")
    s_voice = conv.get("student_voice", "?")
    segments = conv.get("segments", [])

    ev_results = score_record.get("evaluator_results", {}) if score_record else {}
    judge = ev_results.get("llm_judge", {})
    judge_scores = judge.get("scores", {})
    mean_score = judge.get("mean_component_score")
    judge_model = judge.get("model", "?")
    judge_status = judge.get("status", "missing")

    lines: list[str] = []
    lines.append(_THICK)
    lines.append(f"Conversation {idx:03d}   pid={pid}   attempt={attempt}")
    lines.append(_THICK)
    lines.append(
        f"Backends  tutor={t_backend} ({t_voice})   student={s_backend} ({s_voice})"
    )
    if duration is not None:
        lines.append(f"Duration  {duration:.1f}s")

    lines.append("")
    lines.append("QUESTION")
    lines.append(_SEP)
    for line in question.splitlines():
        lines.append(f"  {line}")
    lines.append("")
    lines.append(f"Ground-truth answer: {answer}")

    lines.append("")
    lines.append("TRANSCRIPT")
    lines.append(_SEP)
    lines.append(_format_transcript(segments))

    lines.append("")
    lines.append("LLM JUDGE (naive)")
    lines.append(_SEP)
    if judge_status not in ("ok", "missing"):
        lines.append(f"  Status: {judge_status}")
        if "error" in judge:
            lines.append(f"  Error: {judge['error']}")
    else:
        lines.append(
            f"  Model: {judge_model}   Mean component score: {mean_score if mean_score is not None else '--'}/5"
        )
        lines.append("")
        lines.append(_format_rubric(judge_scores, _AXIS_GROUPS, _RUBRIC_LABELS))

    judge_bea = ev_results.get("llm_judge_bea", {})
    if judge_bea:
        bea_scores = judge_bea.get("scores", {})
        bea_mean = judge_bea.get("mean_component_score")
        bea_model = judge_bea.get("model", "?")
        bea_status = judge_bea.get("status", "missing")
        lines.append("")
        lines.append("LLM JUDGE (BEA 2025 — Maurya et al., NAACL 2025)")
        lines.append(_SEP)
        if bea_status not in ("ok", "missing"):
            lines.append(f"  Status: {bea_status}")
            if "error" in judge_bea:
                lines.append(f"  Error: {judge_bea['error']}")
        else:
            lines.append(
                f"  Model: {bea_model}   Mean component score: {bea_mean if bea_mean is not None else '--'}/5"
            )
            lines.append("")
            lines.append(
                _format_rubric(bea_scores, _AXIS_GROUPS_BEA, _RUBRIC_LABELS_BEA)
            )

    stats_text = _format_stats(ev_results)
    if stats_text:
        lines.append("")
        lines.append("STATS")
        lines.append(_SEP)
        lines.append(stats_text)

    tt_text = _format_turn_taking(ev_results)
    if tt_text:
        lines.append("")
        lines.append("TURN-TAKING")
        lines.append(_SEP)
        lines.append(tt_text)

    lines.append("")
    out_path.write_text("\n".join(lines))


def _write_axis_block(
    lines: list,
    axis_groups: dict,
    rubric_by_key: dict[str, list[int]],
    labels: dict,
    rubric_keys: tuple,
) -> None:
    lines.append("Axis breakdown")
    lines.append(_SEP)
    for axis_name, keys in axis_groups.items():
        axis_scores: list[float] = []
        for k in keys:
            axis_scores.extend(rubric_by_key.get(k, []))
        if not axis_scores:
            lines.append(f"  {axis_name:<30} -- (no data)")
            continue
        mu = statistics.mean(axis_scores)
        med = statistics.median(axis_scores)
        sd = statistics.stdev(axis_scores) if len(axis_scores) > 1 else 0.0
        lines.append(
            f"  {axis_name:<30}  mean={mu:.2f}  median={med:.1f}  sd={sd:.2f}"
            f"  n={len(axis_scores)}"
        )

    lines.append("")
    lines.append("Per-criterion breakdown")
    lines.append(_SEP)
    for k in rubric_keys:
        label = labels.get(k, k)
        vals = rubric_by_key.get(k, [])
        if not vals:
            lines.append(f"  {label:<26}  -- (no data)")
            continue
        mu = statistics.mean(vals)
        med = statistics.median(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        dist = {i: vals.count(i) for i in range(6)}
        dist_str = "  ".join(f"{i}:{dist[i]}" for i in range(6))
        lines.append(f"  {label:<26}  mean={mu:.2f}  median={med:.1f}  sd={sd:.2f}")
        lines.append(f"    score distribution (0–5): {dist_str}")


def _write_summary(
    all_scores: list[dict],
    rubric_by_key: dict[str, list[int]],
    rubric_by_key_bea: dict[str, list[int]],
    out_path: Path,
) -> None:
    n = len(all_scores)
    lines: list[str] = []
    lines.append(_THICK)
    lines.append(f"EVALUATION SUMMARY   ({n} conversations)")
    lines.append(_THICK)

    lines.append("")
    lines.append("NAIVE JUDGE")
    lines.append(_SEP)
    _write_axis_block(lines, _AXIS_GROUPS, rubric_by_key, _RUBRIC_LABELS, RUBRIC_KEYS)

    lines.append("")
    lines.append("BEA 2025 JUDGE  (Maurya et al., NAACL 2025)")
    lines.append(_SEP)
    _write_axis_block(
        lines, _AXIS_GROUPS_BEA, rubric_by_key_bea, _RUBRIC_LABELS_BEA, _RUBRIC_KEYS_BEA
    )

    lines.append("")
    out_path.write_text("\n".join(lines))


def write_reports(
    scores_path: Path,
    conversations_jsonl_path: Path,
    reports_dir: Path,
    *,
    conv_index: int | None = None,
) -> None:
    """Write per-dialog .txt reports and a summary.txt into reports_dir."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    score_records = _load_jsonl(scores_path)
    scores_by_idx = {int(r.get("conv_index", -1)): r for r in score_records}

    conversations = _load_jsonl(conversations_jsonl_path)

    rubric_by_key: dict[str, list[int]] = {k: [] for k in RUBRIC_KEYS}
    rubric_by_key_bea: dict[str, list[int]] = {k: [] for k in _RUBRIC_KEYS_BEA}
    written = 0

    for pos, conv in enumerate(conversations):
        idx = int(conv.get("conv_index", pos))
        if conv_index is not None and idx != conv_index:
            continue
        score_record = scores_by_idx.get(idx)
        out_path = reports_dir / f"conv_{idx:03d}.txt"
        _write_conv_report(conv, score_record, idx, out_path)
        written += 1

        if score_record:
            ev = score_record.get("evaluator_results", {})
            judge = ev.get("llm_judge", {})
            if judge.get("status") == "ok":
                for k in RUBRIC_KEYS:
                    entry = judge.get("scores", {}).get(k)
                    if isinstance(entry, dict) and isinstance(entry.get("score"), int):
                        rubric_by_key[k].append(entry["score"])
            judge_bea = ev.get("llm_judge_bea", {})
            if judge_bea.get("status") == "ok":
                for k in _RUBRIC_KEYS_BEA:
                    entry = judge_bea.get("scores", {}).get(k)
                    if isinstance(entry, dict) and isinstance(entry.get("score"), int):
                        rubric_by_key_bea[k].append(entry["score"])

    if conv_index is None:
        summary_path = reports_dir / "summary.txt"
        _write_summary(score_records, rubric_by_key, rubric_by_key_bea, summary_path)

    logger.info("Wrote %d report(s) + summary to %s", written, reports_dir)
