"""Load speech_eval JSONL outputs into long-form pandas DataFrames.

Per-run inputs live at ``speech_eval_output/<run>/{audiobox.jsonl,
turntaking_judge.jsonl}``. This module concatenates many runs into two
DataFrames keyed by ``(run, conv_index)`` so the downstream stats / plot
helpers can operate uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("tutor", "student", "mixed")
AUDIOBOX_AXES = ("PQ", "PC", "CE", "CU")
JUDGE_LABELS = ("C", "NA", "I", "BC", "T")
VAP_SCOPES = ("tutor", "student", "pooled")
VAP_CELLS = (
    "appropriate_uptake",
    "missed_yield",
    "false_interruption",
    "appropriate_restraint",
)
VAP_CUE_LABELS = (
    "Early-Yield",
    "Late-Yield",
    "Yield-NotCued",
    "Strong-Hold",
    "Weak-Hold",
    "Hold-Misread",
)


@dataclass
class AnalysisData:
    """Container for everything the stats/plot helpers need."""

    audiobox: pd.DataFrame  # one row per (run, conv_index)
    judge: pd.DataFrame  # one row per (run, conv_index)
    vap: pd.DataFrame  # one row per (run, conv_index) for runs scored by VAP
    runs: list[str]
    input_root: Path


def discover_runs(input_root: Path) -> list[str]:
    """Return all run subdirs of ``input_root`` (excluding ``_*`` like ``_analysis``)."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    return sorted(
        p.name
        for p in input_root.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _audiobox_row(run: str, raw: dict) -> dict:
    out: dict = {"run": run, "conv_index": int(raw["conv_index"])}
    channels = raw["channels"]
    for ch in CHANNELS:
        for ax in AUDIOBOX_AXES:
            out[f"ab_{ch}_{ax}"] = float(channels[ch][ax])
    return out


def _judge_row(run: str, raw: dict) -> dict:
    out: dict = {
        "run": run,
        "conv_index": int(raw["conv_index"]),
        "judge_n_windows": int(
            raw.get("n_windows_processed", len(raw.get("windows", [])))
        ),
    }
    total_frames = 0
    counts: dict[str, int] = {lbl: 0 for lbl in JUDGE_LABELS}
    for w in raw.get("windows", []):
        total_frames += int(w.get("n_frames", 0))
        for lbl, n in w.get("label_counts", {}).items():
            if lbl in counts:
                counts[lbl] += int(n)
    out["judge_n_frames"] = total_frames
    for lbl in JUDGE_LABELS:
        # percentage of frames; NaN if no frames so missing-data doesn't read as 0%
        out[f"judge_pct_{lbl}"] = (
            100.0 * counts[lbl] / total_frames if total_frames else float("nan")
        )
    return out


def _vap_row(run: str, raw: dict) -> dict:
    out: dict = {"run": run, "conv_index": int(raw["conv_index"])}
    ne = raw.get("n_events", {})
    for k in ("turn_end", "intra_pause", "dropped_guard"):
        out[f"vap_n_{k}"] = int(ne.get(k, 0))
    for scope in VAP_SCOPES:
        r = raw["responder"][scope]
        for c in VAP_CELLS:
            out[f"vap_{scope}_{c}"] = int(r.get(c, 0))
        fr = r.get("false_interruption_rate")
        out[f"vap_{scope}_FI_rate"] = float(fr) if fr is not None else float("nan")
        lat = list(r.get("uptake_latency_s") or [])
        out[f"vap_{scope}_uptake_lat"] = lat
        out[f"vap_{scope}_uptake_lat_median"] = (
            float(np.median(lat)) if lat else float("nan")
        )
    for role in ("tutor", "student"):
        cu = raw["cues"][role]
        for lbl in VAP_CUE_LABELS:
            out[f"vap_cue_{role}_{lbl}"] = int(cu.get(lbl, 0))
    return out


def load_run(
    input_root: Path, run: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a single run's audiobox + judge (+ optional vap) outputs."""
    run_dir = input_root / run
    ab_path = run_dir / "audiobox.jsonl"
    judge_path = run_dir / "turntaking_judge.jsonl"
    if not ab_path.is_file():
        raise FileNotFoundError(f"missing {ab_path}")
    if not judge_path.is_file():
        raise FileNotFoundError(f"missing {judge_path}")
    ab_df = pd.DataFrame(_audiobox_row(run, r) for r in _load_jsonl(ab_path))
    judge_df = pd.DataFrame(_judge_row(run, r) for r in _load_jsonl(judge_path))
    vap_path = run_dir / "vap.jsonl"
    if vap_path.is_file():
        vap_df = pd.DataFrame(_vap_row(run, r) for r in _load_jsonl(vap_path))
    else:
        vap_df = pd.DataFrame()
    return ab_df, judge_df, vap_df


def load_all(input_root: Path, runs: list[str]) -> AnalysisData:
    """Load and concatenate many runs."""
    ab_frames: list[pd.DataFrame] = []
    judge_frames: list[pd.DataFrame] = []
    vap_frames: list[pd.DataFrame] = []
    for run in runs:
        ab, judge, vap = load_run(input_root, run)
        ab_frames.append(ab)
        judge_frames.append(judge)
        if not vap.empty:
            vap_frames.append(vap)
    return AnalysisData(
        audiobox=pd.concat(ab_frames, ignore_index=True),
        judge=pd.concat(judge_frames, ignore_index=True),
        vap=(
            pd.concat(vap_frames, ignore_index=True) if vap_frames else pd.DataFrame()
        ),
        runs=list(runs),
        input_root=input_root,
    )
