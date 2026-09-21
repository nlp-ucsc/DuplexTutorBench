"""Turn-taking event detection + metric logic for the VAP scorer.

Pure Python + numpy — **no torch**. This is the file a researcher edits to
re-score without re-running the GPU forward pass. It consumes:

* the forced-alignment timeline (``aligned/whisper_mfa.jsonl`` segments, with
  word-level boundaries), and
* the per-conversation VAP frames (``p_now``/``p_future`` at 50 Hz, on the
  original conversation timeline),

and produces three readouts (see ``speech_eval/docs/architecture.md``):

1. **Cue production** — at each speaker turn-end, Early/Late-Yield; at each
   intra-turn pause, Strong/Weak-Hold (the 2023 VAP-evaluator metric).
2. **Responder appropriateness** — a 2x2 of VAP's prediction (yield/hold) x what
   the *other* role actually did (took the floor / stayed silent) within a
   decision window. Headline: false-interruption rate + uptake latency.
3. (the raw 50 Hz trajectories for distributional work are dumped by the remote
   inference script, not here.)

Channel order is fixed ``[tutor=0, student=1]`` (the stereo stacking order in
``combined.wav``); flipping it would flip every ``p_now``/``p_future`` index.

All decision thresholds are module constants below. They default to the natural
0.5 boundary of VAP's normalized 2-class head and are **calibration-tunable**
against a human-human baseline (see follow-up in the plan).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROLES = ("tutor", "student")
ROLE_IDX = {"tutor": 0, "student": 1}


def _other(role: str) -> str:
    return "student" if role == "tutor" else "tutor"


# --- Calibration-tunable constants -------------------------------------------
# Words closer than this merge into one "speech run" (below pause perception).
MERGE_GAP_S = 0.18
# A handoff gap (turn-end) longer than this is a conversational stall, not a
# turn-taking decision.
MAX_GAP_S = 4.0
# Intra-turn pause must last at least this; longer than MAX_GAP_S is a stall.
PAUSE_MIN_S = 0.25
PAUSE_MAX_S = 4.0
# Responder must be silent on this pre-window for the read to be uncontaminated
# (VAP is causal: the responder's future action must be absent from its input).
GUARD_PRE_S = 0.30
# A human-plausible window for "did the responder take the floor promptly",
# inside VAP's 0.6-2.0 s p_future horizon.
W_DECISION_S = 1.5
# Decision boundaries on the normalized 2-class probability.
YIELD_THRESH = 0.5
HOLD_THRESH = 0.5
P_NOW_SHIFT_THRESH = 0.5
# Number of frames (ending at t_evt) to causal-mean when reading VAP. 1 = the
# single nearest frame.
READ_SMOOTH_FRAMES = 1
# Distributional realism (compute_distributions): floor-transfer offsets are
# clipped to +/- this many seconds before the KS/EMD/JS distances are taken, so
# the tail (backchannels read as huge overlaps, conversational stalls read as
# huge gaps) does not dominate. Standard Heldner & Edlund (2010) FTO range. The
# raw unclipped lists are still stored per conversation; clipping happens in the
# analyze distance step.
FTO_CLIP_S = 2.0


def thresholds_dict() -> dict:
    """The active thresholds, embedded per-row so a result is reproducible."""
    return {
        "merge_gap_s": MERGE_GAP_S,
        "max_gap_s": MAX_GAP_S,
        "pause_min_s": PAUSE_MIN_S,
        "pause_max_s": PAUSE_MAX_S,
        "guard_pre_s": GUARD_PRE_S,
        "w_decision_s": W_DECISION_S,
        "yield_thresh": YIELD_THRESH,
        "hold_thresh": HOLD_THRESH,
        "p_now_shift_thresh": P_NOW_SHIFT_THRESH,
        "read_smooth_frames": READ_SMOOTH_FRAMES,
        "fto_clip_s": FTO_CLIP_S,
    }


# --- Distributional realism (timing-only, no VAP frames) ---------------------
DIST_KEYS = ("fto_s", "gap_s", "overlap_s", "within_pause_s", "turn_dur_s")


def compute_distributions(segments: list[dict]) -> dict:
    """Pure-timing turn-taking distributions from the aligned timeline.

    These describe how the conversation *flows* and feed the distributional
    realism metric (KS / Wasserstein-EMD / JS vs a human reference). They use
    only word/turn boundaries — no VAP frames, no causality guard — so they are
    independent of the responder-appropriateness readout in ``detect_and_score``.

    Speech runs are the per-role merged IPUs from ``build_timeline``. Walking the
    onset-sorted stream, each adjacent pair yields either a **floor-transfer
    offset** (FTO, role changes) or an **intra-turn pause** (same role resumes):

    * ``fto_s`` — signed FTO at every role change: positive = a between-turn gap,
      negative = an overlap (the next speaker started before the holder finished).
    * ``gap_s`` — the positive FTOs (hand-off silences).
    * ``overlap_s`` — magnitudes of the negative FTOs (overlap durations).
    * ``within_pause_s`` — same-speaker resume silences.
    * ``turn_dur_s`` — merged speech-run (turn) durations, both roles.

    All values are raw and unclipped; the ``FTO_CLIP_S`` window is applied later,
    at the distance step, so the stored data stays faithful.
    """
    tl = build_timeline(segments)

    turn_dur = [e - s for runs in tl.runs_by_role.values() for (s, e) in runs]
    fto: list[float] = []
    gap: list[float] = []
    overlap: list[float] = []
    within: list[float] = []

    for (_s_i, e_i, role_i), (s_j, _e_j, role_j) in zip(tl.ordered, tl.ordered[1:]):
        delta = s_j - e_i
        if role_i == role_j:
            if delta > 0:  # same speaker resumes -> intra-turn pause
                within.append(delta)
        else:  # floor changes hands -> floor-transfer offset
            fto.append(delta)
            (gap if delta >= 0 else overlap).append(abs(delta))

    return {
        "fto_s": [round(x, 4) for x in fto],
        "gap_s": [round(x, 4) for x in gap],
        "overlap_s": [round(x, 4) for x in overlap],
        "within_pause_s": [round(x, 4) for x in within],
        "turn_dur_s": [round(x, 4) for x in turn_dur],
    }


# --- Aligned-timeline loading ------------------------------------------------
def load_aligned_conversations(aligned_path: Path) -> dict[int, dict]:
    """Map ``conv_index`` -> aligned conversation row."""
    out: dict[int, dict] = {}
    with aligned_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["conv_index"])] = row
    return out


def _segment_word_intervals(seg: dict) -> list[tuple[float, float]]:
    """Word-level [start, end] intervals for a segment (fall back to segment)."""
    words = seg.get("words") or []
    ivals = [
        (float(w["start_time"]), float(w["end_time"]))
        for w in words
        if w.get("start_time") is not None and w.get("end_time") is not None
    ]
    if not ivals:
        ivals = [(float(seg["start_time"]), float(seg["end_time"]))]
    return ivals


def first_onset(segments: list[dict]) -> float | None:
    """Earliest word/segment start across both roles (for the leading trim)."""
    starts = [iv[0] for seg in segments for iv in _segment_word_intervals(seg)]
    return min(starts) if starts else None


def _merge(
    intervals: list[tuple[float, float]], gap: float
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


@dataclass
class Timeline:
    """Per-role merged speech runs + a global onset-sorted view."""

    runs_by_role: dict[str, list[tuple[float, float]]]
    # (start, end, role) sorted by start
    ordered: list[tuple[float, float, str]] = field(default_factory=list)


def build_timeline(segments: list[dict], merge_gap_s: float = MERGE_GAP_S) -> Timeline:
    by_role: dict[str, list[tuple[float, float]]] = {r: [] for r in ROLES}
    for seg in segments:
        role = seg["role"]
        if role not in by_role:
            continue
        by_role[role].extend(_segment_word_intervals(seg))
    runs_by_role = {r: _merge(by_role[r], merge_gap_s) for r in ROLES}
    ordered = sorted(
        [(s, e, r) for r in ROLES for (s, e) in runs_by_role[r]], key=lambda x: x[0]
    )
    return Timeline(runs_by_role=runs_by_role, ordered=ordered)


def _is_silent(runs: list[tuple[float, float]], a: float, b: float) -> bool:
    """True if no run overlaps the open-ish interval [a, b]."""
    for s, e in runs:
        if e > a and s < b:
            return False
    return True


def _first_onset_in(
    runs: list[tuple[float, float]], a: float, b: float
) -> float | None:
    """Earliest run start strictly in (a, b], else None."""
    cand = [s for (s, _e) in runs if a < s <= b]
    return min(cand) if cand else None


# --- VAP frame reading -------------------------------------------------------
def read_vap_at(
    frame_times: np.ndarray,
    p_now: np.ndarray,  # [T, 2] = (tutor, student)
    p_future: np.ndarray,  # [T, 2]
    t: float,
    smooth: int = READ_SMOOTH_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_now_pair, p_future_pair) at time ``t`` (causal mean of ``smooth``)."""
    idx = int(np.argmin(np.abs(frame_times - t)))
    lo = max(0, idx - (smooth - 1))
    pn = p_now[lo : idx + 1].mean(axis=0)
    pf = p_future[lo : idx + 1].mean(axis=0)
    return pn, pf


# --- Cue classification (read the holder's own channel) ----------------------
def classify_turn_end_cue(pn_h: float, pf_h: float) -> str:
    if pf_h >= YIELD_THRESH:
        return "Yield-NotCued"
    return "Early-Yield" if pn_h < P_NOW_SHIFT_THRESH else "Late-Yield"


def classify_pause_cue(pn_h: float, pf_h: float) -> str:
    if pf_h < HOLD_THRESH:
        return "Hold-Misread"
    return "Strong-Hold" if pn_h >= HOLD_THRESH else "Weak-Hold"


CUE_LABELS = (
    "Early-Yield",
    "Late-Yield",
    "Yield-NotCued",
    "Strong-Hold",
    "Weak-Hold",
    "Hold-Misread",
)
CELLS = (
    "appropriate_uptake",
    "missed_yield",
    "false_interruption",
    "appropriate_restraint",
)


def _responder_cell(vap_says_yield: bool, took_floor: bool) -> str:
    if vap_says_yield:
        return "appropriate_uptake" if took_floor else "missed_yield"
    return "false_interruption" if took_floor else "appropriate_restraint"


# --- Main entry --------------------------------------------------------------
def detect_and_score(segments: list[dict], frames: dict) -> dict:
    """Detect events and compute all three readouts for one conversation.

    ``frames`` provides numpy arrays ``frame_times_original`` and the channel
    columns ``p_now0/p_now1/p_future0/p_future1``. Returns a JSON-serializable
    dict (n_events, cues, responder, events).
    """
    frame_times = np.asarray(frames["frame_times_original"], dtype=np.float64)
    p_now = np.stack(
        [np.asarray(frames["p_now0"]), np.asarray(frames["p_now1"])], axis=1
    )
    p_future = np.stack(
        [np.asarray(frames["p_future0"]), np.asarray(frames["p_future1"])], axis=1
    )

    tl = build_timeline(segments)
    ordered = tl.ordered

    cues = {r: {lbl: 0 for lbl in CUE_LABELS} for r in ROLES}
    responder: dict[str, dict] = {
        scope: {c: 0 for c in CELLS} | {"uptake_latency_s": []}
        for scope in ("tutor", "student", "pooled")
    }
    n_events = {"turn_end": 0, "intra_pause": 0, "dropped_guard": 0}
    events: list[dict] = []

    if frame_times.size == 0:
        return _finalize(cues, responder, n_events, events)

    t_lo, t_hi = float(frame_times[0]), float(frame_times[-1])

    for cur, nxt in zip(ordered, ordered[1:]):
        s_i, e_i, role_i = cur
        s_j, _e_j, role_j = nxt
        gap = s_j - e_i
        t_evt = e_i

        if role_i == role_j:
            # Same speaker resumes -> intra-turn pause (holder keeps the floor).
            if not (PAUSE_MIN_S <= gap <= PAUSE_MAX_S):
                continue
            etype = "intra_pause"
            holder = role_i
        else:
            # Floor changes hands -> turn-end (holder yields).
            if not (0.0 <= gap <= MAX_GAP_S):
                continue
            etype = "turn_end"
            holder = role_i

        responder_role = _other(holder)

        # Only score events whose frame is inside the (trimmed) VAP window.
        if not (t_lo <= t_evt <= t_hi):
            continue

        # Causality guard: responder must be silent just before the event.
        if not _is_silent(tl.runs_by_role[responder_role], t_evt - GUARD_PRE_S, t_evt):
            n_events["dropped_guard"] += 1
            continue

        n_events[etype] += 1

        # Responder action within the decision window.
        onset = _first_onset_in(
            tl.runs_by_role[responder_role], t_evt, t_evt + W_DECISION_S
        )
        took_floor = onset is not None
        latency = (onset - t_evt) if took_floor else None

        pn, pf = read_vap_at(frame_times, p_now, p_future, t_evt)
        h = ROLE_IDX[holder]
        o = ROLE_IDX[responder_role]
        pn_h, pf_h = float(pn[h]), float(pf[h])
        pf_o = float(pf[o])

        # Readout 1: cue production (holder's own channel).
        if etype == "turn_end":
            cue_label = classify_turn_end_cue(pn_h, pf_h)
        else:
            cue_label = classify_pause_cue(pn_h, pf_h)
        cues[holder][cue_label] += 1

        # Readout 2: responder 2x2 (responder's channel, p_future).
        vap_says_yield = pf_o > YIELD_THRESH
        cell = _responder_cell(vap_says_yield, took_floor)
        responder[responder_role][cell] += 1
        responder["pooled"][cell] += 1
        if cell == "appropriate_uptake":
            responder[responder_role]["uptake_latency_s"].append(float(latency))
            responder["pooled"]["uptake_latency_s"].append(float(latency))

        events.append(
            {
                "type": etype,
                "t_evt": round(t_evt, 3),
                "holder": holder,
                "responder": responder_role,
                "gap_s": round(gap, 3),
                "p_now": [round(float(pn[0]), 4), round(float(pn[1]), 4)],
                "p_future": [round(float(pf[0]), 4), round(float(pf[1]), 4)],
                "vap_pred": "YIELD" if vap_says_yield else "HOLD",
                "responder_action": "took_floor" if took_floor else "stayed_silent",
                "latency_s": round(latency, 3) if took_floor else None,
                "cue_label": cue_label,
                "cell": cell,
            }
        )

    return _finalize(cues, responder, n_events, events)


def _finalize(cues, responder, n_events, events) -> dict:
    for scope in ("tutor", "student", "pooled"):
        fi = responder[scope]["false_interruption"]
        ar = responder[scope]["appropriate_restraint"]
        denom = fi + ar
        responder[scope]["false_interruption_rate"] = (fi / denom) if denom else None
    return {
        "n_events": n_events,
        "cues": cues,
        "responder": responder,
        "events": events,
    }
