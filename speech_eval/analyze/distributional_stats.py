"""Distributional turn-taking realism: distance of each system's timing
distributions from a human reference.

The VAP responder / cue readouts say whether a system makes the *right* call at
each moment. This module asks the complementary question: does the system's
turn-taking *look human in aggregate*? It compares the **distribution** of
floor-transfer offsets (and gaps / overlaps / pauses / turn durations) against a
human-human reference corpus scored through the identical vap pipeline (e.g. the
HCRC Map Task ``maptask_ref`` run), via three distances:

* **KS** — Kolmogorov-Smirnov statistic (max CDF gap), shape/location sensitive.
* **EMD** — Wasserstein-1 distance, in **seconds** (the interpretable headline:
  "this system's gaps run ~180 ms longer than human").
* **JS** — Jensen-Shannon distance on shared-bin histograms, symmetric, ∈ [0, 1].

Each carries a percentile bootstrap CI (resampling the system distribution, the
reference held fixed as the anchor). The per-conversation distribution lists are
produced by ``vap_events.compute_distributions`` and stored in each run's
``vap.jsonl`` ``distributions`` block; this module pools them per run.

A system being close on KS/EMD/JS is *necessary but not sufficient* for human-
like turn-taking (a model can match the marginal histogram while making locally
wrong calls) — read it alongside the responder 2×2 in ``vap_stats``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from speech_eval.vap_events import FTO_CLIP_S

logger = logging.getLogger(__name__)

# The metrics from the vap.jsonl "distributions" block, in report order.
DIST_METRICS = ("fto_s", "gap_s", "overlap_s", "within_pause_s", "turn_dur_s")

# Per-metric support (seconds) the distances are computed over. FTO is signed
# (negative = overlap); the rest are non-negative durations. Clipping keeps the
# tail (backchannels read as huge overlaps, stalls as huge gaps, a monologue as
# a 200 s "turn") from dominating EMD/JS. Raw unclipped values stay in vap.jsonl.
DIST_CLIP: dict[str, tuple[float, float]] = {
    "fto_s": (-FTO_CLIP_S, FTO_CLIP_S),
    "gap_s": (0.0, FTO_CLIP_S),
    "overlap_s": (0.0, FTO_CLIP_S),
    "within_pause_s": (0.0, FTO_CLIP_S),
    "turn_dur_s": (0.0, 20.0),
}

# Human-readable labels for the report / plots.
DIST_LABELS: dict[str, str] = {
    "fto_s": "Floor-transfer offset",
    "gap_s": "Between-turn gap",
    "overlap_s": "Overlap",
    "within_pause_s": "Within-turn pause",
    "turn_dur_s": "Turn duration",
}

N_BOOTSTRAP = 1000
JS_BINS = 40
_RESULT_COLUMNS = [
    "system",
    "metric",
    "n_system",
    "n_reference",
    "ks",
    "ks_lo",
    "ks_hi",
    "emd",
    "emd_lo",
    "emd_hi",
    "js",
    "js_lo",
    "js_hi",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_distributions(input_root: Path, run: str) -> dict[str, np.ndarray]:
    """Pool a run's per-conversation ``distributions`` lists into one array per
    metric. Returns empty arrays if the run has no ``vap.jsonl`` or no block."""
    path = input_root / run / "vap.jsonl"
    pooled: dict[str, list[float]] = {k: [] for k in DIST_METRICS}
    if not path.is_file():
        logger.warning("distributional: no vap.jsonl for run %s (%s)", run, path)
        return {k: np.array([], dtype=float) for k in DIST_METRICS}
    for row in _load_jsonl(path):
        block = row.get("distributions") or {}
        for k in DIST_METRICS:
            pooled[k].extend(block.get(k) or [])
    return {k: np.asarray(v, dtype=float) for k, v in pooled.items()}


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if x.size == 0:
        return x
    return x[(x >= lo) & (x <= hi)]


def _ks(sysv: np.ndarray, refv: np.ndarray) -> float:
    return float(stats.ks_2samp(sysv, refv).statistic)


def _emd(sysv: np.ndarray, refv: np.ndarray) -> float:
    return float(stats.wasserstein_distance(sysv, refv))


def _js(sysv: np.ndarray, refv: np.ndarray, lo: float, hi: float) -> float:
    """JS distance ∈ [0, 1] on shared-edge histograms over [lo, hi]."""
    edges = np.linspace(lo, hi, JS_BINS + 1)
    ps, _ = np.histogram(sysv, bins=edges)
    pr, _ = np.histogram(refv, bins=edges)
    if ps.sum() == 0 or pr.sum() == 0:
        return float("nan")
    # jensenshannon normalizes its inputs; base=2 bounds the distance to [0, 1].
    return float(jensenshannon(ps, pr, base=2.0))


def _bootstrap_ci(
    sysv: np.ndarray,
    refv: np.ndarray,
    fn,
    *,
    n: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile (2.5, 97.5) CI for ``fn(system, reference)``, resampling the
    system distribution with the reference held fixed."""
    if sysv.size < 2 or refv.size < 2 or n <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        resampled = rng.choice(sysv, size=sysv.size, replace=True)
        vals[i] = fn(resampled, refv)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def reference_distances(
    input_root: Path,
    systems: list[str],
    reference: str | None,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Distance of each system's distributions from the reference's.

    One row per (system, metric) with KS / EMD / JS + bootstrap CIs. Returns a
    typed empty frame when there is no reference or no systems to compare (so the
    single-run / no-reference cases never crash a CSV writer)."""
    systems = [s for s in (systems or []) if s != reference]
    if not reference or not systems:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    ref_dists = load_distributions(input_root, reference)
    if all(ref_dists[m].size == 0 for m in DIST_METRICS):
        logger.warning(
            "distributional: reference %s has no distributions; "
            "did you score it with the vap component? skipping distances.",
            reference,
        )
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    rows: list[dict] = []
    for s_i, run in enumerate(systems):
        sys_dists = load_distributions(input_root, run)
        for m_i, metric in enumerate(DIST_METRICS):
            lo, hi = DIST_CLIP[metric]
            sysv = _clip(sys_dists[metric], lo, hi)
            refv = _clip(ref_dists[metric], lo, hi)
            row: dict[str, object] = {
                "system": run,
                "metric": metric,
                "n_system": int(sysv.size),
                "n_reference": int(refv.size),
            }
            if sysv.size < 2 or refv.size < 2:
                for key in ("ks", "emd", "js"):
                    row[key] = float("nan")
                    row[f"{key}_lo"] = float("nan")
                    row[f"{key}_hi"] = float("nan")
            else:
                seed_base = 1000 * (s_i + 1) + m_i
                row["ks"] = _ks(sysv, refv)
                row["ks_lo"], row["ks_hi"] = _bootstrap_ci(
                    sysv, refv, _ks, n=n_bootstrap, seed=seed_base + 1
                )
                row["emd"] = _emd(sysv, refv)
                row["emd_lo"], row["emd_hi"] = _bootstrap_ci(
                    sysv, refv, _emd, n=n_bootstrap, seed=seed_base + 2
                )
                row["js"] = _js(sysv, refv, lo, hi)
                row["js_lo"], row["js_hi"] = _bootstrap_ci(
                    sysv,
                    refv,
                    lambda a, b: _js(a, b, lo, hi),
                    n=n_bootstrap,
                    seed=seed_base + 3,
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)
