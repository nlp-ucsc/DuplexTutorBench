"""Statistics over the Audiobox channel × axis scores.

All inputs are the long ``audiobox`` DataFrame from ``loader.AnalysisData``:
columns ``run, conv_index, ab_<channel>_<axis>``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from speech_eval.analyze.loader import AUDIOBOX_AXES, CHANNELS

# Headline axes for the main means/CI panel. PC is uncorrelated with speech
# quality (Audiobox paper §3) so we keep it out of headline figures and
# mention it in prose only.
HEADLINE_AXES = ("PQ", "CE", "CU")


def _ci95_halfwidth(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    se = stats.sem(x, nan_policy="omit")
    return float(se * stats.t.ppf(0.975, df=n - 1))


def system_means_ci(
    df: pd.DataFrame,
    *,
    axes: Iterable[str] = AUDIOBOX_AXES,
    channels: Iterable[str] = CHANNELS,
) -> pd.DataFrame:
    """Long-form table: run, channel, axis, n, mean, ci_half, lo, hi."""
    rows = []
    for run, g in df.groupby("run", sort=False):
        for ch in channels:
            for ax in axes:
                col = f"ab_{ch}_{ax}"
                x = g[col].dropna().to_numpy()
                m = float(np.mean(x)) if len(x) else float("nan")
                h = _ci95_halfwidth(x)
                rows.append(
                    {
                        "run": run,
                        "channel": ch,
                        "axis": ax,
                        "n": int(len(x)),
                        "mean": m,
                        "ci_half": h,
                        "lo": m - h,
                        "hi": m + h,
                    }
                )
    return pd.DataFrame(rows)


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values for a single family.

    Returns adjusted p-values in the original input order. Adjusted p_i =
    max over k<=i of (m-k+1) * p_(k), then clipped to <= 1. Standard form.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adj_sorted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * p_values[idx]
        running = max(running, scaled)
        adj_sorted[rank] = min(running, 1.0)
    out = [0.0] * m
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out


def paired_wilcoxon_table(
    df: pd.DataFrame,
    *,
    channel: str,
    axis: str,
    runs: list[str] | None = None,
) -> pd.DataFrame:
    """Pairwise paired Wilcoxon signed-rank test for one (channel, axis).

    Inner-joins on ``conv_index`` per system pair, so ``n_pairs`` only counts
    conversations present in both runs. With our matched MathVista design
    that's 30 per pair.

    Holm-Bonferroni correction is applied within this single family (one
    call = one family of ``C(len(runs), 2)`` tests).
    """
    col = f"ab_{channel}_{axis}"
    if runs is None:
        runs = list(dict.fromkeys(df["run"].tolist()))

    per_run = {
        r: g.set_index("conv_index")[col].dropna()
        for r, g in df[df["run"].isin(runs)].groupby("run", sort=False)
    }

    rows = []
    for a, b in combinations(runs, 2):
        sa = per_run.get(a)
        sb = per_run.get(b)
        if sa is None or sb is None or sa.empty or sb.empty:
            rows.append(
                {
                    "system_a": a,
                    "system_b": b,
                    "n_pairs": 0,
                    "W": float("nan"),
                    "p_raw": float("nan"),
                }
            )
            continue
        joined = pd.concat([sa, sb], axis=1, join="inner").dropna()
        if len(joined) < 2:
            rows.append(
                {
                    "system_a": a,
                    "system_b": b,
                    "n_pairs": int(len(joined)),
                    "W": float("nan"),
                    "p_raw": float("nan"),
                }
            )
            continue
        x = joined.iloc[:, 0].to_numpy()
        y = joined.iloc[:, 1].to_numpy()
        # zero_method="wilcox" drops zero-diff pairs per the classical recipe.
        # If all diffs are zero, scipy raises; treat that as p=1.
        try:
            res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            W = float(res.statistic)
            p = float(res.pvalue)
        except ValueError:
            W = float("nan")
            p = 1.0
        rows.append(
            {
                "system_a": a,
                "system_b": b,
                "n_pairs": int(len(joined)),
                "W": W,
                "p_raw": p,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        # <2 runs -> no pairs -> empty rows -> no columns. Return a typed empty
        # frame so callers (CSV writers, Holm) can index columns safely.
        return pd.DataFrame(
            columns=[
                "system_a",
                "system_b",
                "n_pairs",
                "W",
                "p_raw",
                "p_holm",
                "significant_at_05",
            ]
        )
    p_raws = out["p_raw"].tolist()
    # Holm needs concrete numbers; treat NaN as 1.0 for ranking purposes but
    # propagate NaN back to the output.
    p_for_holm = [1.0 if (p is None or np.isnan(p)) else p for p in p_raws]
    p_holm = _holm_bonferroni(p_for_holm)
    out["p_holm"] = [
        float("nan") if np.isnan(raw) else adj for raw, adj in zip(p_raws, p_holm)
    ]
    out["significant_at_05"] = out["p_holm"].apply(
        lambda v: bool(v <= 0.05) if not (v is None or np.isnan(v)) else False
    )
    return out


def all_pairwise_tables(
    df: pd.DataFrame,
    *,
    channels: Iterable[str] = CHANNELS,
    axes: Iterable[str] = HEADLINE_AXES,
    runs: list[str] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Return a dict keyed by (channel, axis) of pairwise tables."""
    out = {}
    for ch in channels:
        for ax in axes:
            out[(ch, ax)] = paired_wilcoxon_table(df, channel=ch, axis=ax, runs=runs)
    return out


def axis_correlations(
    df: pd.DataFrame,
    *,
    channel: str = "mixed",
) -> dict[str, pd.DataFrame]:
    """Per-run 4×4 Pearson correlation matrix across the four Audiobox axes."""
    out: dict[str, pd.DataFrame] = {}
    for run, g in df.groupby("run", sort=False):
        cols = [f"ab_{channel}_{ax}" for ax in AUDIOBOX_AXES]
        sub = g[cols].rename(columns={f"ab_{channel}_{ax}": ax for ax in AUDIOBOX_AXES})
        out[run] = sub.corr(method="pearson")
    return out


def tutor_student_gap(df: pd.DataFrame, *, axis: str = "PQ") -> pd.DataFrame:
    """Per-system tutor vs. student means with delta, for the sanity panel."""
    rows = []
    for run, g in df.groupby("run", sort=False):
        t = float(g[f"ab_tutor_{axis}"].mean())
        s = float(g[f"ab_student_{axis}"].mean())
        rows.append(
            {"run": run, "tutor": t, "student": s, "delta_tutor_minus_student": t - s}
        )
    return pd.DataFrame(rows)
